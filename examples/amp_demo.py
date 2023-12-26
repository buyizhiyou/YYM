#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   amp_demo.py
@Time    :   2023/11/07 20:47:29
@Author  :   shiqing
@Version :   Cinnamoroll V1
'''

import torch, time, gc

# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(
        torch.cuda.max_memory_allocated()))


##########################################################
# A simple network
# The following sequence of linear layers and ReLUs should show a speedup with mixed precision.
def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()


batch_size = 512  # Try, for example, 128, 256, 513.
in_size = 4096
out_size = 4096
num_layers = 3
num_batches = 50
epochs = 3

# Creates data in default precision.
# The same data is used for both default and mixed precision trials below.
# You don't need to manually change inputs' dtype when enabling mixed precision.
data = [
    torch.randn(batch_size, in_size, device="cuda") for _ in range(num_batches)
]
targets = [
    torch.randn(batch_size, out_size, device="cuda")
    for _ in range(num_batches)
]

loss_fn = torch.nn.MSELoss().cuda()

##########################################################
# Default Precision
# Without ``torch.cuda.amp``, the following simple network executes all ops in default precision (``torch.float32``):
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad(
        )  # set_to_none=True here can modestly improve performance
end_timer_and_print("Default precision:")

##########################################################
# Adding autocast
# ---------------
# Instances of `torch.cuda.amp.autocast <https://pytorch.org/docs/stable/amp.html#autocasting>`_
# serve as context managers that allow regions of your script to run in mixed precision.
for epoch in range(1):  # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under autocast.
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers autocast to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because mse_loss layers autocast to float32.
            assert loss.dtype is torch.float32

        # Exits autocast before backward().
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad(
        )  # set_to_none=True here can modestly improve performance

##########################################################
# Adding GradScaler
scaler = torch.cuda.amp.GradScaler()
for epoch in range(0):  # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(opt)

        # Updates the scale for next iteration.
        scaler.update()

        opt.zero_grad(
        )  # set_to_none=True here can modestly improve performance

##########################################################
#All together: “Automatic Mixed Precision”
use_amp = True
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type='cuda',
                            dtype=torch.float16,
                            enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(
        )  # set_to_none=True here can modestly improve performance
end_timer_and_print("Mixed precision:")

##########################################################
# Inspecting/modifying gradients (e.g., clipping)
# All gradients produced by ``scaler.scale(loss).backward()`` are scaled.  If you wish to modify or inspect
# the parameters' ``.grad`` attributes between ``backward()`` and ``scaler.step(optimizer)``, you should
# unscale them first using `scaler.unscale_(optimizer)
for epoch in range(0):  # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(opt)

        # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
        # You may use the same value for max_norm here as you would without gradient scaling.
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(
        )  # set_to_none=True here can modestly improve performance

##########################################################
# Saving/Resuming
# To save/resume Amp-enabled runs with bitwise accuracy, use
# `scaler.state_dict <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.state_dict>`_ and
# `scaler.load_state_dict <https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.load_state_dict>`_.
# When saving, save the scaler state dict alongside the usual model and optimizer state dicts.
# Do this either at the beginning of an iteration before any forward passes, or at the end of
# an iteration after ``scaler.update()``.
checkpoint = {
    "model": net.state_dict(),
    "optimizer": opt.state_dict(),
    "scaler": scaler.state_dict()
}
# Write checkpoint as desired, e.g.,
# torch.save(checkpoint, "filename")

##########################################################
# When resuming, load the scaler state dict alongside the model and optimizer state dicts.
# Read checkpoint as desired, e.g.,
# dev = torch.cuda.current_device()
# checkpoint = torch.load("filename",
#                         map_location = lambda storage, loc: storage.cuda(dev))
net.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
scaler.load_state_dict(checkpoint["scaler"])
