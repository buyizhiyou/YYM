#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vmap_demo.py
@Time    :   2023/11/16 21:24:19
@Author  :   shiqing 
@Version :   Cinnamoroll V1
'''

#https://pytorch.org/functorch/0.2.1/notebooks/ensembling.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
torch.manual_seed(0)

# Here's a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


device = 'cuda'
num_models = 10

data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)

models = [SimpleMLP().to(device) for _ in range(num_models)]

#different minibatch for each model
minibatches = data[:num_models]
predictions_diff_minibatch_loop = [model(minibatch) for model, minibatch in zip(models, minibatches)]

#Same minibatch
minibatch = data[0]
predictions2 = [model(minibatch) for model in models]


#Using vmap to vectorize the ensemble
from functorch import combine_state_for_ensemble

fmodel, params, buffers = combine_state_for_ensemble(models)
[p.requires_grad_() for p in params]

print([p.size(0) for p in params]) # show the leading 'num_models' dimension

assert minibatches.shape == (num_models, 64, 1, 28, 28) # verify minibatch has leading dimension of size 'num_models'

from functorch import vmap

predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# verify the vmap predictions match the 
assert torch.allclose(predictions1_vmap, torch.stack(predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)

predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)

assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-3, rtol=1e-5)

from torch.utils.benchmark import Timer
without_vmap = Timer(
    stmt="[model(minibatch) for model, minibatch in zip(models, minibatches)]",
    globals=globals())
with_vmap = Timer(
    stmt="vmap(fmodel)(params, buffers, minibatches)",
    globals=globals())
print(f'Predictions without vmap {without_vmap.timeit(100)}')
print(f'Predictions with vmap {with_vmap.timeit(100)}')