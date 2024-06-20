import torch
import numpy as np


def iterate_data_msp(data_loader, model, device):
    confs = []
    m = torch.nn.Softmax(dim=-1).to(device)
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, device, epsilon, temper):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = x.to(device)
        x.required_grad = True
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = (torch.LongTensor(maxIndexTemp).to(device))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model((tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))

    return np.array(confs)


def iterate_data_energy(data_loader, model, device, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_gradnorm(data_loader, model, device, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for b, (x, y) in enumerate(data_loader):
        inputs = x.to(device)
        inputs.requires_grad = True

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).to(device)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        layer_grad = model.head.conv.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)
