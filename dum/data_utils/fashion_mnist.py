"""
Fashion-MNIST used as an OOD dataset.
"""

import torch
from torchvision import datasets, transforms


def get_loaders(batch_size, train=False, num_workers=4, pin_memory=True, **kwargs):

    # define transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,),)])

    # load the dataset
    data_dir = kwargs['root']

    dataset = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform,)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
    )

    return loader
