"""
Create train, valid, test iterators for CIFAR-10.
Train set size: 45000
Val set size: 5000
Test set size: 10000
"""

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_train_valid_loader(batch_size, augment, val_seed, val_size=0.0, num_workers=4, pin_memory=False, contrastive=0, **kwargs):

    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize,
        ])

    # load the dataset
    data_dir = kwargs['root']
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size, num_workers=4, pin_memory=False, size=32,sample_size=1000,**kwargs):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    # size = 224
    torch.manual_seed(1)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(size, padding=4),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize,
    ])

    data_dir = kwargs['root']
    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    num_train = len(dataset)
    if (num_train >= sample_size):
        indices = list(range(num_train))
        split = sample_size
        np.random.seed(1)
        np.random.shuffle(indices)
        valid_idx = indices[:split]
        dataset = Subset(dataset, valid_idx)
        
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=pin_memory,
    )

    return data_loader


if __name__ == '__main__':
    # dataloader = get_test_loader(32, root="../../data")
    test_loader = get_test_loader(root="../../data/", batch_size=32,sample_size=100)
    for i in  range(10):
        print(i)
        for x,y in test_loader:
            print(x.mean())
        