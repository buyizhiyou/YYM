"""
Create train, valid, test iterators for CIFAR-10.
Train set size: 45000
Val set size: 5000
Test set size: 10000
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class GaussDataset(Dataset):
    """generate gaussian dataset."""

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        sample = torch.randn((3, 32, 32))
        y = torch.randint(0, 10, (1,))[0]

        return sample, y


def get_test_loader(batch_size, num_workers=4, pin_memory=False, **kwargs):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = GaussDataset()
    num_train = len(dataset)
    print(f"gauss test:{num_train}")
    if (num_train >= 1000):
        indices = list(range(num_train))
        split = 1000
        np.random.seed(1)
        np.random.shuffle(indices)
        valid_idx = indices[:split]
        dataset = Subset(dataset, valid_idx)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


if __name__ == '__main__':
    test_loader = get_test_loader(64)
    for x, y in test_loader:
        print(y.shape)
