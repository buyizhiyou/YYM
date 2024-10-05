"""
FastMNIST taken from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
"""
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

class FastMNIST(MNIST):
    def __init__(self,device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.repeat(3,1,1)
        img = transforms.Resize((32, 32))(img)

        return img, target


def create_MNIST_dataset(device):

    train_dataset = FastMNIST(device,"data", train=True, download=True)
    test_dataset = FastMNIST(device,"data", train=False, download=True)

    return train_dataset, test_dataset
