"""Implementation of Lenet in pytorch.
Refernece:
[1] LeCun,  Y.,  Bottou,  L.,  Bengio,  Y.,  & Haffner,  P. (1998).
    Gradient-based  learning  applied  to  document  recognition.
    Proceedings of the IEEE, 86, 2278-2324.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):

    def __init__(self, emb_size, head_size=512):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.hidden(h)
        h = F.relu(h)
        h = self.out(h)
        return h
    
class LeNet(nn.Module):
    def __init__(self, num_classes, temp=1.0, mnist=True, **kwargs):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1 if mnist else 3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.temp = temp
        self.embedding = None  # 对比loss的embedding
        self.feature = None

        self.projection_head = ProjectionHead(84, 256)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        self.embedding = self.projection_head(out)  # 对比loss的embedding
        self.feature = out
        out = self.fc3(out) / self.temp
        return out


def lenet(num_classes=10, temp=1.0, mnist=True, **kwargs):
    return LeNet(num_classes=num_classes, temp=temp, mnist=True, **kwargs)
