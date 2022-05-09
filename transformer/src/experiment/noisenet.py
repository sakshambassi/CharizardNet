import torch
# import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


learning_rate = 0.01
momentum = 0.5
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


class NoiseNet_y1(nn.Module):
    def __init__(self):
        super(NoiseNet_y1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.9)
        self.fc1 = nn.Linear(1440, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return F.log_softmax(x)

class NoiseNet_y0(nn.Module):
    def __init__(self):
        super(NoiseNet_y0, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.9)
        self.fc1 = nn.Linear(1690, 60)
        self.fc2 = nn.Linear(60, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.dropout(x)
        return F.log_softmax(x)
