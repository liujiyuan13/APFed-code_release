import torch.nn as nn
import torch.nn.functional as F


class EncNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DecNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return x


class TaskNet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.fc1 = nn.Linear(d_in, 256)
        self.fc2 = nn.Linear(256, d_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Proj_Head(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.fc = nn.Linear(d_in,  d_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
