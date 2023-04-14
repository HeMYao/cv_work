from torch.nn import Module
from torch import nn


# 定义各个层的功能
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        # 池化层
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # 全连接层
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        # 全连接层
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        # 池化层
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        # 池化层
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        # 全连接层
        y = self.fc1(y)
        y = self.relu3(y)
        # 全连接层
        y = self.fc2(y)
        y = self.relu4(y)
        # 全连接层
        y = self.fc3(y)
        y = self.relu5(y)
        return y
