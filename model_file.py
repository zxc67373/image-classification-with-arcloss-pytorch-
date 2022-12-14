import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from arcloss import ArcNet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 14*14

            nn.Conv2d(32, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 7*7

            nn.Conv2d(64, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 3*3

        )
        self.feature = nn.Linear(128*16*16, 2)
        # self.output = nn.Linear(2, 10)
        self.arcsoftmax = ArcNet(2, 2)

    def forward(self, x):
        y_conv = self.conv_layer(x)
        y_conv = torch.reshape(y_conv, [-1, 128*16*16])
        y_feature =self.feature(y_conv)

        # print(y_feature.shape)  # torch.Size([100, 2])

        # 在训练的时候，同时训练了Net_model的参数，也训练了Arcsoftmax的参数
        y_output = torch.log(self.arcsoftmax(y_feature))
        # print(y_output.shape)  # torch.Size([100, 10])

        return y_feature, y_output

if __name__ == '__main__':
    net = Net().cuda()
    a = torch.randn(100, 1, 28, 28).cuda()
    net(a)

