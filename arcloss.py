import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcNet(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super(ArcNet, self).__init__()
        # 生成一个隔离带向量，训练这个向量和原来的特征向量分开，达到增加角度的目的
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim), requires_grad=True)
        # print(self.W.shape)  # torch.Size([2, 10])

    def forward(self, feature, m=1, s=10):
        # 对特征维度进行标准化
        x = F.normalize(feature, dim=1)
        # print(x.shape)  # torch.Size([100, 2])
        w = F.normalize(self.W, dim=0)
        # print(w.shape)  # torch.Size([2, 10])

        # s = 64 一般训练人脸的时候用到该超参
        # s = torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        # print(s)  # tensor(31.6228, device='cuda:0', grad_fn=<MulBackward0>)
        # 做L2范数化，将cosa变小，防止acosa梯度爆炸
        cosa = torch.matmul(x, w) / s
        # print(cosa.shape)  # torch.Size([100, 10])

        a = torch.acos(cosa)  # 反三角函数得出的是弧度，而非角度，1弧度=1*180/3.14=50度
        # print(a)  # torch.Size([100, 10])

        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True)
                                     - torch.exp(s * cosa) + torch.exp(s * torch.cos(a + m)))
        # print(arcsoftmax)
        # print(arcsoftmax.shape)  # torch.Size([100, 10])
        '''这里对e的指数cos(a+m)再乘回来， 让指数函数的输出更大，从而使得arcsoftmax输出更小，
        即log_arcsoftmax输出更大。
            这里argsoftmax的概率不为1，小于1，这会导致交叉熵损失看起来很大，且最优点损失也很大。
            将arcsoftmax放在输出层去训练，就变成一个网络去训练
        '''
        # print(torch.sum(arcsoftmax, dim=1))

        # AM_softmax = torch.exp(
        #     s * (torch.cos(a) - m) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True)
        #                               - torch.exp(s * cosa) + torch.exp(s * (torch.cos(a) - m)))
        #
        # )

        return arcsoftmax


if __name__ == '__main__':
    arc = ArcNet(feature_dim=2, cls_dim=10)

    feature = torch.randn(100, 2)
    out = arc(feature)
    # print(feature)  # 原来的特征数据
