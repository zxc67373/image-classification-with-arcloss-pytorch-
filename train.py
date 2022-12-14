import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from my_dataset import my_dataloader
from model_file import Net
import os
import numpy as np
import torch.utils.data as Data
import cv2
# from my_eval import pred_res
from torchmetrics import F1Score, Accuracy, Recall, Precision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    def pred_res(y_label, y_pred, debug=False):
        f1_score = F1Score(task= 'multiclass',num_classes=2, average="weighted")
        precision_score = Precision(task= 'multiclass',num_classes=2, average="weighted")
        recall_score = Recall(task= 'multiclass',num_classes=2, average="weighted")
        accuracy_score = Accuracy(task= 'multiclass',num_classes=2, average="weighted")

        acc = accuracy_score(y_pred, y_label)
        f1 = f1_score(y_pred, y_label)
        pr = precision_score(y_pred, y_label)
        re = recall_score(y_pred, y_label)
        return acc, pr, re, f1

    save_path = "./models/net_arcloss.pth"
    train_loader,test_loader = my_dataloader()
    net = Net().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path,map_location=device))
    else:
        print("No Param")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'

    loss_fn = nn.NLLLoss()
    # optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(),lr=1e-2, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)

    epoch = 0
    while True:
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            x = x.transpose(1, 3)
            y = y.to(device)
            feature, output = net.forward(x)
            # print(feature.shape)  # torch.Size([100, 2])
            # print(output.shape)  # torch.Size([100, 10])

            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("epoch:", epoch, "i:", i, "arcsoftmax_loss:", loss.item())



        torch.save(net.state_dict(), save_path)

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):  # 加验证集
                x = x.to(device)
                x = x.transpose(1, 3)
                y = y.to(device)
                feature, output = net.forward(x)
                loss = loss_fn(output, y)
                output = torch.argmax(output, 1).to(device)
                # print(output.shape)  # torch.Size([100])
                # print(y.shape)  # torch.Size([100])
                if i % 600 == 0:
                    print("epoch:", epoch, "i:", i, "validate_loss:", loss.item())
                print(y,output)
            acc, pr, re, f1 = pred_res(y_label = y.to('cpu'), y_pred = output.to('cpu'))
            print('Precision: {}, Recall: {}, F1-score: {}'.format(pr, re, f1))

        epoch += 1
        if epoch == 100:
            break
