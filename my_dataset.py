import os
import cv2
import torch
import torch.utils.data as Data
def my_dataloader():
    imgs = []
    labels = []
    path = '../badface'
    for i in os.listdir(path):
        if i == '.DS_Store':continue
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img,(128,128))
        label = torch.tensor(0)
        imgs.append(img)
        labels.append(label)
    path = '../goodface'
    for i in os.listdir(path):
        if i == '.DS_Store':continue
        img = cv2.imread(os.path.join(path, i))
        img = cv2.resize(img,(128,128))
        label = torch.tensor(1)
        imgs.append(img)
        labels.append(label)
    dataset = Data.TensorDataset(torch.as_tensor(imgs).to(torch.float32),torch.as_tensor(labels))
    traindataset, testdataset = torch.utils.data.random_split(dataset, [len(labels)-200, 200])

    
    testdataloader = Data.DataLoader(
        dataset=testdataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
        )
    traindataloader = Data.DataLoader(
        dataset=traindataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
        )
    return traindataloader,testdataloader

if __name__ == '__main__':
    x, y = my_dataloader()
    print(x,y)