import torch
torch.cuda.set_device('cuda:3')
from torchmetrics import F1Score, Accuracy, Recall, Precision
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