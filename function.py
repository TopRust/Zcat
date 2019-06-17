import torch

def accuracy(y, t):
    _, pred_y = torch.max(y.data, 1)
    correct = pred_y.eq(t.data).cpu().sum().float()
    accuracy = correct / y.size(0)
    return accuracy.item()
