import torch

def accuracy(correct, total):
    return torch.sum(torch.tensor(correct)).item() /len(total)

def compare(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds==labels)
    