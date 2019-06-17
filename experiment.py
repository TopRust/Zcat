# gstate hold global variable
import gstate
import extensions
import function as F
import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
import numpy
from torchvision import transforms

class E_basic(nn.Module):
    def __init__(self, predictor):
        super(E_basic, self).__init__()
        self.predictor = predictor
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t):
        y = self.predictor(x)
        loss = nn.CrossEntropyLoss()(y, t)
        accuracy = F.accuracy(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)
        return loss
