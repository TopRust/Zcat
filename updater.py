import gstate
from torch import nn
# when you need different optimizers to update the net, you can use Update, the function is the same as optim
class Updater(nn.Module):
    def __init__(self, *args):
        super(Updater, self).__init__()
        self.optimizers = args
        self.param_groups = []
        for optim in self.optimizers:
            for param_group in optim.param_groups:
                self.param_groups.append(param_group)
                
    def zero_grad():
        for optimizer in self.optimizers:
            optimizer.zero_grad()
            
    def step():
        for optimizer in self.optimizers:
            optimizer.step()