# Modules

## Dataset

you should keep dataset and loader in dataset.py

##Model

you should keep models in model.py

## Gstate

gstate has global variables. You can use it to avoid unnecessary parameters, and save model and load model in an much easy method.

## Experiment

you should keep experiment task in experiment.py. Experiment is the whole forward process of the experiment, and return loss. The example as following.

```python
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
```

## Updater

If you have the special requirments different optimizers update the net, you can use updater keep some optimizers. The updater has the same interface as optim.

## Trainer

Trainer is the encapsulation class which can extend extra functions. 

## Extensions

extensions.py contains many functions. The functions as following.

- test

- report_log

- print_log

- save_log

- drop_lr

- gs_best

- save_best

- save_trigger

- print_best

- basic_load

## Train

combine all trainning processes

