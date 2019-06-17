
import gstate
import model
import updater
import experiment
import extensions
import torch
import torch.optim as optim
from torch import nn
from trainer import Trainer
from torch.utils import data

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    # train arguments
    is_download = True
    lr_trigger = [100, 200, 300, 400]
    sv_trigger = [10 * i for i in range(1, 41)]
    lrs = [0.1, 0.01, 0.001, 0.0001]

    learning_rate = lrs[0]
    decay = 1e-4
    max_epoch = sv_trigger[-1]
    size_train = 100
    size_test = 100

    # prepare dataset
    transforms_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))
                                        ])
    transforms_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))
                                        ]) 
    trainset = datasets.CIFAR10('../../data', train=True, download=is_download, transform=transforms_train)
    testset = datasets.CIFAR10('../../data', train=False, download=is_download, transform=transforms_test)
    trainloader = data.DataLoader(trainset, batch_size=size_train, shuffle=True, num_workers=8)
    testloader = data.DataLoader(testset, batch_size=size_test, shuffle=False, num_workers=8)

    # model
    predictor = model.ResNet18()
    # net
    net = experiment.E_basic(predictor)
    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater)                                          

    print('trainer completed')

    trainer.headtrain(extensions.basic_load, args.resume_epoch, 'accuracy')

    trainer.headepoch(extensions.drop_lr, lr_trigger, lrs)

    # set tailepoch extensions, the work to do at the end of each epoch

    trainer.tailepoch(extensions.test, testloader, use_cuda, supervised=True)
    trainer.tailepoch(extensions.report_log)
    trainer.tailepoch(extensions.print_log)
    trainer.tailepoch(extensions.save_log)
    trainer.tailepoch(extensions.gs_best, 'accuracy')
    trainer.tailepoch(extensions.save_best, 'accuracy')

	trainer.tailtrain(extensions.print_best, 'accuracy')

    # run trainer
    trainer.run()

