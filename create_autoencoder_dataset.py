import gstate
import dataset
import model
import updater
import experiment
import extensions
import torch
from torch.utils import data
import torch.optim as optim
import numpy

import pickle
from torch.autograd import Variable

import model
# create e1 e2 dataset 
if __name__ == '__main__':


    trainloader = dataset.cifar10_train_loader(is_normal=False)
    encoder = model.WAE_Encoder()
    decoder = model.WAE_Decoder()
    # load WAE net
    net = experiment.E_WAE(encoder, decoder)
    checkpoint = torch.load('epoch_400.t7', map_location='cpu')
    net.load_state_dict(checkpoint['experiment'])


    use_cuda = torch.cuda.is_available()

    batch_xs = torch.zeros(50000, 3072)
    batch_ts = torch.zeros(50000)

    # use cuda
    if use_cuda:
        net.cuda()
        batch_xs = batch_xs.cuda()
        batch_ts = batch_ts.cuda()

    # decode data augment images
    for i, (x, t) in enumerate(trainloader):
        if use_cuda:
            x = x.cuda()
        x = Variable(x)
        z_tilde = net.encoder(x)
        x_recon = net.decoder(z_tilde)
        batch_xs[i * 100: (i + 1) * 100] = (x_recon.data * 255).view(100, 3072).byte()
        batch_ts[i * 100: (i + 1) * 100] = t.data

    batch_xs = numpy.uint8(batch_xs.cpu().numpy())
    batch_ts = batch_ts.cpu().numpy()
    batch_ts = batch_ts.tolist()

    # save data_batch from 6 to 10
    for i in range(6, 11):
        file_name = '../../data/cifar-10-batches-py/data_batch_{}'.format(i)
        batch = {}
        batch[b'data'] = batch_xs[(i - 6) * 10000: (i - 5) * 10000]
        batch[b'labels'] = batch_ts[(i - 6) * 10000: (i - 5) * 10000]
        with open(file_name, 'wb') as f:
            pickle.dump(batch, f)
        print('{} completed'.format(file_name))
        
