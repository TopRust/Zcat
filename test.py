
import gstate
import dataset
import model
import updater
import experiment
import extensions
import torch
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable

from trainer import Trainer

import argparse


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='e_noda')
parser.add_argument('--task_id', type=str, default='0')
parser.add_argument('--no_augment', dest='is_augment', action='store_false', default=True)
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--wae_z_dim', type=int, default=512)

args = parser.parse_args()

# train arguments
lr_trigger = [50, 100, 200, 400]
sv_trigger = [10 * i for i in range(1, 41)]
lrs = [0.1, 0.01, 0.001, 0.0001]

learning_rate = lrs[0]
decay = 1e-4
max_epoch = sv_trigger[-1]

# process arguments
loader_args = {
                'e_noDA': (True, True, args.is_augment, False),
                'e_autoencoder': (False, True, args.is_augment, False),
                'e_00': (False, False, args.is_augment, True, True),
                'e_01': (False, False, args.is_augment, True, False),
                'e_2': (False, False, args.is_augment, False),
                'e_wae': (False, False, args.is_augment, False, False)
                }
'''
is_download=False,
is_augment=False,
is_normal=False,
is_autoencoder=False,
only_autoencoder=True
'''

gstate.set_value('task_name', args.task_name)
gstate.set_value('task_id', args.task_id)


def e_basic_test():

    testloader = dataset.cifar10_test_loader(*loader_args[args.task_name][:2])

    directory = '../checkpoint/{}/{}'.format(gstate.get('task_name'), gstate.get('task_id'))

    if args.resume_epoch < 0:
        loadpath = '{}/{}.t7'.format(directory, 'best_accuracy')
    else:
        loadpath = '{}/epoch_{}.t7'.format(directory, args.resume_epoch)

    checkpoint = torch.load(loadpath)

    log = open('{}/log'.format(directory), 'r')
    string = log.read()
    log.close()
    gstate.set_value('log', eval(string)[: checkpoint['epoch']])
    gstate.set_value('start_time', gstate.get('start_time') - gstate.get('log')[-1]['elapsed_time'])

    print(gstate.get('log')[-1])

    net = checkpoint['experiment']

    gstate.clear_statics('number', 'loss', 'accuracy')

    net.eval()
    for x, t in testloader:
        if use_cuda:
            x, t = x.cuda(), t.cuda()
        x, t = Variable(x), Variable(t)
        loss = net(x, t)

    print(gstate.get('statics'))

def e_wae_test():
    testloader = dataset.cifar10_test_loader(*loader_args[args.task_name][:2])

    directory = '../checkpoint/{}/{}'.format(gstate.get('task_name'), gstate.get('task_id'))

    if args.resume_epoch < 0:
        loadpath = '{}/{}.t7'.format(directory, 'best_loss')
    else:
        loadpath = '{}/epoch_{}.t7'.format(directory, args.resume_epoch)

    checkpoint = torch.load(loadpath)

    log = open('{}/log'.format(directory), 'r')
    string = log.read()
    log.close()
    gstate.set_value('log', eval(string)[: checkpoint['epoch']])
    gstate.set_value('start_time', gstate.get('start_time') - gstate.get('log')[-1]['elapsed_time'])

    print(gstate.get('log')[-1])
        encoder = model.WAE_Encoder(args.wae_z_dim)
        encoder = model.WAE_Encoder(args.wae_z_dim)
    if args.name == 'e_vae':
        enc_mu = torch.nn.Linear(100, 8)
        enc_log_sigma = torch.nn.Linear(100, 8)

    if args.name == 'e_wae':
        net = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)
    elif args.name == 'e_vae':
        net = experiment.E_VAE(encoder, decoder, enc_mu, enc_log_sigma, use_cuda=use_cuda)

    net.load_state_dict(checkpoint['experiment'])     

    if use_cuda:
        net.cuda()

    net.eval()
    for i, (x, t) in enumerate(testloader):
        if use_cuda:
            x = x.cuda()
        x = Variable(x)
        z_tilde = net.encoder(x)
        x_recon = net.decoder(z_tilde)
        save_image(x_recon[0],'{}/recon_{}.png'.format(directory, i))
        save_image(x[0], '{}/{}.png'.format(directory, i))
    print(gstate.get('statics'))

main ={
        'e_noda': e_basic_test,
        'e_wae': e_wae_test,
        'e_vae': e_wae_test
    
        }
if __name__ == '__main__':

    main[args.task_name]()
