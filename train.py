
import gstate
import dataset
import model
import updater
import experiment
import extensions
import torch
import torch.optim as optim
from torch import nn
from trainer import Trainer
from torch.utils import data

import argparse


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='e_noda')
parser.add_argument('--task_id', type=str, default='0')
parser.add_argument('--no_augment', dest='is_augment', action='store_false', default=True)
parser.add_argument('--no_z_process', dest='z_process', action='store_false', default=True)
parser.add_argument('--no_pre', dest='use_pre', action='store_false', default=True)
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--wae_z_dim', type=int, default=512)
parser.add_argument('--vae_z_dim', type=int, default=512)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--no_train', dest='is_train', action='store_false', default=True)
parser.add_argument('--linear_ae_z_size', default=2048, type=int)
parser.add_argument('--is_multi', default=False, action='store_true', dest='is_multi')
parser.add_argument('--dropout_rate', type=float, default=0.5)

args = parser.parse_args()

# train arguments
lr_trigger = [100, 200, 300, 400]
sv_trigger = [10 * i for i in range(1, 41)]
lrs = [0.1, 0.01, 0.001, 0.0001]
ftlrs = [0.0001, 0.0001, 0.0001, 0.0001]

learning_rate = lrs[0]
decay = 1e-4
max_epoch = sv_trigger[-1]


# process arguments
loader_args = {
                'e_noda': (True, True, args.is_augment, False),
                'e_allmixup': (False, True, False, False),
                'e_00': (False, True, args.is_augment, True, True),
                'e_01': (False, True, args.is_augment, True, False),
                'e_1': (False, False, args.is_augment, False),
                'e_1_zprocess': (False, False, args.is_augment, False),
                'e_2': (False, False, args.is_augment, False),
                'e_61': (False, False, args.is_augment, False),
                'e_2_zprocess': (False, False, args.is_augment, False),
                'e_2_eval': (False, False, args.is_augment, False),
                'e_wae': (False, False, args.is_augment, False, False),
                'e_vae': (False, False, args.is_augment, False, False),
                'e_allpre': (False, False, args.is_augment, False),
                'e_allpre_train': (False, False, args.is_augment, False),
                'e_allpre_vae': (False, False, args.is_augment, False),
                'e_3l_autoencoder': (False, False, args.is_augment, False),
                'e_3l_autoencoder_p': (False, False, args.is_augment, False)

                }

'''
is_download=False,
is_normal=False,
is_augment=False,
is_autoencoder=False,
only_autoencoder=True
'''

gstate.set_value('task_name', args.task_name)
gstate.set_value('task_id', args.task_id)

# task e_00 train.py --task_name e_00 --task_id 0 
# task e_001 train.py --task_name e_00 --task_id 1
# task e_002 train.py --task_name e_00 --task_id 2
# task e_01 train.py --task_name e_01 --task_id 0 
# task e_011 train.py --task_name e_01 --task_id 1
# task e_012 train.py --task_name e_01 --task_id 2
# task e_2 train.py --task_name e_2 --task_id 0 
# task e_21 train.py --task_name e_2 --task_id 1
# task e_22 train.py --task_name e_2 --task_id 2

# task e_00na train.py --task_name e_00 --task_id na0 
# task e_001na train.py --task_name e_00 --task_id na1
# task e_002na train.py --task_name e_00 --task_id na2
# task e_01na train.py --task_name e_01 --task_id na0 
# task e_011na train.py --task_name e_01 --task_id na1
# task e_012na train.py --task_name e_01 --task_id na2
# task e_2na train.py --task_name e_2 --task_id na0 
# task e_21na train.py --task_name e_2 --task_id na1
# task e_22na train.py --task_name e_2 --task_id na2

# task e_allmixup --task_name e_allmixup --task_id 0 
# task e_allmixup1 --task_name e_allmixup --task_id 1 
# task e_allmixup2 --task_name e_allmixup --task_id 2 

def e_basic():
    if args.task_name == 'e_allmixup':
        trainset = dataset.All_Mixup_Dataset(use_cuda=False)
        trainloader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
    else:
        trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    predictor = model.ResNet18()
    if args.task_name == 'e_allmixup':
        net = experiment.E_basic_allmixup(predictor)
    else:
        net = experiment.E_basic(predictor)

    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater)
    return trainer

# task e_wae0 train.py --task_name e_wae --task_id 0 --wae_z_dim 10
# task e_wae1 train.py --task_name e_wae --task_id 1 --wae_z_dim 10
# task e_wae2 train.py --task_name e_wae --task_id 2 --wae_z_dim 10

# task e_wae0 train.py --task_name e_wae --task_id 0
# task e_wae1 train.py --task_name e_wae --task_id 1 
# task e_wae2 train.py --task_name e_wae --task_id 2

# task e_vae0 train.py --task_name e_vae --task_id 0
# task e_vae1 train.py --task_name e_vae --task_id 1
# task e_vae2 train.py --task_name e_vae --task_id 2

# train WAE autoencoder
def e_wae():

    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    if args.task_name == 'e_wae':

        encoder = model.WAE_Encoder(args.wae_z_dim)
        decoder = model.WAE_Decoder(args.wae_z_dim)
        net = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)
    elif args.task_name == 'e_vae':

        encoder = model.WAE_Encoder(args.wae_z_dim)
        decoder = model.WAE_Decoder(args.vae_z_dim)
        enc_mu = torch.nn.Linear(args.wae_z_dim, args.vae_z_dim)
        enc_log_sigma = torch.nn.Linear(args.wae_z_dim, args.vae_z_dim)
        net = experiment.E_VAE(encoder, decoder, enc_mu, enc_log_sigma, args.w, args.scale, use_cuda=use_cuda)

    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater, False)
    return trainer
# 
def e_zlinear():
    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    encoder = model.WAE_Encoder(args.wae_z_dim)
    decoder = model.WAE_Decoder(args.wae_z_dim)
    net = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)

    checkpoint = torch.load('epoch_400.t7', map_location='cpu')
    net.load_state_dict(checkpoint['experiment'])

    if args.z_process:
        z_processer = nn.Sequential(
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    )
    else: z_processer = None

    zlinear = nn.Linear(512, 10)
    if use_cuda:
        net.encoder.cuda()
    net.encoder.eval()
    net = experiment.E_1(net.encoder, z_processer, zlinear)
    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater)
    return trainer
# task e_2na train.py --task_name e_2 --task_id 0na --no_pre --no_z_process --no_augment --wae_z_dim 10
# task e_21na train.py --task_name e_2 --task_id 1na --no_pre --no_z_process --no_augment --wae_z_dim 10
# task e_22na train.py --task_name e_2 --task_id 2na --no_pre --no_z_process --no_augment --wae_z_dim 10

# task e_2 train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 10
# task e_21 train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 10
# task e_22 train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 10

# ---------- dropout
# task e_2dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512
# task e_21dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512
# task e_22dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512
# task e_21dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.1
# task e_211dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.1
# task e_212dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.1
# task e_22dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.2
# task e_221dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.2
# task e_222dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.2
# task e_23dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.3
# task e_231dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.3
# task e_232dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.3
# task e_24dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.4
# task e_241dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.4
# task e_242dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.4
# task e_27dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.7
# task e_271dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.7
# task e_272dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.7
# task e_26dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.6
# task e_261dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.6
# task e_262dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.6
# task e_28dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.8
# task e_281dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.8
# task e_282dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.8
# task e_29dr train.py --task_name e_2 --task_id 0 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.9
# task e_291dr train.py --task_name e_2 --task_id 1 --no_pre --no_z_process --wae_z_dim 512 --dropout_rate 0.9
# task e_292dr train.py --task_name e_2 --task_id 2 --no_pre --no_z_process  --wae_z_dim 512 --dropout_rate 0.9

# task e_610na0.2ch train.py --task_name e_61 --task_id 00.2 --no_pre --no_z_process --no_augment --w 0.2
# task e_611na0.2ch train.py --task_name e_61 --task_id 10.2 --no_pre --no_z_process --no_augment --w 0.2
# task e_612na0.2ch train.py --task_name e_61 --task_id 20.2 --no_pre --no_z_process --no_augment --w 0.2
# task e_610na0.25ch train.py --task_name e_61 --task_id 00.25 --no_pre --no_z_process --no_augment --w 0.25
# task e_611na0.25ch train.py --task_name e_61 --task_id 10.25 --no_pre --no_z_process --no_augment --w 0.25
# task e_612na0.25ch train.py --task_name e_61 --task_id 20.25 --no_pre --no_z_process --no_augment --w 0.25
# task e_610na0.5ch train.py --task_name e_61 --task_id 00.5 --no_pre --no_z_process --no_augment --w 0.5
# task e_611na0.5ch train.py --task_name e_61 --task_id 10.5 --no_pre --no_z_process --no_augment --w 0.5
# task e_612na0.5ch train.py --task_name e_61 --task_id 20.5 --no_pre --no_z_process --no_augment --w 0.5

# task e_610na train.py --task_name e_61 --task_id 0 --no_pre --no_z_process --no_augment
# task e_611na train.py --task_name e_61 --task_id 1 --no_pre --no_z_process --no_augment
# task e_612na train.py --task_name e_61 --task_id 2 --no_pre --no_z_process --no_augment
# task e_610na0.1 train.py --task_name e_61 --task_id 00.1 --no_pre --no_z_process --no_augment --w 0.1
# task e_611na0.1 train.py --task_name e_61 --task_id 10.1 --no_pre --no_z_process --no_augment --w 0.1
# task e_612na0.1 train.py --task_name e_61 --task_id 20.1 --no_pre --no_z_process --no_augment --w 0.1
# task e_610na0.2 train.py --task_name e_61 --task_id 00.2 --no_pre --no_z_process --no_augment --w 0.2
# task e_611na0.2 train.py --task_name e_61 --task_id 10.2 --no_pre --no_z_process --no_augment --w 0.2
# task e_612na0.2 train.py --task_name e_61 --task_id 20.2 --no_pre --no_z_process --no_augment --w 0.2
# task e_610na0.25 train.py --task_name e_61 --task_id 00.25 --no_pre --no_z_process --no_augment --w 0.25
# task e_611na0.25 train.py --task_name e_61 --task_id 10.25 --no_pre --no_z_process --no_augment --w 0.25
# task e_612na0.25 train.py --task_name e_61 --task_id 20.25 --no_pre --no_z_process --no_augment --w 0.25
# task e_610na0.5 train.py --task_name e_61 --task_id 00.5 --no_pre --no_z_process --no_augment --w 0.5
# task e_611na0.5 train.py --task_name e_61 --task_id 10.5 --no_pre --no_z_process --no_augment --w 0.5
# task e_612na0.5 train.py --task_name e_61 --task_id 20.5 --no_pre --no_z_process --no_augment --w 0.5
# task e_610na2.0 train.py --task_name e_61 --task_id 02.0 --no_pre --no_z_process --no_augment --w 2.0
# task e_611na2.0 train.py --task_name e_61 --task_id 12.0 --no_pre --no_z_process --no_augment --w 2.0
# task e_612na2.0 train.py --task_name e_61 --task_id 22.0 --no_pre --no_z_process --no_augment --w 2.0
# task e_610na4.0 train.py --task_name e_61 --task_id 04.0 --no_pre --no_z_process --no_augment --w 4.0
# task e_611na4.0 train.py --task_name e_61 --task_id 14.0 --no_pre --no_z_process --no_augment --w 4.0
# task e_612na4.0 train.py --task_name e_61 --task_id 24.0 --no_pre --no_z_process --no_augment --w 4.0
# task e_610na5.0 train.py --task_name e_61 --task_id 05.0 --no_pre --no_z_process --no_augment --w 5.0
# task e_611na5.0 train.py --task_name e_61 --task_id 15.0 --no_pre --no_z_process --no_augment --w 5.0
# task e_612na5.0 train.py --task_name e_61 --task_id 25.0 --no_pre --no_z_process --no_augment --w 5.0
# task e_610na10.0 train.py --task_name e_61 --task_id 010.0 --no_pre --no_z_process --no_augment --w 10.0
# task e_611na10.0 train.py --task_name e_61 --task_id 110.0 --no_pre --no_z_process --no_augment --w 10.0
# task e_612na10.0 train.py --task_name e_61 --task_id 210.0 --no_pre --no_z_process --no_augment --w 10.0

# task e_610 train.py --task_name e_61 --task_id 0 --no_pre --no_z_process
# task e_611 train.py --task_name e_61 --task_id 1 --no_pre --no_z_process
# task e_612 train.py --task_name e_61 --task_id 2 --no_pre --no_z_process
# task e_6100.1 train.py --task_name e_61 --task_id 00.1 --no_pre --no_z_process --w 0.1
# task e_6110.1 train.py --task_name e_61 --task_id 10.1 --no_pre --no_z_process --w 0.1
# task e_6120.1 train.py --task_name e_61 --task_id 20.1 --no_pre --no_z_process --w 0.1
# task e_6100.2 train.py --task_name e_61 --task_id 00.2 --no_pre --no_z_process --w 0.2
# task e_6110.2 train.py --task_name e_61 --task_id 10.2 --no_pre --no_z_process --w 0.2
# task e_6120.2 train.py --task_name e_61 --task_id 20.2 --no_pre --no_z_process --w 0.2
# task e_6100.25 train.py --task_name e_61 --task_id 00.25 --no_pre --no_z_process --w 0.25
# task e_6110.25 train.py --task_name e_61 --task_id 10.25 --no_pre --no_z_process --w 0.25
# task e_6120.25 train.py --task_name e_61 --task_id 20.25 --no_pre --no_z_process --w 0.25
# task e_6100.5 train.py --task_name e_61 --task_id 00.5 --no_pre --no_z_process --w 0.5
# task e_6110.5 train.py --task_name e_61 --task_id 10.5 --no_pre --no_z_process --w 0.5
# task e_6120.5 train.py --task_name e_61 --task_id 20.5 --no_pre --no_z_process --w 0.5
# task e_6102.0 train.py --task_name e_61 --task_id 02.0 --no_pre --no_z_process --w 2.0
# task e_6112.0 train.py --task_name e_61 --task_id 12.0 --no_pre --no_z_process --w 2.0
# task e_6122.0 train.py --task_name e_61 --task_id 22.0 --no_pre --no_z_process --w 2.0
# task e_6104.0 train.py --task_name e_61 --task_id 04.0 --no_pre --no_z_process --w 4.0
# task e_6114.0 train.py --task_name e_61 --task_id 14.0 --no_pre --no_z_process --w 4.0
# task e_6124.0 train.py --task_name e_61 --task_id 24.0 --no_pre --no_z_process --w 4.0
# task e_6105.0 train.py --task_name e_61 --task_id 05.0 --no_pre --no_z_process --w 5.0
# task e_6115.0 train.py --task_name e_61 --task_id 15.0 --no_pre --no_z_process --w 5.0
# task e_6125.0 train.py --task_name e_61 --task_id 25.0 --no_pre --no_z_process --w 5.0
# task e_61010.0 train.py --task_name e_61 --task_id 010.0 --no_pre --no_z_process --w 10.0
# task e_61110.0 train.py --task_name e_61 --task_id 110.0 --no_pre --no_z_process --w 10.0
# task e_61210.0 train.py --task_name e_61 --task_id 210.0 --no_pre --no_z_process --w 10.0
def e_zcat():

    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    encoder = model.WAE_Encoder(args.wae_z_dim)
    decoder = model.WAE_Decoder(args.wae_z_dim)
    net = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)

    checkpoint = torch.load('../checkpoint/e_wae/1/best_loss.t7', map_location='cpu')
    net.load_state_dict(checkpoint['experiment'])
    if args.z_process:
        z_processer = nn.Sequential(
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    )
    else: z_processer = None
    # zlinear = nn.Sequential(
    #                         nn.Linear(512 + args.wae_z_dim, 2048),
    #                         nn.BatchNorm1d(2048),
    #                         nn.ReLU(),
    #                         nn.Dropout(p=0.5),
    #                         nn.Linear(2048, 1024),
    #                         nn.BatchNorm1d(1024),
    #                         nn.ReLU(),
    #                         nn.Dropout(p=0.5),
    #                         nn.Linear(1024, 10)

    #                         )
    zlinear = nn.Sequential(
                            nn.BatchNorm1d(512 + args.wae_z_dim),
                            nn.ReLU(),
                            nn.Dropout(p=args.dropout_rate),
                            nn.Linear(512 + args.wae_z_dim, 10)

                            )
    predictor = model.ResNet18()

    if args.use_pre:
        directory = '../checkpoint/e_noda/0/best_accuracy.t7'
        checkpoint = torch.load(directory)

        ex = experiment.E_basic(predictor)
        ex.load_state_dict(checkpoint['experiment'])
        predictor = ex.predictor

    if use_cuda:
        net.encoder.cuda()
        predictor.cuda()

    if args.task_name == 'e_61':
        net = experiment.E_61(net.encoder, net.decoder, predictor, zlinear, args.w)
    elif args.task_name == 'e_2':
        net = experiment.E_2(net.encoder, predictor, z_processer, zlinear, args.is_train)
    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater)
    return trainer

# task e_allpre20 train.py --task_name e_allpre --task_id 20 --no_train
# task e_allpre21 train.py --task_name e_allpre --task_id 21 --no_train
# task e_allpre22 train.py --task_name e_allpre --task_id 22 --no_train
# task e_allpre20na train.py --task_name e_allpre --task_id na20 --no_train --no_augment
# task e_allpre21na train.py --task_name e_allpre --task_id na21 --no_train --no_augment
# task e_allpre22na train.py --task_name e_allpre --task_id na22 --no_train --no_augment

# task e_allpre_train20t train.py --task_name e_allpre --task_id t20 
# task e_allpre_train21t train.py --task_name e_allpre --task_id t21 
# task e_allpre_train22t train.py --task_name e_allpre --task_id t22 
# task e_allpre_train20nat train.py --task_name e_allpre --task_id tna20 --no_augment
# task e_allpre_train21nat train.py --task_name e_allpre --task_id tna21 --no_augment
# task e_allpre_train22nat train.py --task_name e_allpre --task_id tna22 --no_augment

# task e_all_train20t train.py --task_name e_allpre --task_id npt20 --no_pre 
# task e_all_train21t train.py --task_name e_allpre --task_id npt21 --no_pre 
# task e_all_train22t train.py --task_name e_allpre --task_id npt22 --no_pre 
# task e_all_train20nat train.py --task_name e_allpre --task_id nptna20 --no_augment --no_pre
# task e_all_train21nat train.py --task_name e_allpre --task_id nptna21 --no_augment --no_pre
# task e_all_train22nat train.py --task_name e_allpre --task_id nptna22 --no_augment --no_pre


# task e_all_train20t1l train.py --task_name e_allpre --task_id 1lnpt20 --no_pre 
# task e_all_train21t1l train.py --task_name e_allpre --task_id 1lnpt21 --no_pre 
# task e_all_train22t1l train.py --task_name e_allpre --task_id 1lnpt22 --no_pre 
# task e_all_train20nat1l train.py --task_name e_allpre --task_id 1lnptna20 --no_augment --no_pre
# task e_all_train21nat1l train.py --task_name e_allpre --task_id 1lnptna21 --no_augment --no_pre
# task e_all_train22nat1l train.py --task_name e_allpre --task_id 1lnptna22 --no_augment --no_pre
# task e_all_train20t1lup train.py --task_name e_allpre --task_id 1lt20
# task e_all_train21t1lup train.py --task_name e_allpre --task_id 1lt21 
# task e_all_train22t1lup train.py --task_name e_allpre --task_id 1lt22 
# task e_all_train20nat1lup train.py --task_name e_allpre --task_id 1ltna20 --no_augment
# task e_all_train21nat1lup train.py --task_name e_allpre --task_id 1ltna21 --no_augment
# task e_all_train22nat1lup train.py --task_name e_allpre --task_id 1ltna22 --no_augment

# ----------
# task e_allpre20 train.py --task_name e_allpre --task_id 20 --no_train --wae_z_dim 10
# task e_allpre21 train.py --task_name e_allpre --task_id 21 --no_train --wae_z_dim 10
# task e_allpre22 train.py --task_name e_allpre --task_id 22 --no_train --wae_z_dim 10
# task e_allpre20na train.py --task_name e_allpre --task_id na20 --no_train --no_augment --wae_z_dim 10
# task e_allpre21na train.py --task_name e_allpre --task_id na21 --no_train --no_augment --wae_z_dim 10
# task e_allpre22na train.py --task_name e_allpre --task_id na22 --no_train --no_augment --wae_z_dim 10

# task e_allpre_train20t train.py --task_name e_allpre --task_id t20  --wae_z_dim 10
# task e_allpre_train21t train.py --task_name e_allpre --task_id t21  --wae_z_dim 10
# task e_allpre_train22t train.py --task_name e_allpre --task_id t22  --wae_z_dim 10
# task e_allpre_train20nat train.py --task_name e_allpre --task_id tna20 --no_augment --wae_z_dim 10
# task e_allpre_train21nat train.py --task_name e_allpre --task_id tna21 --no_augment --wae_z_dim 10
# task e_allpre_train22nat train.py --task_name e_allpre --task_id tna22 --no_augment --wae_z_dim 10

# task e_all_train20t train.py --task_name e_allpre --task_id npt20 --no_pre  --wae_z_dim 10
# task e_all_train21t train.py --task_name e_allpre --task_id npt21 --no_pre  --wae_z_dim 10
# task e_all_train22t train.py --task_name e_allpre --task_id npt22 --no_pre  --wae_z_dim 10
# task e_all_train20nat train.py --task_name e_allpre --task_id nptna20 --no_augment --no_pre --wae_z_dim 10
# task e_all_train21nat train.py --task_name e_allpre --task_id nptna21 --no_augment --no_pre --wae_z_dim 10
# task e_all_train22nat train.py --task_name e_allpre --task_id nptna22 --no_augment --no_pre --wae_z_dim 10



# task e_all_train20t1l train.py --task_name e_allpre --task_id 1lnpt20 --no_pre  --wae_z_dim 10
# task e_all_train21t1l train.py --task_name e_allpre --task_id 1lnpt21 --no_pre  --wae_z_dim 10
# task e_all_train22t1l train.py --task_name e_allpre --task_id 1lnpt22 --no_pre  --wae_z_dim 10
# task e_all_train20nat1l train.py --task_name e_allpre --task_id 1lnptna20 --no_augment --no_pre --wae_z_dim 10
# task e_all_train21nat1l train.py --task_name e_allpre --task_id 1lnptna21 --no_augment --no_pre --wae_z_dim 10
# task e_all_train22nat1l train.py --task_name e_allpre --task_id 1lnptna22 --no_augment --no_pre --wae_z_dim 10
# task e_all_train20t1lup train.py --task_name e_allpre --task_id 1lt20 --wae_z_dim 10
# task e_all_train21t1lup train.py --task_name e_allpre --task_id 1lt21  --wae_z_dim 10
# task e_all_train22t1lup train.py --task_name e_allpre --task_id 1lt22  --wae_z_dim 10
# task e_all_train20nat1lup train.py --task_name e_allpre --task_id 1ltna20 --no_augment --wae_z_dim 10
# task e_all_train21nat1lup train.py --task_name e_allpre --task_id 1ltna21 --no_augment --wae_z_dim 10
# task e_all_train22nat1lup train.py --task_name e_allpre --task_id 1ltna22 --no_augment --wae_z_dim 10
def e_z_allpre():

    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    encoder = model.WAE_Encoder(args.wae_z_dim)
    decoder = model.WAE_Decoder(args.wae_z_dim)
    ex_auto_encoder = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)
 
    checkpoint = torch.load('../checkpoint/e_wae/0/best_loss.t7', map_location='cpu')

    ex_auto_encoder.load_state_dict(checkpoint['experiment'])

    predictor = model.ResNet18()
    if args.use_pre:

        directory = '../checkpoint/e_noda/0/best_accuracy.t7'
        checkpoint = torch.load(directory)

        ex_basic = experiment.E_basic(predictor)
        ex_basic.load_state_dict(checkpoint['experiment'])
        predictor = ex_basic.predictor

    z_processer = nn.Sequential(
                            nn.Linear(512 + args.wae_z_dim, 4096),
                            nn.BatchNorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096, 4096),
                            nn.BatchNorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096, 10)
                            )
    # z_processer = nn.Sequential(
    #                         nn.Linear(512 + args.wae_z_dim, 10)
    #                         )
    if use_cuda:
        ex_auto_encoder.cuda()
        predictor.cuda()
        z_processer.cuda()

    ex_auto_encoder.encoder.eval()

    if not args.is_train:
        predictor.eval()


    net = experiment.E_allpre(ex_auto_encoder.encoder, predictor, z_processer, args.is_train)
    if use_cuda:

        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater)
    return trainer


# task e_allpre_vae0 train.py --task_name e_allpre_vae --task_id 0 --vae_z_dim 10
# task e_allpre_vae1 train.py --task_name e_allpre_vae --task_id 1 --vae_z_dim 10
# task e_allpre_vae2 train.py --task_name e_allpre_vae --task_id 2 --vae_z_dim 10
# task e_allpre_vae0 train.py --task_name e_allpre_vae --task_id 5120 --vae_z_dim 512 --resume_epoch -1
# task e_allpre_vae1 train.py --task_name e_allpre_vae --task_id 5121 --vae_z_dim 512 --resume_epoch -1
# task e_allpre_vae2 train.py --task_name e_allpre_vae --task_id 5122 --vae_z_dim 512 --resume_epoch -1

# task e_allpre_vae0nta train.py --task_name e_allpre_vae --task_id 5120nta --vae_z_dim 512 --no_train
# task e_allpre_vae1nta train.py --task_name e_allpre_vae --task_id 5121nta --vae_z_dim 512 --no_train
# task e_allpre_vae2nta train.py --task_name e_allpre_vae --task_id 5122nta --vae_z_dim 512 --no_train
# task e_allpre_vae0nanta train.py --task_name e_allpre_vae --task_id 5120nanta --vae_z_dim 512 --no_augment --no_train
# task e_allpre_vae1nanta train.py --task_name e_allpre_vae --task_id 5121nanta --vae_z_dim 512 --no_augment --no_train
# task e_allpre_vae2nanta train.py --task_name e_allpre_vae --task_id 5122nanta --vae_z_dim 512 --no_augment --no_train

# task e_allpre_vae0a train.py --task_name e_allpre_vae --task_id 5120a --vae_z_dim 512
# task e_allpre_vae1a train.py --task_name e_allpre_vae --task_id 5121a --vae_z_dim 512
# task e_allpre_vae2a train.py --task_name e_allpre_vae --task_id 5122a --vae_z_dim 512
# task e_allpre_vae0naa train.py --task_name e_allpre_vae --task_id 5120naa --vae_z_dim 512 --no_augment
# task e_allpre_vae1naa train.py --task_name e_allpre_vae --task_id 5121naa --vae_z_dim 512 --no_augment
# task e_allpre_vae2naa train.py --task_name e_allpre_vae --task_id 5122naa --vae_z_dim 512 --no_augment

# task e_allpre_vae0anp train.py --task_name e_allpre_vae --task_id 5120anp --vae_z_dim 512 --no_pre
# task e_allpre_vae1anp train.py --task_name e_allpre_vae --task_id 5121anp --vae_z_dim 512 --no_pre
# task e_allpre_vae2anp train.py --task_name e_allpre_vae --task_id 5122anp --vae_z_dim 512 --no_pre
# task e_allpre_vae0naanp train.py --task_name e_allpre_vae --task_id 5120naanp --vae_z_dim 512 --no_augment --no_pre
# task e_allpre_vae1naanp train.py --task_name e_allpre_vae --task_id 5121naanp --vae_z_dim 512 --no_augment --no_pre
# task e_allpre_vae2naanp train.py --task_name e_allpre_vae --task_id 5122naanp --vae_z_dim 512 --no_augment --no_pre

# task e_allpre_vae0a3 train.py --task_name e_allpre_vae --task_id 35120a --vae_z_dim 512
# task e_allpre_vae1a3 train.py --task_name e_allpre_vae --task_id 35121a --vae_z_dim 512
# task e_allpre_vae2a3 train.py --task_name e_allpre_vae --task_id 35122a --vae_z_dim 512
# task e_allpre_vae0naa3 train.py --task_name e_allpre_vae --task_id 35120naa --vae_z_dim 512 --no_augment
# task e_allpre_vae1naa3 train.py --task_name e_allpre_vae --task_id 35121naa --vae_z_dim 512 --no_augment
# task e_allpre_vae2naa3 train.py --task_name e_allpre_vae --task_id 35122naa --vae_z_dim 512 --no_augment

# task e_allpre_vae0anp3 train.py --task_name e_allpre_vae --task_id 35120anp --vae_z_dim 512 --no_pre
# task e_allpre_vae1anp3 train.py --task_name e_allpre_vae --task_id 35121anp --vae_z_dim 512 --no_pre
# task e_allpre_vae2anp3 train.py --task_name e_allpre_vae --task_id 35122anp --vae_z_dim 512 --no_pre
# task e_allpre_vae0naanp3 train.py --task_name e_allpre_vae --task_id 35120naanp --vae_z_dim 512 --no_augment --no_pre
# task e_allpre_vae1naanp3 train.py --task_name e_allpre_vae --task_id 35121naanp --vae_z_dim 512 --no_augment --no_pre
# task e_allpre_vae2naanp3 train.py --task_name e_allpre_vae --task_id 35122naanp --vae_z_dim 512 --no_augment --no_pre

def e_z_allpre_vae():

    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    encoder = model.WAE_Encoder(args.wae_z_dim)
    decoder = model.WAE_Decoder(args.vae_z_dim)
    enc_mu = torch.nn.Linear(args.wae_z_dim, args.vae_z_dim)
    enc_log_sigma = torch.nn.Linear(args.wae_z_dim, args.vae_z_dim)
    ex_auto_encoder = experiment.E_VAE(encoder, decoder, enc_mu, enc_log_sigma, args.w, args.scale, use_cuda=use_cuda)

    checkpoint = torch.load('../checkpoint/e_vae/0/best_loss.t7', map_location='cpu')

    ex_auto_encoder.load_state_dict(checkpoint['experiment'])

    predictor = model.ResNet18()
    if args.use_pre:
        directory = '../checkpoint/e_noda/0/best_accuracy.t7'
        checkpoint = torch.load(directory)

        ex_basic = experiment.E_basic(predictor)
        ex_basic.load_state_dict(checkpoint['experiment'])
        predictor = ex_basic.predictor

    if use_cuda:
        ex_auto_encoder.cuda()
        predictor.cuda()
    ex_auto_encoder.eval()
    if not args.is_train:
        predictor.eval()

    z_processer = nn.Sequential(
                            nn.Linear(512 + 512, 4096),
                            nn.BatchNorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096, 4096),
                            nn.BatchNorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096, 10)
                            )
    # z_processer = nn.Sequential(
    #                         nn.Linear(512 + 512, 10)
    #                         )
    net = experiment.E_allpre_vae(ex_auto_encoder, predictor, z_processer, args.is_train)
    if args.is_multi:
        net = experiment.E_allpre_multi_vae(ex_auto_encoder, predictor, z_processer)
    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater)
    return trainer
# task e_3l_autoencoder train.py --task_name e_3l_autoencoder --task_id 0 
# task e_3l_autoencoder1 train.py --task_name e_3l_autoencoder --task_id 1
# task e_3l_autoencoder2 train.py --task_name e_3l_autoencoder --task_id 2
# task e_3l_autoencoder3 train.py --task_name e_3l_autoencoder --task_id 10243
# task e_3l_autoencoder4 train.py --task_name e_3l_autoencoder --task_id 10244
# task e_3l_autoencoder5 train.py --task_name e_3l_autoencoder --task_id 10245
# task e_3l_autoencoder6 train.py --task_name e_3l_autoencoder --task_id 10246
# task e_3l_autoencoder7 train.py --task_name e_3l_autoencoder --task_id 10247
# task e_3l_autoencoder8 train.py --task_name e_3l_autoencoder --task_id 10248
# task e_3l_autoencoder5120 train.py --task_name e_3l_autoencoder --task_id 5120 --linear_ae_z_size 512

def e_3l_autoencoder():
    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    encoder = model.WAE_Encoder(args.wae_z_dim)
    decoder = model.WAE_Decoder(args.wae_z_dim)
    net = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)

    checkpoint = torch.load('epoch_400.t7', map_location='cpu')
    net.load_state_dict(checkpoint['experiment'])
    encoder = nn.Sequential(
                            nn.Linear(1024, 2048),
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Linear(2048, args.linear_ae_z_size),

                            )
    decoder = nn.Sequential(
                            nn.Linear(args.linear_ae_z_size, 2048),
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Linear(2048, 1024),

                            )

    predictor = model.ResNet18()

    if args.use_pre:
        directory = '../checkpoint/e_noda/0/best_accuracy.t7'
        checkpoint = torch.load(directory)
        ex = experiment.E_basic(predictor)
        ex.load_state_dict(checkpoint['experiment'])
        predictor = ex.predictor

    if use_cuda:
        net.encoder.cuda()
        encoder.cuda()
        decoder.cuda()
        predictor.cuda()
    net.encoder.eval()
    encoder.eval()
    decoder.eval()
    predictor.eval()

    net = experiment.E_3l_autoencoder(net.encoder, predictor, encoder, decoder)
    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater, False)
    return trainer

# task e_3l_autoencoder_p train.py --task_name e_3l_autoencoder_p --task_id 0
# task e_3l_autoencoder1_p train.py --task_name e_3l_autoencoder_p --task_id 1
# task e_3l_autoencoder2_p train.py --task_name e_3l_autoencoder_p --task_id 2 
# task e_3l_autoencoder_pna train.py --task_name e_3l_autoencoder_p --task_id 0 --no_augment
# task e_3l_autoencoder1_pna train.py --task_name e_3l_autoencoder_p --task_id 1 --no_augment
# task e_3l_autoencoder2_pna train.py --task_name e_3l_autoencoder_p --task_id 2 --no_augment

# task e_3l_autoencoder_pnt train.py --task_name e_3l_autoencoder_p --task_id 0 --no_train
# task e_3l_autoencoder1_pnt train.py --task_name e_3l_autoencoder_p --task_id 1 --no_train
# task e_3l_autoencoder2_pnt train.py --task_name e_3l_autoencoder_p --task_id 2 --no_train 
# task e_3l_autoencoder_pntna train.py --task_name e_3l_autoencoder_p --task_id 0 --no_augment --no_train
# task e_3l_autoencoder1_pntna train.py --task_name e_3l_autoencoder_p --task_id 1 --no_augment --no_train
# task e_3l_autoencoder2_pntna train.py --task_name e_3l_autoencoder_p --task_id 2 --no_augment --no_train

# task e_3l_autoencoder_p512 train.py --task_name e_3l_autoencoder_p --task_id 0512 --linear_ae_z_size 512
# task e_3l_autoencoder1_p512 train.py --task_name e_3l_autoencoder_p --task_id 1512 --linear_ae_z_size 512
# task e_3l_autoencoder2_p512 train.py --task_name e_3l_autoencoder_p --task_id 2512  --linear_ae_z_size 512
# task e_3l_autoencoder_pna512 train.py --task_name e_3l_autoencoder_p --task_id 0512na --no_augment --linear_ae_z_size 512
# task e_3l_autoencoder1_pna512 train.py --task_name e_3l_autoencoder_p --task_id 1512na --no_augment --linear_ae_z_size 512
# task e_3l_autoencoder2_pna512 train.py --task_name e_3l_autoencoder_p --task_id 2512na --no_augment --linear_ae_z_size 512
# task e_3l_autoencoder_pnt512 train.py --task_name e_3l_autoencoder_p --task_id 0512nt --no_train --linear_ae_z_size 512
# task e_3l_autoencoder1_pnt512 train.py --task_name e_3l_autoencoder_p --task_id 1512nt --no_train --linear_ae_z_size 512
# task e_3l_autoencoder2_pnt512 train.py --task_name e_3l_autoencoder_p --task_id 2512nt --no_train  --linear_ae_z_size 512
# task e_3l_autoencoder_pntna512 train.py --task_name e_3l_autoencoder_p --task_id 0512nant --no_augment --no_train --linear_ae_z_size 512
# task e_3l_autoencoder1_pntna512 train.py --task_name e_3l_autoencoder_p --task_id 1512nant --no_augment --no_train --linear_ae_z_size 512
# task e_3l_autoencoder2_pntna512 train.py --task_name e_3l_autoencoder_p --task_id 2512nant --no_augment --no_train --linear_ae_z_size 512

# ------------
# task e_3l_autoencoder_p512m train.py --task_name e_3l_autoencoder_p --task_id 0512m --linear_ae_z_size 512 --is_multi
# task e_3l_autoencoder1_p512m train.py --task_name e_3l_autoencoder_p --task_id 1512m --linear_ae_z_size 512 --is_multi
# task e_3l_autoencoder2_p512m train.py --task_name e_3l_autoencoder_p --task_id 2512m  --linear_ae_z_size 512 --is_multi
# task e_3l_autoencoder_pna512m train.py --task_name e_3l_autoencoder_p --task_id 0512nam --no_augment --linear_ae_z_size 512 --is_multi
# task e_3l_autoencoder1_pna512m train.py --task_name e_3l_autoencoder_p --task_id 1512nam --no_augment --linear_ae_z_size 512 --is_multi
# task e_3l_autoencoder2_pna512m train.py --task_name e_3l_autoencoder_p --task_id 2512nam --no_augment --linear_ae_z_size 512 --is_multi

# task e_3l_autoencoder_p512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 0512m0.1 --linear_ae_z_size 512 --is_multi --w 0.1
# task e_3l_autoencoder1_p512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 1512m0.1 --linear_ae_z_size 512 --is_multi --w 0.1
# task e_3l_autoencoder2_p512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 2512m0.1  --linear_ae_z_size 512 --is_multi --w 0.1
# task e_3l_autoencoder_pna512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.1 --no_augment --linear_ae_z_size 512 --is_multi --w 0.1
# task e_3l_autoencoder1_pna512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.1 --no_augment --linear_ae_z_size 512 --is_multi --w 0.1
# task e_3l_autoencoder2_pna512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.1 --no_augment --linear_ae_z_size 512 --is_multi --w 0.1

# task e_3l_autoencoder_p512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 0512m0.2  --linear_ae_z_size 512 --is_multi --w 0.2 
# task e_3l_autoencoder1_p512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 1512m0.2  --linear_ae_z_size 512 --is_multi --w 0.2 
# task e_3l_autoencoder2_p512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 2512m0.2   --linear_ae_z_size 512 --is_multi --w 0.2 
# task e_3l_autoencoder_pna512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.2  --no_augment --linear_ae_z_size 512 --is_multi --w 0.2 
# task e_3l_autoencoder1_pna512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.2  --no_augment --linear_ae_z_size 512 --is_multi --w 0.2 
# task e_3l_autoencoder2_pna512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.2  --no_augment --linear_ae_z_size 512 --is_multi --w 0.2 

# task e_3l_autoencoder_p512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 0512m0.25 --linear_ae_z_size 512 --is_multi --w 0.25
# task e_3l_autoencoder1_p512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 1512m0.25 --linear_ae_z_size 512 --is_multi --w 0.25
# task e_3l_autoencoder2_p512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 2512m0.25  --linear_ae_z_size 512 --is_multi --w 0.25
# task e_3l_autoencoder_pna512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.25 --no_augment --linear_ae_z_size 512 --is_multi --w  0.25
# task e_3l_autoencoder1_pna512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.25 --no_augment --linear_ae_z_size 512 --is_multi --w 0.25 
# task e_3l_autoencoder2_pna512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.25 --no_augment --linear_ae_z_size 512 --is_multi --w  0.25

# task e_3l_autoencoder_p512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 0512m0.5 --linear_ae_z_size 512 --is_multi --w 0.5
# task e_3l_autoencoder1_p512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 1512m0.5 --linear_ae_z_size 512 --is_multi --w 0.5
# task e_3l_autoencoder2_p512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 2512m0.5  --linear_ae_z_size 512 --is_multi --w 0.5
# task e_3l_autoencoder_pna512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.5 --no_augment --linear_ae_z_size 512 --is_multi --w 0.5
# task e_3l_autoencoder1_pna512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.5 --no_augment --linear_ae_z_size 512 --is_multi --w 0.5
# task e_3l_autoencoder2_pna512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.5 --no_augment --linear_ae_z_size 512 --is_multi --w 0.5

# task e_3l_autoencoder_p512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 0512m2.0 --linear_ae_z_size 512 --is_multi --w 2.0
# task e_3l_autoencoder1_p512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 1512m2.0 --linear_ae_z_size 512 --is_multi --w 2.0
# task e_3l_autoencoder2_p512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 2512m2.0  --linear_ae_z_size 512 --is_multi --w 2.0
# task e_3l_autoencoder_pna512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam2.0 --no_augment --linear_ae_z_size 512 --is_multi --w 2.0
# task e_3l_autoencoder1_pna512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam2.0 --no_augment --linear_ae_z_size 512 --is_multi --w 2.0
# task e_3l_autoencoder2_pna512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam2.0 --no_augment --linear_ae_z_size 512 --is_multi --w 2.0

# task e_3l_autoencoder_p512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 0512m4.0 --linear_ae_z_size 512 --is_multi --w 4.0
# task e_3l_autoencoder1_p512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 1512m4.0 --linear_ae_z_size 512 --is_multi --w 4.0
# task e_3l_autoencoder2_p512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 2512m4.0  --linear_ae_z_size 512 --is_multi --w 4.0
# task e_3l_autoencoder_pna512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam4.0 --no_augment --linear_ae_z_size 512 --is_multi --w 4.0
# task e_3l_autoencoder1_pna512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam4.0 --no_augment --linear_ae_z_size 512 --is_multi --w 4.0
# task e_3l_autoencoder2_pna512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam4.0 --no_augment --linear_ae_z_size 512 --is_multi --w 4.0

# task e_3l_autoencoder_p512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 0512m5.0 --linear_ae_z_size 512 --is_multi --w 5.0
# task e_3l_autoencoder1_p512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 1512m5.0 --linear_ae_z_size 512 --is_multi --w 5.0
# task e_3l_autoencoder2_p512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 2512m5.0  --linear_ae_z_size 512 --is_multi --w 5.0
# task e_3l_autoencoder_pna512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam5.0 --no_augment --linear_ae_z_size 512 --is_multi --w 5.0
# task e_3l_autoencoder1_pna512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam5.0 --no_augment --linear_ae_z_size 512 --is_multi --w 5.0
# task e_3l_autoencoder2_pna512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam5.0 --no_augment --linear_ae_z_size 512 --is_multi --w 5.0

# task e_3l_autoencoder_p512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 0512m10.0 --linear_ae_z_size 512 --is_multi --w 10.0
# task e_3l_autoencoder1_p512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 1512m10.0 --linear_ae_z_size 512 --is_multi --w 10.0
# task e_3l_autoencoder2_p512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 2512m10.0  --linear_ae_z_size 512 --is_multi --w 10.0
# task e_3l_autoencoder_pna512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam10.0 --no_augment --linear_ae_z_size 512 --is_multi --w 10.0
# task e_3l_autoencoder1_pna512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam10.0 --no_augment --linear_ae_z_size 512 --is_multi --w 10.0
# task e_3l_autoencoder2_pna512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam10.0 --no_augment --linear_ae_z_size 512 --is_multi --w 10.0

# -------
# task e_3l_autoencoder_pna512m train.py --task_name e_3l_autoencoder_p --task_id 0512nam --no_augment --linear_ae_z_size 512 --is_multi --no_pre
# task e_3l_autoencoder1_pna512m train.py --task_name e_3l_autoencoder_p --task_id 1512nam --no_augment --linear_ae_z_size 512 --is_multi --no_pre
# task e_3l_autoencoder2_pna512m train.py --task_name e_3l_autoencoder_p --task_id 2512nam --no_augment --linear_ae_z_size 512 --is_multi --no_pre
# task e_3l_autoencoder_pna512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.1 --no_augment --linear_ae_z_size 512 --is_multi --w 0.1 --no_pre
# task e_3l_autoencoder1_pna512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.1 --no_augment --linear_ae_z_size 512 --is_multi --w 0.1 --no_pre
# task e_3l_autoencoder2_pna512m0.1 train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.1 --no_augment --linear_ae_z_size 512 --is_multi --w 0.1 --no_pre
# task e_3l_autoencoder_pna512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.2  --no_augment --linear_ae_z_size 512 --is_multi --w 0.2  --no_pre
# task e_3l_autoencoder1_pna512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.2  --no_augment --linear_ae_z_size 512 --is_multi --w 0.2   --no_pre
# task e_3l_autoencoder2_pna512m0.2  train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.2  --no_augment --linear_ae_z_size 512 --is_multi --w 0.2   --no_pre
# task e_3l_autoencoder_pna512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.25 --no_augment --linear_ae_z_size 512 --is_multi --w  0.25  --no_pre
# task e_3l_autoencoder1_pna512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.25 --no_augment --linear_ae_z_size 512 --is_multi --w 0.25   --no_pre
# task e_3l_autoencoder2_pna512m0.25 train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.25 --no_augment --linear_ae_z_size 512 --is_multi --w  0.25  --no_pre
# task e_3l_autoencoder_pna512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 0512nam0.5 --no_augment --linear_ae_z_size 512 --is_multi --w 0.5  --no_pre
# task e_3l_autoencoder1_pna512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 1512nam0.5 --no_augment --linear_ae_z_size 512 --is_multi --w 0.5  --no_pre
# task e_3l_autoencoder2_pna512m0.5 train.py --task_name e_3l_autoencoder_p --task_id 2512nam0.5 --no_augment --linear_ae_z_size 512 --is_multi --w 0.5  --no_pre
# task e_3l_autoencoder_pna512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam2.0 --no_augment --linear_ae_z_size 512 --is_multi --w 2.0  --no_pre
# task e_3l_autoencoder1_pna512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam2.0 --no_augment --linear_ae_z_size 512 --is_multi --w 2.0  --no_pre
# task e_3l_autoencoder2_pna512m2.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam2.0 --no_augment --linear_ae_z_size 512 --is_multi --w 2.0  --no_pre
# task e_3l_autoencoder_pna512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam4.0 --no_augment --linear_ae_z_size 512 --is_multi --w 4.0  --no_pre
# task e_3l_autoencoder1_pna512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam4.0 --no_augment --linear_ae_z_size 512 --is_multi --w 4.0  --no_pre
# task e_3l_autoencoder2_pna512m4.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam4.0 --no_augment --linear_ae_z_size 512 --is_multi --w 4.0  --no_pre
# task e_3l_autoencoder_pna512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam5.0 --no_augment --linear_ae_z_size 512 --is_multi --w 5.0  --no_pre
# task e_3l_autoencoder1_pna512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam5.0 --no_augment --linear_ae_z_size 512 --is_multi --w 5.0  --no_pre
# task e_3l_autoencoder2_pna512m5.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam5.0 --no_augment --linear_ae_z_size 512 --is_multi --w 5.0  --no_pre
# task e_3l_autoencoder_pna512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 0512nam10.0 --no_augment --linear_ae_z_size 512 --is_multi --w 10.0  --no_pre
# task e_3l_autoencoder1_pna512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 1512nam10.0 --no_augment --linear_ae_z_size 512 --is_multi --w 10.0  --no_pre
# task e_3l_autoencoder2_pna512m10.0 train.py --task_name e_3l_autoencoder_p --task_id 2512nam10.0 --no_augment --linear_ae_z_size 512 --is_multi --w 10.0  --no_pre
def e_3l_autoencoder_p():
    trainloader = dataset.cifar10_train_loader(*loader_args[args.task_name])
    encoder = model.WAE_Encoder(args.wae_z_dim)
    decoder = model.WAE_Decoder(args.wae_z_dim)
    net = experiment.E_WAE(encoder, decoder, use_cuda=use_cuda)

    checkpoint = torch.load('epoch_400.t7', map_location='cpu')
    net.load_state_dict(checkpoint['experiment'])
    encoder = nn.Sequential(
                            nn.Linear(1024, 2048),
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Linear(2048, args.linear_ae_z_size),

                            )
    decoder = nn.Sequential(
                            nn.Linear(args.linear_ae_z_size, 2048),
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Linear(2048, 1024),

                            )

    predictor = model.ResNet18()

    if args.use_pre:
        directory = '../checkpoint/e_noda/0/best_accuracy.t7'
        checkpoint = torch.load(directory)

        ex = experiment.E_basic(predictor)
        ex.load_state_dict(checkpoint['experiment'])
        predictor = ex.predictor

    if use_cuda:
        net.encoder.cuda()
        encoder.cuda()
        decoder.cuda()
        predictor.cuda()
    net.encoder.eval()
    if not args.is_multi:
        encoder.eval()
        decoder.eval()
    predictor.eval()

    net = experiment.E_3l_autoencoder(net.encoder, predictor, encoder, decoder)
    directory = '../checkpoint/e_3l_autoencoder/5120/best_loss.t7'
    checkpoint = torch.load(directory)
    net.load_state_dict(checkpoint['experiment'])
    zlinear = nn.Sequential(
                            nn.Linear(args.linear_ae_z_size, 2048),
                            nn.BatchNorm1d(2048),
                            nn.ReLU(),
                            nn.Linear(2048, 10),

                            )
    net.eval()
    if args.is_multi:
        net = experiment.E_3l_autoencoder_p_multi(net, zlinear, args.w)

    else:
        net = experiment.E_3l_autoencoder_p(net, zlinear, args.is_train)


    if use_cuda:
        net.cuda()
    updater = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay)
    trainer = Trainer(use_cuda, trainloader, max_epoch, net, updater, True)
    return trainer

main = {
        'e_noda': e_basic,
        'e_00': e_basic,
        'e_01': e_basic,
        'e_1': e_zlinear,
        'e_1_zprocess': e_zlinear,
        'e_2': e_zcat,
        'e_2_zprocess': e_zcat,
        'e_2_eval': e_zcat,
        'e_wae': e_wae,
        'e_vae': e_wae,
        'e_allmixup': e_basic,
        'e_allpre': e_z_allpre,
        'e_allpre_vae': e_z_allpre_vae,
        'e_3l_autoencoder': e_3l_autoencoder,
        'e_3l_autoencoder_p': e_3l_autoencoder_p,
        'e_allpre_train': e_z_allpre,
        'e_61': e_zcat
        }

if __name__ == '__main__':

    # get the trainer
    trainer = main[args.task_name]()
    print('trainer completed')

    # testloader
    testloader = dataset.cifar10_test_loader(*loader_args[args.task_name][:2])

    # set headtrain extensions, the work to do before trainning 
    if args.task_name == 'e_wae' or args.task_name == 'e_vae':
        trainer.headtrain(extensions.basic_load, args.resume_epoch, 'loss')
    else:
    	trainer.headtrain(extensions.basic_load, args.resume_epoch, 'accuracy')
    if args.task_name == 'e_allpre' or args.task_name == 'e_allpre_train' or args.task_name == 'e_allpre_vae' or args.task_name == 'e_3l_autoencoder' or args.task_name == 'e_3l_autoencoder_p':
        if args.use_pre:
            lrs = ftlrs
    # set headepoch extensions, the work to do at the begin of each epoch
    trainer.headepoch(extensions.drop_lr, lr_trigger, lrs)

    # set tailepoch extensions, the work to do at the end of each epoch
    if args.task_name == 'e_wae' or args.task_name == 'e_vae' or args.task_name == 'e_3l_autoencoder':
        trainer.tailepoch(extensions.test, testloader, use_cuda, False)
    else:
        trainer.tailepoch(extensions.test, testloader, use_cuda, True)

    trainer.tailepoch(extensions.report_log)
    trainer.tailepoch(extensions.print_log)
    trainer.tailepoch(extensions.save_log)

    # the task of trainning autoencoder: loss, the task of trainning classifier: accuracy
    if args.task_name == 'e_wae' or args.task_name == 'e_vae' or args.task_name == 'e_3l_autoencoder':
        trainer.tailepoch(extensions.gs_best, 'loss')
        trainer.tailepoch(extensions.save_best, 'loss')
    else:
	    trainer.tailepoch(extensions.gs_best, 'accuracy')
	    trainer.tailepoch(extensions.save_best, 'accuracy')

    # trainer.tailepoch(extensions.save_trigger, sv_trigger)
    # set tailtrain extensions, the work to do at the end of train
    if args.task_name == 'e_wae' or args.task_name == 'e_vae' or args.task_name == 'e_3l_autoencoder':
        trainer.tailtrain(extensions.print_best, 'loss')
    else:
    	trainer.tailtrain(extensions.print_best, 'accuracy')

    # run trainer
    trainer.run()

