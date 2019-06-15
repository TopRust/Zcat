
import gstate
import time
import os
import json
import torch
from torch.autograd import Variable

# test testset
def test(trainer, loader_test, use_cuda, supervised=True):

    gstate.set_value('train_statics', gstate.get('statics').copy())
    gstate.clear_statics()
    trainer.experiment.eval()
    for x, t in loader_test:
        if not supervised:
            if use_cuda:
                x = x.cuda()
            x = x.float()
            x = Variable(x)
            loss = trainer.experiment(x)
        else:
            if use_cuda:
                x, t = x.cuda(), t.cuda()
            x, t = Variable(x), Variable(t)
            loss = trainer.experiment(x, t)
    trainer.experiment.train()
    gstate.set_value('test_statics', gstate.get('statics').copy())

# save log
def report_log(trainer):

    train_statics = gstate.get('train_statics')
    test_statics = gstate.get('test_statics')
    log_dict = {}
    log_dict['epoch'] = gstate.get('epoch')
    log_dict['elapsed_time'] = time.time() - gstate.get('start_time')
    for key in train_statics:
        if key != 'number':
            log_dict['train_{}'.format(key)] = train_statics[key] / train_statics['number']
            log_dict['test_{}'.format(key)] = test_statics[key] / test_statics['number']

    basic_log(trainer, **log_dict)

def basic_log(trainer, **kwargs):

    gstate.get('log').append(kwargs)

# end epoch
def print_log(trainer):
    
    statics = gstate.get('log')[-1]
    print('{}: {}  {}: {:.2f}'.format('epoch', statics['epoch'], 'elapsed_time', statics['elapsed_time']), end='')

    for key in statics:
        if not (key == 'epoch' or key == 'elapsed_time'):
            print('  {}: {:.6f}'.format(key, statics[key]), end='')

    print()

# save log
def save_log(trainer):
    directory = '../checkpoint/{}/{}'.format(gstate.get('task_name'), gstate.get('task_id'))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    log = open('{}/log'.format(directory), 'w')
    string = json.dumps(gstate.get('log'), sort_keys=False, indent=4, separators=(',', ': '))
    log.write(string)
    log.close()

# drop learning rate
def drop_lr(trainer, lr_trigger, lrs):
    for i in range(len(lr_trigger)):
        if gstate.get('epoch') <= lr_trigger[i]:
            for param_group in trainer.updater.param_groups:
                param_group['lr'] = lrs[i]
            break

# record the best data of testset
def gs_best(trainer, key):

    value = gstate.get('log')[-1]['test_{}'.format(key)]
    if key == 'loss':
        if value < gstate.get('best_loss'):
            gstate.set_value('best_epoch', gstate.get('epoch'))
            gstate.set_value('best_{}'.format(key), value)
    elif key == 'accuracy':
        if value > gstate.get('best_accuracy'):
            gstate.set_value('best_epoch', gstate.get('epoch'))
            gstate.set_value('best_{}'.format(key), value)

# save the best model
def save_best(trainer, key):
    if gstate.get('best_epoch') == gstate.get('epoch'):
        save_experiment(trainer, 'best_{}'.format(key))

# save model in trigger list
def save_trigger(trainer, sv_trigger):

    for epoch in sv_trigger:
        if epoch == gstate.get('epoch'):
            name = 'epoch_{}'.format(epoch)
            save_experiment(trainer, name)
            break
# save experiment
def save_experiment(trainer, name):

    basic_save(trainer, name,
        epoch=gstate.get('epoch'),
        experiment=trainer.experiment.state_dict(),
        updater=trainer.updater.state_dict()
        )
# basic end epoch 
def basic_save(trainer, name, **kwargs):

    state = kwargs
    directory = '../checkpoint/{}/{}'.format(gstate.get('task_name'), gstate.get('task_id'))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    torch.save(state, '{}/{}.t7'.format(directory, name))

# print the best record
def print_best(trainer, key):
    print('best epoch: {}, best {}: {:.6f}'.format(gstate.get('best_epoch'),  key, gstate.get('best_{}'.format(key))))

# resume the resume_epoch model. the default value of resume_epoch is -1, resume the best model
def basic_load(trainer, resume_epoch, key):

    if resume_epoch == 0:
        pass

    else:
        directory = '../checkpoint/{}/{}'.format(gstate.get('task_name'), gstate.get('task_id'))

        if resume_epoch < 0:
            loadpath = '{}/{}.t7'.format(directory, 'best_{}'.format(key))
        else:
            loadpath = '{}/epoch_{}.t7'.format(directory, resume_epoch)
        if os.path.exits(loadpath):
            checkpoint = torch.load(loadpath)

            log = open('{}/log'.format(directory), 'r')
            string = log.read()
            log.close()

            gstate.set_value('log', eval(string)[: checkpoint['epoch']])
            gstate.set_value('epoch', checkpoint['epoch'] + 1)
            trainer.experiment.load_state_dict(checkpoint['experiment'])     
            trainer.updater.load_state_dict(checkpoint['updater'])
            gstate.set_value('start_time', gstate.get('start_time') - gstate.get('log')[-1]['elapsed_time'])

