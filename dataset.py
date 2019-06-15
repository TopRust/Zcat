
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
from PIL import Image

from function import unpickle
import numpy

# cifar10 trainset
def cifar10_train(is_download=False, is_augment=False, is_normal=False):
    transforms_train = get_transform_train(is_augment, is_normal)
    trainset = datasets.CIFAR10('../../data', train=True, download=is_download, transform=transforms_train)
    return trainset

# cidar10 testset
def cifar10_test(is_download=False, is_normal=False):
    transforms_test = get_transform_test(is_normal)
    testset = datasets.CIFAR10('../../data', train=False, download=is_download, transform=transforms_test)
    return testset

# train transform
def get_transform_train(is_augment=False, is_normal=False):

    # train transform
    if is_augment and is_normal:
        transforms_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010))
                                            ])
        print('Data augment, normalize')

    elif is_augment:
        transforms_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                            ])
        print('Data augment')

    elif is_normal:
        transforms_train = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010))
                                            ])
        print('Data normalize')


    else:
        transforms_train = transforms.Compose([transforms.ToTensor()])
        print('Data')

    return transforms_train

# test transform
def get_transform_test(is_normal=False):
    # test transform
    if is_normal:
        transforms_test = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010))
                                            ])
    
    else:
        transforms_test = transforms.Compose([transforms.ToTensor()]) 

    return transforms_test


from PIL import Image

# e2 dataset
class AutoMix_Dataset(Dataset):

    def __init__(self, is_augment=False, is_normal=True):
        self.transform = get_transform_train(is_augment, is_normal=is_normal)
        self.orignal_trainset = datasets.CIFAR10('../../data', train=True, download=False, transform=self.transform)
        self.data, self.labels = get_auto_dataset()
 

    def __getitem__(self, index):
        if index < 50000:
            image, label = self.orignal_trainset[index]
        else:
            index = index - 50000
            image, label = self.data[index], self.labels[index]
            image = image.reshape(3, 32, 32)
            image = numpy.transpose(image, (1, 2, 0))
            image = Image.fromarray(image)
            image = self.transform(image)
            label = int(label)
        return image, label

    def __len__(self):
        return 2 * len(self.orignal_trainset)

# e1 dataset
class Auto_Dataset(Dataset):

    def __init__(self, is_augment=False, is_normal=True):
        self.transform = get_transform_train(is_augment, is_normal=is_normal)
        self.data, self.labels = get_auto_dataset()
 
    def __getitem__(self, index):

        image, label = self.data[index], self.labels[index]
        image = image.reshape(3, 32, 32)
        image = numpy.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        image = self.transform(image)
        label = int(label)
        return image, label

    def __len__(self):
        return len(self.labels)

# read 5--11 batches
from function import unpickle
def get_auto_dataset():

    for i in range(6, 11):
        batch = unpickle('../../data/cifar-10-batches-py/data_batch_{}'.format(i))
        data = batch[b'data'] if i == 6 else numpy.concatenate([data, batch[b'data']])
        label_np = numpy.array(batch[b'labels'])
        labels = label_np if i == 6 else numpy.concatenate([labels, label_np])
    
    return data, labels

# cifar10 trainloader
def cifar10_train_loader(is_download=False, is_normal=True, is_augment=False, is_autoencoder=False, only_autoencoder=True):
    
    size_train = 100
    if is_autoencoder:
        if only_autoencoder:
            trainset = Auto_Dataset(is_augment, is_normal)
        else:
            trainset = AutoMix_Dataset(is_augment, is_normal)
            size_train = 200
    else:
        trainset= cifar10_train(is_download, is_augment, is_normal)

    trainloader = data.DataLoader(trainset, batch_size=size_train, shuffle=True, num_workers=8)
    
    return trainloader

# cifar10 testloader
def cifar10_test_loader(is_download=False, is_normal=False):
    
    testset = cifar10_test(is_download, is_normal)

    testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    
    return testloader

# class All_Mixup_Dataset():


#     def __init__(self, is_normal=True, use_cuda=False, n_merged=10):
#         self.use_cuda = use_cuda
#         self.orignal_trainset = cifar10_train(is_normal=is_normal)
#         self.n_merged = n_merged
#         self.images = torch.zeros(len(self.orignal_trainset), 3, 32, 32)
#         self.labels = torch.zeros(len(self.orignal_trainset)).int()
#         self.vector_labels = torch.zeros(n_merged, 10)
#         self.eye = torch.ones(10)
#         if use_cuda:
#             self.images = self.images.cuda()
#             self.labels = self.labels.cuda()
#             self.vector_labels.cuda()
#             self.eye.cuda()

#         for i, (image, label) in enumerate(self.orignal_trainset):
#             if self.use_cuda:
#                 self.images[i] = image.cuda()
#                 self.labels[i] = label.cuda()
#             else:
#                 self.images[i] = image
#                 self.labels[i] = label

#     def __getitem__(self, index):

#         if self.use_cuda:
#             rand_ws = torch.rand(self.n_merged).cuda()
#         else:
#             rand_ws = torch.rand(self.n_merged)

#         weight = rand_ws / rand_ws.sum()
#         rand_weight_images = (weight * self.images[index * self.n_merged: (index + 1) * self.n_merged].permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
#         image = rand_weight_images.sum(0)

#         for i in range(self.vector_labels.size(0)):
#             self.vector_labels[i] = self.eye[self.labels[i + index * self.n_merged]]
#         label = (weight * self.vector_labels.permute(1, 0)).permute(0, 1).sum(0)

#         return image, label

#     def __len__(self):
#         return (len(self.orignal_trainset) - 1) // self.n_merged + 1


# eye[1] * -F.log_softmax(a,dim=0)

if __name__ == '__main__':

    trainloader = cifar10_train_loader(is_download=True, is_augment=False, is_normal=False, is_autoencoder=False)
    trainloader = cifar10_train_loader(is_download=False, is_augment=False, is_normal=False, is_autoencoder=False)
    trainloader = cifar10_train_loader(is_download=False, is_augment=True, is_normal=False, is_autoencoder=False)
    trainloader = cifar10_train_loader(is_download=False, is_augment=False, is_normal=True, is_autoencoder=False)
    trainloader = cifar10_train_loader(is_download=False, is_augment=False, is_normal=False, is_autoencoder=True, only_autoencoder=True)
    trainloader = cifar10_train_loader(is_download=False, is_augment=False, is_normal=False, is_autoencoder=True, only_autoencoder=False)
    testloader = cifar10_test_loader()
    testloader = cifar10_test_loader(is_normal=True)
    # trainset = All_Mixup_Dataset()
    # print(trainset[0])
