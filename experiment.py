
import gstate
import extensions
import function as F
import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
import numpy
from torchvision import transforms
# all experiment
# details are in the train.py
# 想法是z_loss等价于recon_loss
# gstate implements global variable cross files, it helps to reduce the number of pass parameters, save train model and resume model.
# use the specification model, experiment, gstate and train modules can revise the code for new requirments at most in 70 lines.

class E_Create_Autocoder_Dataset(nn.Module):
    def __init__(self, encoder, decoder):
        super(E_Create_Autocoder_Dataset, self).__init__()
        encoder.eval()
        decoder.eval()
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x):
        z = self.encoder(x)
        xap = self.decoder(z)
        return xap

class E_WAE(nn.Module):

    def __init__(self, encoder, decoder, w=100, use_cuda=True):
        super(E_WAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # w is the weight of multitask 
        self.w = w
        self.use_cuda = use_cuda
        gstate.clear_statics('number', 'loss')

    def forward(self, x):
        z_tilde = self.encoder(x)
        x_recon = self.decoder(z_tilde)
        z_sample = torch.randn(x.size(0), self.encoder.z_dim)
        if self.use_cuda:
            z_sample = z_sample.cuda()
        z_sample = Variable(z_sample)

        recon_loss = nn.MSELoss()(x_recon, x)
        mmd_loss = F.MMD_Loss(z_tilde, z_sample, z_var=1.0)
        total_loss = recon_loss + self.w * mmd_loss

        gstate.summary(number=x.size(0), loss=recon_loss.item())

        return total_loss

class E_VAE(nn.Module):

    def __init__(self, encoder, decoder, enc_mu, enc_log_sigma, w, scale, use_cuda=True):
        super(E_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_mu = enc_mu
        self.enc_log_sigma = enc_log_sigma
        self.w = w
        self.scale = scale
        self.use_cuda = use_cuda
        gstate.clear_statics('number', 'recon_loss', 'kl_loss', 'loss')

    def forward(self, x, is_z=False):
 

        # Reparameterization trick
        if self.training == True:

            # VAE paper 
            h_encoder = self.encoder(x)
            mu = self.enc_mu(h_encoder)
            log_sigma = self.enc_log_sigma(h_encoder)
            sigma = torch.exp(log_sigma)
            std_z = torch.from_numpy(numpy.random.normal(0, 1, size=sigma.size())).float()

            if self.use_cuda:
                std_z = std_z.cuda()
            z_tilde = mu + sigma * Variable(std_z, requires_grad=False)

            x_recon = self.decoder(z_tilde)

            recon_loss = nn.MSELoss()(x_recon, x)
            kl_loss = F.KL_Loss(z_tilde, sigma)

            total_loss = recon_loss + self.scale * kl_loss

            gstate.summary(number=x.size(0), recon_loss=recon_loss.item(), kl_loss=kl_loss.item(), loss=total_loss.item())
            return total_loss

        else:

            h_encoder = self.encoder(x)
            mu = self.enc_mu(h_encoder)
            z_tilde = mu
            if is_z == True:
                return z_tilde
            x_recon = self.decoder(z_tilde)

            recon_loss = nn.MSELoss()(x_recon, x)

            gstate.summary(number=x.size(0), recon_loss=recon_loss.item(), loss=recon_loss.item())
            return recon_loss


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

# class E_basic_allmixup(nn.Module):
#     def __init__(self, predictor):
#         super(E_basic_allmixup, self).__init__()
#         self.predictor = predictor
#         gstate.clear_statics('number', 'loss', 'accuracy')

#     def forward(self, x, t):
#         y = self.predictor(x)
#         print(type(t))
#         if t.dtype == torch.int64:
#             loss = nn.CrossEntropyLoss()(y, t)
#             accuracy = F.accuracy(y, t)
#             gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)

#         elif t.dtype == torch.float32:
#             loss = (t * (-functional.log_softmax(y, dim=1))).sum()
#             gstate.summary(number=y.size(0), loss=loss.item())
#         return loss


class E_1(nn.Module):
    def __init__(self, encoder, z_processer, zlinear):
        super(E_1, self).__init__()
        gstate.set_value('encoder', encoder)
        self.z_processer = z_processer
        self.zlinear = zlinear
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t):

        z_tilde = gstate.get('encoder')(x)
        if not self.z_processer:
            z_tilde = self.z_processer(z_tilde)
        y = self.zlinear(z_tilde)
        loss = nn.CrossEntropyLoss()(y, t)
        accuracy = F.accuracy(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)

        return loss

class E_2(nn.Module):
    def __init__(self, encoder, predictor, z_processer, zlinear, train_predictor=True):
        super(E_2, self).__init__()
        gstate.set_value('encoder', encoder)
        self.train_predictor = train_predictor
        if train_predictor:
            self.predictor = predictor
        else:
            gstate.set_value('predictor', predictor)
        self.z_processer = z_processer
        self.zlinear = zlinear
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t):

        z_tilde = gstate.get('encoder')(x)
        # transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # x_list = []
        # for xi in x:
        #     x_list.append(transform(x))
        # x_trans = torch.cat(x_list, 0)
        if not self.z_processer == None:
            z_tilde = self.z_processer(z_tilde)
            
        if self.train_predictor:
            predictor = self.predictor
        else:
            predictor = gstate.get('predictor')

        z_resnet = predictor(x, lout=4)

        z = torch.cat((z_tilde, z_resnet), 1)
        y = self.zlinear(z)
        loss = nn.CrossEntropyLoss()(y, t)
        accuracy = F.accuracy(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)

        return loss

class E_61(nn.Module):
    def __init__(self, encoder, decoder, predictor, zlinear, w=1.0):
        super(E_61, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.zlinear = zlinear
        self.mean = torch.from_numpy(numpy.array((0.4914, 0.4822, 0.4465), dtype=numpy.float32)).cuda()
        self.var = torch.from_numpy(numpy.array((0.2023, 0.1994, 0.2010), dtype=numpy.float32)).cuda()
        self.w = w
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t):

        z_tilde = self.encoder(x)
        x_recon = self.decoder(z_tilde)
        x_transform = functional.batch_norm(x, self.mean, self.var)

        z_resnet = self.predictor(x_transform, lout=4)

        z = torch.cat((z_tilde, z_resnet), 1)
        y = self.zlinear(z)
        recon_loss = nn.MSELoss()(x_recon, x)
        class_loss = nn.CrossEntropyLoss()(y, t)
        loss = recon_loss + self.w * class_loss
        accuracy = F.accuracy(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)

        return loss


class E_allpre(nn.Module):
    def __init__(self, encoder, predictor, predictor1, is_train):
        super(E_allpre, self).__init__()
        gstate.set_value('encoder', encoder)
        self.is_train = is_train
        if is_train:
            self.predictor = predictor
        else:
            gstate.set_value('predictor', predictor)
        self.predictor1 = predictor1
        self.mean = torch.from_numpy(numpy.array((0.4914, 0.4822, 0.4465), dtype=numpy.float32)).cuda()
        self.var = torch.from_numpy(numpy.array((0.2023, 0.1994, 0.2010), dtype=numpy.float32)).cuda()
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t): 
        x.requires_grad = False
        z_encode = gstate.get('encoder')(x)
        x_transform = functional.batch_norm(x, self.mean, self.var)
        x_transform = Variable(x_transform.data)
        if self.is_train:
            f = self.predictor(x_transform, lout=4)
        else: 
            f = gstate.get('predictor')(x_transform, lout=4)
        z_cat = torch.cat((z_encode, f), 1)
        y = self.predictor1(z_cat)

        loss = nn.CrossEntropyLoss()(y, t)
        accuracy = F.accuracy(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)
        return loss

class E_allpre_vae(nn.Module):
    def __init__(self, auto_encoder, predictor, predictor1, is_train):
        super(E_allpre_vae, self).__init__()
        gstate.set_value('auto_encoder', auto_encoder)
        if is_train:
            self.predictor = predictor
        else:
            gstate.set_value('predictor', predictor)
        self.is_train = is_train
        self.predictor1 = predictor1
        self.mean = torch.from_numpy(numpy.array((0.4914, 0.4822, 0.4465), dtype=numpy.float32)).cuda()
        self.var = torch.from_numpy(numpy.array((0.2023, 0.1994, 0.2010), dtype=numpy.float32)).cuda()
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t): 
        x.requires_grad = False

        z_encode = gstate.get('auto_encoder')(x, is_z=True)

        x_transform = functional.batch_norm(x, self.mean, self.var)
        x_transform = Variable(x_transform.data)
        if self.is_train:
            f = self.predictor(x_transform, lout=4)
        else:
            f = gstate.get('predictor')(x_transform, lout=4)
        z_cat = torch.cat((z_encode, f), 1)
        y = self.predictor1(z_cat)

        loss = nn.CrossEntropyLoss()(y, t)
        accuracy = F.accuracy(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)
        return loss

class E_3l_autoencoder(nn.Module):
    def __init__(self, netencoder, predictor, encoder, decoder):
        super(E_3l_autoencoder, self).__init__()
        gstate.set_value('netencoder', netencoder)
        gstate.set_value('predictor', predictor)
        self.encoder = encoder
        self.decoder = decoder
        self.mean = torch.from_numpy(numpy.array((0.4914, 0.4822, 0.4465), dtype=numpy.float32)).cuda()
        self.var = torch.from_numpy(numpy.array((0.2023, 0.1994, 0.2010), dtype=numpy.float32)).cuda()
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, lout=False, zcatout=False):
        x.requires_grad = False
        z_encode = gstate.get('netencoder')(x)
        x_transform = functional.batch_norm(x, self.mean, self.var)
        x_transform = Variable(x_transform.data) 
        f = gstate.get('predictor')(x_transform, lout=4)
        z_cat = torch.cat((z_encode, f), 1)
        if zcatout:
            return z_cat

        z = self.encoder(z_cat)
        y = self.decoder(z)
        if lout:
            return z

        loss = nn.MSELoss()(z_cat, y)
        gstate.summary(number=y.size(0), loss=loss.item())
        return loss

class E_3l_autoencoder_p(nn.Module):
    def __init__(self, net, zlinear, is_train=False):
        super(E_3l_autoencoder_p, self).__init__()
        if is_train:
            self.netencoderz = net
        else:
            gstate.set_value('netencoderz', net)
        self.is_train = is_train
        self.zlinear = zlinear
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t):
        x.requires_grad = False
        if self.is_train:
            z = self.netencoderz(x, lout=True)
        else:
            z = gstate.get('netencoderz')(x, lout=True)

        new_x = Variable(z.data)
        y = self.zlinear(new_x)
        accuracy = F.accuracy(y, t)
        loss = nn.CrossEntropyLoss()(y, t)
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)
        return loss

class E_3l_autoencoder_p_multi(nn.Module):
    def __init__(self, net, zlinear, w=1.0):
        super(E_3l_autoencoder_p_multi, self).__init__()

        gstate.set_value('netencoderz', net)
        self.encoder = gstate.get('netencoderz').encoder
        self.decoder = gstate.get('netencoderz').decoder
        self.zlinear = zlinear
        self.w = w
        gstate.clear_statics('number', 'loss', 'accuracy')

    def forward(self, x, t):
        x.requires_grad = False
 
        zcat = gstate.get('netencoderz')(x, zcatout=True)

        z = self.encoder(zcat)
        y = self.decoder(z)
        recon_loss = nn.MSELoss()(zcat, y)
        
        y = self.zlinear(z)
        accuracy = F.accuracy(y, t)
        class_loss = nn.CrossEntropyLoss()(y, t)
        loss = recon_loss + self.w * class_loss
        gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)
        return loss


# class E_01(nn.Module):
#     def __init__(self, predictor, encoder, decoder):
#         super(E_0, self).__init__()
#         self.predictor = predictor
#         encoder.eval()
#         decoder.eval()
#         self.encoder = encoder
#         self.decoder = decoder
#         gstate.init(number=0, loss=0.0, accuracy=0.0)

#     def forward(self, x, t):
#         z = self.encoder(x)
#         augm_x = self.decoder(z)
#         y = self.predictor(augm_x)

#         loss = nn.CrossEntropyLoss()(y, t)
#         accuracy = F.accuracy(y, t)
#         gstate.summary(number=y.size(0), loss=loss.item(), accuracy=accuracy)
#         return loss

#     def eval():
#         self.predictor.eval()

#     def train():
#         self.predictor.train()

if __name__ == '__main__':
    net = E_noDA()
    net = E_00()
    net = E_Create_Autocoder_Dataset()