from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='MNIST | cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
elif opt.dataset == 'MNIST':
    dataset = dset.MNIST(root=opt.dataroot,
                         transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.cuda.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1

# write loss
loss_writer = csv.writer(open("./loss_and_probs.csv", 'w'))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # print(input)
            # print(input.size())
            # error: must be a Variable, so that you can forward
            output = self.main(input)
        return output


netG = Generator(ngpu).cuda()#.to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.Q = nn.Sequential(
            nn.Linear(in_features=8192, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10, bias=True)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            # print(output.view(64, -1, 1, 1).size())
            # error resize outpur of discriminator to 64 x 8192 for Linear layer
            d_output = self.discriminator(output)
            
            q_output = self.Q(output.view(64, -1))

        return d_output.view(-1, 1).squeeze(1), q_output


netD = Discriminator(ngpu).cuda()#.to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

d_criterion = nn.BCELoss().cuda()
q_criterion = nn.CrossEntropyLoss().cuda()

# fixed noise: 54 ~ N(0,1) + 10 one-hot encoder

# this is for each train, we have to sample noise
def _noise_sample(dis_c, noise, bs, device=device):

    noise = torch.randn(64, 54).cuda()
    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs), idx] = 1.0
    c = torch.cuda.FloatTensor(c)
    # dis_c.data.copy_(torch.Tensor(c))
    # print(noise.size())
    # print(dis_c.size())
    # error combine should be same type, here: FloatTensor + FloatTensor
    z = torch.cat([noise, c], 1).view(-1, 64, 1, 1)
    # print(z.size())
    z = torch.autograd.Variable(z)
    return z, idx

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-3, betas=(opt.beta1, 0.999))

real_x = torch.cuda.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).cuda()
label = torch.cuda.FloatTensor(opt.batchSize).cuda()
dis_c = torch.cuda.FloatTensor(opt.batchSize, 10).cuda()
noise = torch.cuda.FloatTensor(opt.batchSize, 54).cuda()

real_x = torch.autograd.Variable(real_x)
label = torch.autograd.Variable(label)
dis_c = torch.autograd.Variable(dis_c)
noise = torch.autograd.Variable(noise)

fixed_z, fixed_idx = _noise_sample(dis_c, noise, 64, device=device)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # real part
        optimizerD.zero_grad()
        x, _ = data

        bs = x.size(0)
        real_x.data.resize_(x.size())
        label.data.resize_(bs)
        dis_c.data.resize_(bs, 10)
        noise.data.resize_(bs, 54)
        
        real_x.data.copy_(x)
        d_out1, q_out1 = netD(real_x) # d_out1 is probs_real
        label.data.fill_(1)
        loss_real = d_criterion(d_out1, label)
        loss_real.backward()

        # fake part
        z, idx = _noise_sample(dis_c, noise, bs)
        fake_x = netG(z)
        d_out2, q_out2 = netD(fake_x.detach()) # d_out2 is probs_fake
        label.data.fill_(0)
        loss_fake = d_criterion(d_out2, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        print(D_loss)
        optimizerD.step()

        # G and Q part
        optimizerG.zero_grad()

        d_out3, q_out3 = netD(fake_x)
        label.data.fill_(1.0)

        reconstruct_loss = d_criterion(d_out3, label)
        
        class_ = torch.LongTensor(idx).cuda()
        target = torch.autograd.Variable(class_)
        Q_loss = q_criterion(q_out3, target)
        
        G_loss = reconstruct_loss + Q_loss
        G_loss.backward()
        optimizerG.step()        

        print('[%d/%d][%d/%d] D_loss: %.4f G_loss: %.4f Q_loss: %.4f prob_real: %.4f prob_fake_before: %.4f prob_fake_after: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 D_loss.item(), G_loss.item(), Q_loss.item(), d_out1, d_out2, d_out3))
        if i % 100 == 0:
            loss_writer.writerow([epoch, opt.niter, i, len(dataloader),
                 D_loss.item(), G_loss.item(), Q_loss.item(), d_out1, d_out2, d_out3])
            vutils.save_image(data,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)

            fake = netG(fixed_z)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

torch.save(netG.state_dict(), '%s/netG_final.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD_final.pth' % (opt.outf))