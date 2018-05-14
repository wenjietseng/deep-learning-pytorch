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
from models import *

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
parser.add_argument('--netQ', default='', help="path to netQ (to continue training)")
parser.add_argument('--netFE', default='', help="path to netFE (to continue training)")

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

if opt.dataset == 'MNIST':
    dataset = dset.MNIST(root=opt.dataroot,
                         transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor()
                         ]))
else:
    print("currently for MNIST only")
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 10

# write loss
loss_writer = csv.writer(open("./loss_and_probs.csv", 'w'))

# declare models
netG = Generator(ngpu).to(device)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netFE = FrontEnd(ngpu).to(device)
# netFE.apply(weights_init)
if opt.netFE != '':
    netFE.load_state_dict(torch.load(opt.netFE))
print(netFE)

netD = D(ngpu).to(device)
# netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netQ = Q(ngpu).to(device)
# netQ.apply(weights_init)
if opt.netQ != '':
    netQ.load_state_dict(torch.load(opt.netQ))
print(netQ)

# fixed noise: 54 ~ N(0,1) + 10 one-hot encoder
# this is for each train, we have to sample noise
def _noise_sample(batchSize, nz, nc, device=device):
    idx = np.random.randint(nc, size=batchSize)
    c = np.zeros((batchSize, nc))
    c[range(batchSize), idx] = 1.0
    c = np.asarray(c)

    noise = torch.randn(batchSize, nz - nc, device=device)
    c_tensor = torch.FloatTensor(c).cuda()
    # error combine should be same type, here: FloatTensor + FloatTensor
    z = torch.cat([noise, c_tensor], dim=1).view(-1, nz, 1, 1)
    # print(z.size())
    # z = torch.autograd.Variable(z)
    return z, idx

real_label = 1
fake_label = 0

# loss function
d_criterion = nn.BCELoss().cuda()
q_criterion = nn.CrossEntropyLoss().cuda()

# setup optimizer
optimizerD = optim.Adam([{'params':netFE.parameters()}, {'params':netD.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam([{'params':netG.parameters()}, {'params':netQ.parameters()}], lr=1e-3, betas=(opt.beta1, 0.999))

# fixed noise
fixed_z, _ = _noise_sample(opt.batchSize, nz, nc, device=device)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        
        # Update D network: log(D(x)) + log(1-D(G(z)))
        # real part
        optimizerD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        fe_out1 = netFE(real_cpu)
        d_out1 = netD(fe_out1)
        errD_real = d_criterion(d_out1, label)
        errD_real.backward()
        probs_real = d_out1.mean().item()

        # fake part
        z, idx = _noise_sample(batch_size, nz, nc, device=device)
        fake_x = netG(z)
        label.fill_(fake_label)
        fe_out2 = netFE(fake_x.detach())
        d_out2 = netD(fe_out2) # d_out2 is probs_fake
        errD_fake = d_criterion(d_out2, label)
        
        errD_fake.backward()
        probs_fake_before_G = d_out2.mean().item()

        D_loss = errD_real + errD_fake
        optimizerD.step()

        # Update G network
        # G and Q part
        optimizerG.zero_grad()
        label.fill_(real_label)
        fe_out3 = netFE(fake_x)
        d_out3= netD(fe_out3)
        probs_fake_after_G = d_out3.mean().item()
        
        reconstruct_loss = d_criterion(d_out3, label)
        
        q_out3 = netQ(fe_out3)
   
        target = torch.LongTensor(idx).cuda()
        Q_loss = q_criterion(q_out3, target)
   
        G_loss = reconstruct_loss + Q_loss
        G_loss.backward()
        optimizerG.step()

        # get value: Tensor.item(), Variable.data[0]
        print('[%d/%d][%d/%d] D_loss: %.4f G_loss: %.4f Q_loss: %.4f prob_real: %.4f prob_fake_before: %.4f prob_fake_after: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 D_loss.item(), G_loss.item(), Q_loss.item(), probs_real,
                 probs_fake_before_G, probs_fake_after_G))
        if i % 100 == 0:
            loss_writer.writerow([epoch, opt.niter, i, len(dataloader),
                 D_loss.item(), G_loss.item(), Q_loss.item(), probs_real,
                 probs_fake_before_G, probs_fake_after_G])
            
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)

            fake = netG(fixed_z)
            vutils.save_image(fake.detach().data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

torch.save(netG.state_dict(), '%s/netG_final.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD_final.pth' % (opt.outf))