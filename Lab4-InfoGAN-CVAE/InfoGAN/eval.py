from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='enter path to G model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

ngpu = 1
nz = 54
ngf = 64
ndf = 64
nc = 10
batch_size = 64

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
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
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
# load network
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(opt.model))
netG.cuda()
netG.eval()

# generate a z as lab4 desciption
idx = np.random.randint(nc, size=batch_size)
c = np.zeros((batch_size, nc))
c[range(batch_size), idx] = 1.0

noise = torch.randn(batch_size, nz - nc, device=device)
c_tensor = torch.cuda.FloatTensor(batch_size, nc).cuda()
# error combine should be same type, here: FloatTensor + FloatTensor
c_tensor.data.copy_(torch.Tensor(c))
z = torch.cat([noise, c_tensor], 1).view(-1, nz, 1, 1)
print(z.size())

fake = netG(z)
vutils.save_image(fake.detach().data, 'eval-out.png',
    normalize=True)
