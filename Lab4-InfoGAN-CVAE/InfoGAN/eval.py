from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from main import Generator
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
