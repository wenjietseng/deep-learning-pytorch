from __future__ import print_function
import argparse
import torch
from main import Generator
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

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
