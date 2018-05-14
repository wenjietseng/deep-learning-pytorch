from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import argparse
import numpy as np
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='enter path to G model')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

ngpu = 1
nz = 64
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
plt.clf()
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.01, hspace=0.01)
img_no = 0
for k in range(10):
    with torch.no_grad():
        same_noise = torch.randn(1, 54).to(device)
        same_noise = same_noise.expand(10, -1)
        one_hot = []
        for i in range(10):
            c = np.zeros(10, dtype=float)
            c[i] = 1.0
            one_hot.append(c)
        one_hot_tensor = torch.FloatTensor(np.asarray(one_hot)).cuda()
        z = torch.cat([same_noise, one_hot_tensor], 1).view(10, nz, 1, 1)
        # print(z.size())
        fake = netG(z).cpu()
        fake = fake.detach()
        # print(fake.size())
        for j in range(10):
            plt.subplot(gs1[img_no])
            plt.imshow(fake[j].view(64,64).data.numpy(), plt.cm.gray)
            plt.axis('off')
            img_no+=1
plt.savefig('eval_imgs/eval-final.png', dpi=300)
        # vutils.save_image(fake.detach().data, './eval_imgs/eval-out'+ str(k) + '.png',
            # normalize=True)

#python3 eval.py --model ./out_figs/netG_final.pth --cuda