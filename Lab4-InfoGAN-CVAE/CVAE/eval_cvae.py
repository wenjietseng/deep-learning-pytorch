from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import csv
from models import CVAE

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='enter path to saved model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

model = CVAE().to(device)
model.load_state_dict(torch.load(args.model))
model.eval()

plt.clf()
img_no = 1
for k in range(10):
    with torch.no_grad():
        # same_noise = torch.randn(10, 20).to(device)
        same_noise = torch.randn(1, 20).to(device)
        same_noise = same_noise.expand(10, -1)
        one_hot = []
        for i in range(10):
            c = np.zeros(10, dtype=float)
            c[i] = 1.0
            one_hot.append(c)
        one_hot_tensor = torch.FloatTensor(np.asarray(one_hot)).cuda()
        sample = torch.cat((same_noise, one_hot_tensor), dim=1)
        sample = model.decode(sample).cpu()
        for j in range(10):
            plt.subplot(10, 10, img_no)
            plt.imshow(sample[j].view(28, 28).data.numpy())
            plt.axis('off')
            img_no+=1
plt.savefig('eval_results/eval-cvae.png', dpi=300)
        # save_image(sample.view(10, 1, 28, 28),
        #         'eval_results/'+str(k)+'.png')

#python3 eval_cvae.py --model ./out_model/model_final.pth