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
from matplotlib import gridspec as gridspec

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
# plt.subplots_adjust(wspace=0, hspace=0)
gs1 = gridspec.GridSpec(10, 10)
gs1.update(wspace=0.01, hspace=0.01)
img_no = 0
for k in range(3): # turn it into 3
    with torch.no_grad():
        # same_noise = torch.randn(10, 20).to(device)
        same_noise = torch.randn(1, 20).to(device)
        # same_noise = same_noise.expand(10, -1) # may need to change :)
        one_hot = []
        # turn it into given number
        c = np.zeros(10, dtype = float)
        c[0] = 1.0
        one_hot = c
        # for i in range(10): 
            # c = np.zeros(10, dtype=float)
            # c[i] = 1.0
            # one_hot.append(c)
        one_hot_tensor = torch.FloatTensor(np.asarray(one_hot)).cuda()
        sample = torch.cat((same_noise, one_hot_tensor), dim=1)
        sample = model.decode(sample).cpu()
        save_image(sample.view(10, 1, 28, 28),
                   'eval_results/demo' + str(k) + '.png')
#         for j in range(10):
#             plt.subplot(gs1[img_no])
#             plt.imshow(sample[j].view(28, 28).data.numpy(), plt.cm.gray)
#             plt.axis('off')
#             img_no+=1
# plt.savefig('eval_results/demo.png', dpi=300)

        # use this line to create images
        # save_image(sample.view(10, 1, 28, 28),
        #         'eval_results/'+str(k)+'.png')

#python3 eval_cvae.py --model ./out_model/model_final.pth
