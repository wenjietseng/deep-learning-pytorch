# --- Import libs ---
from __future__ import print_function
import matplotlib
# or use ssh -X
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline

import sys, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv

import numpy as np
from models import *

import torch
import torch.optim

from torch.autograd import Variable
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.
fname = './images/brown-falcon.jpg'

# --- Load image ---
img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)

# --- Load which setup ---
para_setup = sys.argv[1]
img_noisy_np = pil_to_np(img_noisy_pil)
if para_setup == "1":
    # 1. image only
    img_noisy_np = pil_to_np(img_noisy_pil)
elif para_setup == "2":
    # 2. image + noise
    img_noisy_np = img_noisy_np + np.random.uniform(size=img_noisy_np.shape)	
    #img_noisy_np = np.clip(img_noisy_np, 0, 1)
elif para_setup == "3":
    # 3. image shuffled: random shuffle columns and rows
    np.random.shuffle(img_noisy_np)
elif para_setup == "4":
    # 4. noise only
    img_noisy_np = np.random.uniform(size=img_noisy_np.shape)
else:
	exit("parameter setup number fault")


# As we don't have ground truth
img_pil = img_noisy_pil
img_np = img_noisy_np
    
#if PLOT:
#    plot_image_grid([img_np], 4, 5)
        

# --- Setup ---
INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 500
num_iter=2400
input_depth = 3
figsize = 5 

net = skip(
            input_depth, 3, 
            num_channels_down = [8, 16, 32, 64, 128], 
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0, 0, 4, 4], 
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

net = net.type(dtype)

#print(net)

# net_input is z in Lab2 description
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
#print(net_input)
# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

# !!!!!!
img_noisy_var = np_to_var(img_noisy_np).type(dtype)
# --- Optimize ---
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

training_loss_writer = csv.writer(open("./output/outi"+ para_setup +".csv", 'w'))

i = 0
def closure():
    
    global i
    global training_loss_writer 
    
	# add normal noise each iteration
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
   
    total_loss = mse(out, img_noisy_var)
    total_loss.backward()
    


    print ('Iteration %05d    Loss %f' % (i, total_loss.data[0]), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = var_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        plt.savefig("./out_imgs/setup" + para_setup + "-" + str(i) + ".png", bbox_inches="tight")
        plt.close()

    training_loss_writer.writerow([i, total_loss.data[0]])
    i += 1

    return total_loss



# main
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_np = var_to_np(net(net_input))
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13) 
plt.savefig("./out_imgs/setup" + para_setup + "-final-compare.png", bbox_inches="tight")
plt.close()
