# --- Import libs ---
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline

import argparse
import sys, os
import csv
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1 
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
LR_image_path = './images/LowResolution.png'
GT_image_path = './images/SR_GT.png'

# record result of each training
time = sys.argv[1]
if time == None:
    exit("enter something to record the name of this training!")

# --- Load Image and Baselines ---
img_LR_pil = crop_image(get_image(LR_image_path, imsize)[0], d=32)
img_GT_pil = crop_image(get_image(GT_image_path, imsize)[0], d=32)

img_LR_np = pil_to_np(img_LR_pil)
img_GT_np = pil_to_np(img_GT_pil)

# imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)
# imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
# if PLOT:
#     plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12)
#     print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
#                                         compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
#                                         compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

plot_image_grid([img_LR_np], factor=13)
plot_image_grid([img_GT_np], factor=13)


# --- Set up parameters and net --
input_depth = 32
 
INPUT       = 'noise'
pad         = 'reflection'
OPT_OVER    = 'net'
KERNEL_TYPE = 'lanczos2'

LR = 0.01
tv_weight = 0.0

OPTIMIZER = 'adam'

if factor == 4: 
    num_iter = 2000
    reg_noise_std = 1.0 / 30.0
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'

net_input = get_noise(input_depth, INPUT, (img_GT_pil.size[1], img_GT_pil.size[0])).type(dtype).detach()
NET_TYPE = 'skip' # UNet, ResNet
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128, 
              skip_n33u=128, 
              skip_n11=4, 
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_var(img_LR_np).type(dtype)
downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

# --- Define closure and optimize ---
def closure():
    global i
    
    if reg_noise_std > 0:
        net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out_HR = net(net_input)
    out_LR = downsampler(out_HR)
    out_LR = pil_to_np(crop_image(np_to_pil(out_LR)))
    # use the mse of downsampled img of out_HR to do backpropagation
    total_loss = mse(out_LR, img_LR_var) 
    
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)
        
    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(img_LR_np, var_to_np(out_LR))
    psnr_HR = compare_psnr(img_GT_np, var_to_np(out_HR))
    print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
                      
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    psnr_writer.writerow([psnr_LR, psnr_HR])

    if PLOT and i % 500 == 0:
        out_HR_np = var_to_np(out_HR)
        plot_image_grid([np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)
        plt.savefig("./out_imgs/req3-" + time + "-" + str(i) + ".png", bbox_inches="tight")
        plt.close()
    i += 1
    
    return total_loss

psnr_writer = csv.writer(open("./output/req3-psnr-out-" + time + ".csv", "w"))
psnr_history = [] 
net_input_saved = net_input.data.clone()
noise = net_input.data.clone()

i = 0
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out_HR_np = np.clip(var_to_np(net(net_input)), 0, 1)
# result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we actually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
plot_image_grid([img_GT_np,
                 out_HR_np], factor=4, nrow=1)

plt.savefig("./out_imgs/req3-final" + time + "-" + str(i) + ".png", bbox_inches="tight")
plt.close()