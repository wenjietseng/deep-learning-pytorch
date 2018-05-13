python3 main.py --dataset MNIST --dataroot ~/Data \
--outf ./out_figs --cuda --niter 10

# python3 eval.py --model ./out_figs/netG_final.pth --cuda