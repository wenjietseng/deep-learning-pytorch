# python3 main.py --dataset MNIST --dataroot ~/Data \
# --outf ./out_figs --cuda

python3 eval.py --dataset MNIST --dataroot . --model ./out_figs/netG_final.pth --cuda