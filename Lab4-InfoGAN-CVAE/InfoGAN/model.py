import torch.nn as nn
nz = 64
ngf = 64
ndf = 64

def weights_init(m):
    """ custom weights initialization called on netG and netD """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output



class FrontEnd(nn.Module):
    """ Front end part of discriminator and netQ, they shared the same architecture """
    def __init__(self, ngpu):
        super(FrontEnd, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf*2, eps=1e-05, momentum=0.1, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf*4, eps=1e-05, momentum=0.1, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf*8, eps=1e-05, momentum=0.1, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input).view(-1, 1)

        return output

class Q(nn.Module):
    def __init__(self, ngpu):
        super(Q, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=8192, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10, bias=True)
        )
    
    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        disc_logits = self.main(input)
        return disc_logits