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

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/Data', train=True, download=True,
                   transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/Data', train=False,                          
                   transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(11, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(3, 1, 3, 1, 1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(784, 400, bias=True),
            nn.ReLU()
        )
        self.fc21 = nn.Linear(400, 20, bias=True)
        self.fc22 = nn.Linear(400, 20, bias=True)
        self.fc3 = nn.Sequential(
            nn.Linear(30, 392, bias=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 11, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(11, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def encode(self, x):
        out = self.conv1(x)
        out = self.fc1(out.view(-1, 784))
        return self.fc21(out), self.fc22(out)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        
        h3 = self.fc3(z)
        out = self.conv2(h3.view(-1, 2, 14, 14))
        return out

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 11, 28, 28))
        z = self.reparameterize(mu, logvar)
        c = torch.Tensor(c).cuda()
        # print(c.size())
        # print(z.size())
        new_z = torch.cat((z, c), dim=1)
        return self.decode(new_z), mu, logvar


model = CVAE().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    BCE = F.mse_loss(recon_x, x.view(-1, 784), size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.to(device)
        
        # add one-hot to each pixel of img
        one_hot_lst = []
        for idx in y:
            one_hot = np.zeros((10), dtype=float)
            one_hot[idx.item()] = 1.0
            one_hot_lst.append(one_hot)
        one_hot_lst = np.asarray(one_hot_lst)
        one_hot_tensor = torch.Tensor(one_hot_lst).view(-1, 10, 1, 1).cuda() # batch_size x 10 x 1 x 1
        one_hot_tensor = one_hot_tensor.expand(-1, -1, 28, 28) # batch_size x 10 x 28 x 28
        new_data = torch.cat((data, one_hot_tensor), dim=1)
        # print(new_data.size()) 128 x 11 x 28 x 28

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(new_data, one_hot_lst)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')