import torch
import torch.nn as nn

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
            nn.Conv2d(3, 1, 3, 1, 1),
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
        mu, logvar = self.encode(x)#.view(-1, 11, 28, 28))
        # print(mu.mean().item())
        # print(logvar.mean().item())
        z = self.reparameterize(mu, logvar)
        c = torch.Tensor(c).cuda()
        # print(c.size())
        # print(z.size())
        new_z = torch.cat((z, c), dim=-1)
        return self.decode(new_z), mu, logvar
