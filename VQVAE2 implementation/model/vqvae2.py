import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, K, D, beta=0.25):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.emb = nn.Embedding(K, D)
        self.emb.weight.data.uniform_(-1.0 / K, 1.0 / K)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.D)

        d2 = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.emb.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.emb.weight.t())
        )

        encoding_indices = torch.argmin(d2, dim=1).unsqueeze(1)
        z_q = self.emb(encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.net(x)

class VQVAE2(nn.Module):
    def __init__(self, in_ch=4, hidden_ch=128, num_embeddings=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_ch // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_ch // 2, hidden_ch, 4, 2, 1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.BatchNorm2d(hidden_ch),
            ResidualBlock(hidden_ch),
            ResidualBlock(hidden_ch)
        )

        self.vq = VectorQuantizer(num_embeddings, hidden_ch)

        self.dec = nn.Sequential(
            ResidualBlock(hidden_ch),
            ResidualBlock(hidden_ch),
            nn.ConvTranspose2d(hidden_ch, hidden_ch // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_ch // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_ch // 2, in_ch, 4, 2, 1)
        )

    def forward(self, x):
        z = self.enc(x)
        z_q, loss = self.vq(z)
        out = self.dec(z_q)
        return out, loss
