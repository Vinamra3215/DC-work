import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    def forward(self, x):
        return x + self._block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens // 2, 4, 2, 1)
        self._bn_1 = nn.BatchNorm2d(num_hiddens // 2)
        self._conv_2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 4, 2, 1)
        self._bn_2 = nn.BatchNorm2d(num_hiddens)
        self._conv_3 = nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1)
        self._bn_3 = nn.BatchNorm2d(num_hiddens)
        
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens) 
              for _ in range(num_residual_layers)]
        )

    def forward(self, inputs):
        x = F.relu(self._bn_1(self._conv_1(inputs)), inplace=False)
        x = F.relu(self._bn_2(self._conv_2(x)), inplace=False)
        x = self._residual_stack(F.relu(self._bn_3(self._conv_3(x)), inplace=False))
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
        
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens) 
              for _ in range(num_residual_layers)]
        )
        
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, 2, 1)
        self._bn_1 = nn.BatchNorm2d(num_hiddens // 2)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens // 2, out_channels, 4, 2, 1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = F.relu(self._bn_1(self._conv_trans_1(x)), inplace=False)
        return self._conv_trans_2(x)

class VQVAE(nn.Module):
    def __init__(self, in_ch=4, hidden_ch=128, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()
        
        num_residual_layers = 2
        num_residual_hiddens = 32
        
        self._encoder = Encoder(in_ch, hidden_ch, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(hidden_ch, embedding_dim, 1, 1)
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost=0.25)
        
        self._decoder = Decoder(embedding_dim, hidden_ch, num_residual_layers, num_residual_hiddens, in_ch)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        
        quantized, vq_loss = self._vq_vae(z)
        
        x_recon = self._decoder(quantized)

        return x_recon, vq_loss
