import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import cfg
from torch.nn import TransformerEncoder,TransformerEncoderLayer

args = cfg.parse_args()


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.channels = args.channels
        self.latent_dim = args.latent_dim
        self.seq_len = args.seq_len
        self.embed_dim = args.embed_dim
        self.patch_size = args.patch_size
        self.depth = args.g_depth
        self.dropout = args.dropout
        self.num_heads = args.heads
        
        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))  ###############################

        encoder_layers = TransformerEncoderLayer(self.embed_dim, self.num_heads, self.latent_dim, self.dropout,
                                                 batch_first=True,norm_first=True)
        self.blocks = TransformerEncoder(encoder_layers, self.depth)

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)
        return output

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels, patch_size, emb_size, seq_length):
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # print('####1',x.shape) # 32, 1, 1, 50 b c h w
        x = self.projection(x)
        # print('####2', x.shape) # [32, 10, 32]) # b patch embedding_size
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        # print('####3',x.shape) # torch.Size([32, 11, 50]) b patch+cls embedding_size
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 args):
        super(Discriminator,self).__init__()
        self.in_channels = args.channels
        self.patch_size = args.patch_size
        self.embed_dim = args.embed_dim
        self.seq_length = args.seq_len
        self.depth = args.d_depth
        self.n_classes = args.n_classes
        self.num_heads = args.heads
        self.latent_dim = args.latent_dim
        self.dropout = args.dropout

        self.patch_embedding = PatchEmbedding_Linear(self.in_channels, self.patch_size, self.embed_dim, self.seq_length)

        encoder_layers = TransformerEncoderLayer(self.embed_dim, self.num_heads, self.latent_dim, self.dropout,
                                                 batch_first=True, norm_first=True)

        self.dis_blocks = TransformerEncoder(encoder_layers, self.depth)
        self.classification_layer = ClassificationHead(self.embed_dim, self.n_classes)
    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.dis_blocks(x)
        out = self.classification_layer(x)
        return out

# if __name__ == '__main__':
#     X = torch.rand(2, 1, 1, 50)
#     X2 = torch.rand(2, args.latent_dim)
#     dis_net = Discriminator()
#     gen_net = Generator()
#     gen_out = gen_net(X2)
#     out = dis_net(X)
#     print(out.shape)
    # print(gen_out.shape)

