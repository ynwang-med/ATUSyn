from . import layers
from . import up_or_down_sampling, dense_layer
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
dense = dense_layer.dense

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups,in_channel, style_dim):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channel, affine=False, eps=1e-6)
        self.style = dense(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        # Query: temb (time embedding)
        # Key/Value: text_emb (text embedding)
        
        # Project the inputs to the appropriate dimensions
        q = self.q_proj(query)  # [batch_size, seq_len, dim]#1,1,256
        k = self.k_proj(key)    # [batch_size, seq_len, dim]
        v = self.v_proj(value)  # [batch_size, seq_len, dim]

        # Compute attention scores (scaled dot-product attention)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # Apply attention weights to the value vectors
        attention_output = torch.matmul(attention_weights, v)  # [batch_size, seq_len, dim]

        # Project back to the original dimension
        output = self.out_proj(attention_output)  # [batch_size, seq_len, dim]

        return output

class ResnetBlockBigGANpp_Adagn(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, zemb_dim=None, up=False, down=False,
                 dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
                 skip_rescale=True, init_scale=0., text_emb_dim=None):
        super().__init__()
        # print(zemb_dim)
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = AdaptiveGroupNorm(min(in_ch // 4, 32), in_ch, zemb_dim)
        self.zemb_dim=zemb_dim
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = AdaptiveGroupNorm(min(out_ch // 4, 32), out_ch, zemb_dim)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Add the cross-attention layer for merging text and time embeddings
        self.cross_attention = CrossAttention(dim=zemb_dim, num_heads=8)
        
        self.text_emb_dim = text_emb_dim  # The dimension of the text embedding

    def forward(self, x, temb=None, zemb=None, text_emb=None):
        # print(self.out_ch, self.in_ch)
        h = self.act(self.GroupNorm_0(x, zemb)) #1,64,256,256 #自适应分组归一化并激活

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)
        h = self.Conv_0(h) #1,64,256,256 之后会调整为128
        # If text_emb is provided, apply cross-attention between temb and text_emb
        if text_emb is not None and temb is not None:
            # Cross-attention between time embedding (temb) and text embedding (text_emb)
            text = self.cross_attention(h, text_emb, text_emb)  # Cross-attention between time and text embedding #1,1,256
            t = self.Dense_0(self.act(temb))[:, :, None, None] #1,64,1,1
            h = h+text+t

        # Add bias to each feature map conditioned on the time embedding
        # if temb is not None:
        #     h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h, zemb))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
       
        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)
