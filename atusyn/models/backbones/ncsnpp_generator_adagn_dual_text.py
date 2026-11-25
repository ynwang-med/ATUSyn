# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer, layersspp_text
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
# ResnetBlockBigGAN = layersspp_text.ResnetBlockBigGANpp_Adagn
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

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


class BottleneckTextCrossAttention(nn.Module):
    """Spatial text cross-attention at the U-Net bottleneck.

    The original text-time attention path is kept unchanged. This module adds
    a second, spatial path where bottleneck feature tokens query CLIP text
    tokens. A small learnable residual gate makes the pilot experiment stable.
    """

    def __init__(self, in_ch, txt_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32),
            num_channels=in_ch,
            eps=1e-6,
        )
        self.proj_in = nn.Conv2d(in_ch, txt_dim, kernel_size=1)
        self.text_norm = nn.LayerNorm(txt_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=txt_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj_out = nn.Conv2d(txt_dim, in_ch, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, text_emb):
        if text_emb is None:
            return h

        x = self.proj_in(self.norm(h))
        b, c, height, width = x.shape
        q = x.permute(0, 2, 3, 1).reshape(b, height * width, c)
        kv = self.text_norm(text_emb)

        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out.reshape(b, height, width, c).permute(0, 3, 1, 2)
        out = self.proj_out(out)
        return h + self.gate * out

@utils.register_model(name='ncsnpp_dual_text')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.not_use_tanh = config.not_use_tanh
    self.act = act = nn.SiLU()
    self.z_emb_dim = z_emb_dim = config.z_emb_dim
    
    self.nf = nf = config.num_channels_dae
    ch_mult = config.ch_mult
    self.num_res_blocks = num_res_blocks = config.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.attn_resolutions
    dropout = config.dropout
    resamp_with_conv = config.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.conditional  # noise-conditional
    fir = config.fir
    fir_kernel = config.fir_kernel
    self.skip_rescale = skip_rescale = config.skip_rescale
    self.resblock_type = resblock_type = config.resblock_type.lower()
    self.progressive = progressive = config.progressive.lower()
    self.progressive_input = progressive_input = config.progressive_input.lower()
    self.embedding_type = embedding_type = config.embedding_type.lower()
    init_scale = 0.
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      #assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf #目前用的是这个嵌入

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional: #被调用
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)
    elif resblock_type == 'biggan_oneadagn':
      ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4,
                                      zemb_dim = z_emb_dim)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    in_channels  = config.num_channels#*2
    out_channels = config.out_channels
    if progressive_input != 'none':
      input_pyramid_ch = in_channels 

    modules.append(conv3x3(in_channels , nf))
    hs_c = [nf]
    #下采样模块
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(BottleneckTextCrossAttention(in_ch=in_ch, txt_dim=256, num_heads=8))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, out_channels, init_scale=init_scale))
            pyramid_ch = out_channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, out_channels, bias=True, init_scale=init_scale))
            pyramid_ch = out_channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      modules.append(conv3x3(in_ch, out_channels, init_scale=init_scale)) #决定输出通道

    self.all_modules = nn.ModuleList(modules)
    
    
    mapping_layers = [PixelNorm(),
                      dense(config.nz, z_emb_dim),
                      self.act,]
    for _ in range(config.n_mlp):
        mapping_layers.append(dense(z_emb_dim, z_emb_dim))
        mapping_layers.append(self.act)
    self.z_transform = nn.Sequential(*mapping_layers)
    # self.text_embedding = nn.Linear(text_embedding_dim=768, num_channels_dae=256)  # text_embedding_dim是文本编码的维度
    self.text_embedding = nn.Linear(1024, 256)
    self.cross_attention = CrossAttention(dim=256, num_heads=8)
    # self.text_proj = nn.Sequential( #尝试使用这个方法
    # nn.Linear(1024, 512),
    # nn.ReLU(),
    # nn.Linear(512, 256)
    # )


  def forward(self, x, time_cond, z, text_emb=None):
    # timestep/noise_level embedding; only for continuous training
    zemb = self.z_transform(z)
    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
     
      temb = layers.get_timestep_embedding(timesteps, self.nf)

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if text_emb is not None: #B,77,1024
          text_emb = self.text_embedding(text_emb) #8,77,256
          # text_emb = self.text_proj(text_emb)
    if self.conditional: #在
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))#8,256
      m_idx += 1
      if text_emb is not None and temb is not None:
        q = temb[:, None, :] 
        text = self.cross_attention(q, text_emb, text_emb)  # 8,1,256
        temb = text.squeeze(1) +temb
    else:
      temb = None

    if not self.config.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.
    # labelmap_guidance = x[:,-1,:,:]
    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)] #这里同时先用一个全连接层处理三个模态
    m_idx += 1
    
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        # if text_emb is not None:
        #   print(text_emb.shape)
        #   h = modules[m_idx](hs[-1], temb, zemb, text_emb) #zemb shape:1,256
        # else:
        h = modules[m_idx](hs[-1], temb, zemb) #hs[-1]为全连接层提到的所有特征
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb, zemb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb, zemb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, text_emb)
    m_idx += 1
    h = modules[m_idx](h, temb, zemb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb, zemb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
    
    if not self.not_use_tanh:

        return torch.tanh(h)
    else:
        return h
