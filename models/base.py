import csv

import geffnet
import mmcv
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init, constant_init)

from mmcv.runner.base_module import BaseModule

from models.builder import MODULECONV, LOSSES
from models.grid_mask import GridMask
from models.ms2one import build_ms2one
from models.sparse_ins import _make_stack_3x3_convs
from mmdet3d.models import build_backbone, build_neck
@MODULECONV.register_module()
class GlobalFilter(BaseModule):
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 uncertainty_model=0, perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0):
        super().__init__()
        self.complex_weight_h = nn.Parameter(torch.randn(w, int(h/2+1), dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_l = nn.Parameter(torch.randn(w, int(h / 2 + 1), dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.uncertainty_model = uncertainty_model
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t




    def filter_spectrum(self, img_fft, ratiolist, isHighFilter):
        batch_size, h, w, _ = img_fft.shape



        # 创建一个在频率域中的掩码。
        rows = torch.linspace(-0.5, 0.5, w, dtype=torch.float32).to(img_fft.device)
        cols = torch.linspace(-0.5, 0.5, h, dtype=torch.float32).to(img_fft.device)

        cols, rows = torch.meshgrid(cols, rows)

        radius = torch.sqrt(rows ** 2 + cols ** 2).to(img_fft.device)

        # 存储每一批次处理后的 fft 图像
        img_filtered_list = []

        for i in range(batch_size):
            ratio = ratiolist[i]

            # 当 isHighFilter 为 True 时，创建高通滤波器；反之，创建低通滤波器
            if isHighFilter:
                # 在高频 (大于ratio) 中为一，在低频 (小于ratio) 中为零
                mask = (radius > ratio)
            else:
                # 在低频 (小于ratio) 中为一，在高频 (大于ratio) 中为零
                mask = (radius <= ratio)

            # 将创建的掩码转化为 PyTorch 的 tensor
            # mask_tensor = torch.tensor(mask.clone().detach(), device=img_fft.device, dtype=img_fft.dtype)
            mask_tensor=mask.clone().detach()
            # 在频率域乘以掩码来应用滤波器
            img_filtered = img_fft[i] * mask_tensor[None, :, :, None]
            img_filtered_list.append(img_filtered)

        return torch.stack(img_filtered_list)
    def forward(self, x,radio):
        B, C,a,b = x.shape

        x=rearrange(x,'b c w h-> b w h c')
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # if self.training:
        #     if self.noise_mode != 0 and self.noise_layer_flag == 1:
        #         x = self.spectrum_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
        #                                 uncertainty_model=self.uncertainty_model,
        #                                 gauss_or_uniform=self.gauss_or_uniform)
        x_h=self.filter_spectrum(x, radio[:,0],isHighFilter=True)
        x_h = torch.squeeze(x_h, dim=1)
        weight_h = torch.view_as_complex(self.complex_weight_h)
        x_h = x_h * weight_h
        x_h = torch.fft.irfft2(x_h, s=(a, b), dim=(1, 2), norm='ortho')
        x_h = rearrange(x_h, 'b w h c-> b c w h')
        x_l = self.filter_spectrum(x, radio[:,1], isHighFilter=False)
        x_l = torch.squeeze(x_l, dim=1)
        weight_l = torch.view_as_complex(self.complex_weight_l)
        x_l = x_l * weight_l
        x_l = torch.fft.irfft2(x_l, s=(a, b), dim=(1, 2), norm='ortho')
        x_l = rearrange(x_l, 'b w h c-> b c w h')
        return x,x_l,x_h


def base_block(in_filters, out_filters, normalization=False, kernel_size=3, stride=2, padding=1):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers


@MODULECONV.register_module()
class FeatureNet(BaseModule):
    def __init__(self, dim):
        super(FeatureNet, self).__init__()
        self.base_features = nn.Sequential(
            *base_block(3, 32, kernel_size=7, stride=2, padding=3, normalization=True),
            *base_block(32, 32, stride=1, normalization=True),
            *base_block(32, 32, stride=1, normalization=True),
            *base_block(32, 64, normalization=True),
            *base_block(64, 64, stride=1, normalization=True),
            *base_block(64, 64, stride=1, normalization=True),
            *base_block(64, 128, stride=1, normalization=True),
            *base_block(128, 128, stride=1, normalization=True),
            *base_block(128, 128, stride=1, normalization=True),
            *base_block(128, 256, normalization=True),
            *base_block(256, 256, stride=1, normalization=True),
            *base_block(256, 256, stride=1, normalization=True),
            *base_block(256, dim, normalization=True),
        )

    def forward(self, x):
        return self.base_features(x)


@MODULECONV.register_module()
class FeatureNetLite(BaseModule):
    def __init__(self, dim):
        super(FeatureNetLite, self).__init__()
        self.base_features = nn.Sequential(
            *base_block(3, 32, kernel_size=7, stride=2, padding=3, normalization=True),
            *base_block(32, 32, stride=1, normalization=True),
            *base_block(32, 64, normalization=True),
            *base_block(64, 64, stride=1, normalization=True),
            *base_block(64, 128, stride=1, normalization=True),
            *base_block(128, 256, normalization=True),
            *base_block(256, 256, stride=1, normalization=True),
            *base_block(256, dim, normalization=True),
        )

    def forward(self, x):
        return self.base_features(x)


@MODULECONV.register_module()
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base_features = nn.Sequential(
            *base_block(3, 16, normalization=True),
            *base_block(16, 32, normalization=True),
            *base_block(32, 64, normalization=True),
            *base_block(64, 128, normalization=True),
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            *base_block(128, 2),
            # nn.Conv2d(128, 2, 8, padding=0),
        )

        self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活函数

    def forward(self, x):
        x = F.interpolate(x, (512, 512))
        x = self.base_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.sigmoid(x)  # 使用Sigmoid激活函数
        return x


@MODULECONV.register_module()
class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=1)
        # self.end = nn.Sequential(
        #     nn.Conv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1, groups=2),
        #     nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, rgb, other):
        x, y = self.conv1(rgb), self.conv2(other)
        x = torch.cat([x, y], dim=1)
        return F.relu(x)


@MODULECONV.register_module()
class SpatialAttention(BaseModule):
    def __init__(self, embed_dim, num_chans=4, expan_att_chans=2, fusion=True):
        super(SpatialAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.isFution = fusion
        if fusion:
            self.fusionNet = Fusion(embed_dim)
        self.group_qkv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * expan_att_chans, 1, groups=embed_dim),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans),
        )
        self.group_2qkv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 2, 1, groups=embed_dim),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 2),
        )

        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x, inf):
        B, C, H, W = x.size()


        q= self.group_qkv(x).contiguous()
        k,v = self.group_2qkv(inf).contiguous().chunk(2,dim=1)
        C_exp = self.expan_att_chans * C

        # q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        #
        # k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        # v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        k, v = F.normalize(k, dim=-2), F.normalize(v, dim=-2)

        attn = q.transpose(-2, -1) @ k * self.t

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)

        x_ = rearrange(x_, "B C (H W) X -> B (C X) H W", B=B, C=self.num_heads, H=H, W=W).contiguous()
        x_ = self.group_fus(x_)
        return x_


@MODULECONV.register_module()
class DualAdaptiveNeuralBlock(BaseModule):
    def __init__(self, embed_dim, fusion=True):
        super(DualAdaptiveNeuralBlock, self).__init__()
        self.embed_dim = embed_dim
        self.fusion = fusion
        if self.fusion:
            self.pre_conv = nn.ModuleList([
                nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1) for _ in range(2)
            ])
            self.post_conv = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        else:
            self.group_conv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.Conv2d(embed_dim, embed_dim * 2, 7, 1, 3, groups=embed_dim)
            )
            self.post_conv = nn.Conv2d(embed_dim, embed_dim, 1)

    def init_weight(self):
        if self.fusion:
            for m in self.pre_conv.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, gain=0.01)
        else:
            for l in self.group_conv.modules():
                if isinstance(l, nn.Conv2d):
                    xavier_init(l, gain=0.01)
        xavier_init(self.post_conv, gain=0.01)

    def forward(self, x, inf):
        if self.fusion:
            x0, x1 = self.pre_conv[0](x), self.pre_conv[1](inf)
            x_ = F.gelu(x0) * torch.sigmoid(x1)
        else:
            B, C, H, W = x.size()
            x0, x1 = self.group_conv(x).view(B, C, 2, H, W).chunk(2, dim=2)
            x_ = F.gelu(x0.squeeze(2)) * torch.sigmoid(x1.squeeze(2))
        x_ = self.post_conv(x_)
        return x_

class ChannelAttentionAtt(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(ChannelAttentionAtt, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t

        x_ = attn.softmax(dim=-1) @ v
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


class SpatialAttentionAtt(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super(SpatialAttentionAtt, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        q, k, v = self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3,
                                                                                                                  dim=1)
        C_exp = self.expan_att_chans * C

        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * self.t

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)
        x_ = x_.transpose(-2, -1).contiguous()

        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_

@MODULECONV.register_module()
class SCC(BaseModule):
    def __init__(self, embed_dim, squeezes=(2,2), shuffle=2, expan_att_chans=2,fusion=True):
        super(SCC, self).__init__()
        self.embed_dim = embed_dim

        sque_ch_dim = embed_dim // squeezes[0]
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))
        sque_sp_dim = shuf_sp_dim // squeezes[1]

        self.sque_ch_dim = sque_ch_dim
        self.shuffle = shuffle
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim
        self.fusion=fusion
        if fusion:
            self.channel_attention=ChannelAttentionRs(sque_sp_dim,sque_ch_dim, expan_att_chans,fusion=fusion)
            self.ch_sp_squeeze2 = nn.Sequential(
                # nn.Conv2d(embed_dim, sque_ch_dim, 1),
                nn.Conv2d(embed_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
            )
        else:
            self.channel_attention = ChannelAttentionAtt(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.ch_sp_squeeze = nn.Sequential(
            # nn.Conv2d(embed_dim, sque_ch_dim, 1),
            nn.Conv2d(embed_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
        )
        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            # Add padding to maintain height of 45
            nn.PixelShuffle(shuffle),
            # nn.ZeroPad2d((0, 0, 1, 0)),  # Padding (top, bottom)
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x,feature):
        # group_num = self.sque_ch_dim
        # each_group = self.sque_sp_dim // self.sque_ch_dim
        # idx = [i + j * group_num for i in range(group_num) for j in range(each_group)]
        # nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        x = self.ch_sp_squeeze(x)
        # x = x[:, idx, :, :]
        if self.fusion:
            feature = self.ch_sp_squeeze2(feature)
            # feature = feature[:, idx, :, :]
            x=self.channel_attention(x,feature)
        else:
            x = self.channel_attention(x)
        # x = x[:, nidx, :, :]
        x = self.sp_ch_unsqueeze(x)
        return x

@MODULECONV.register_module()
class CnnFusionRs(BaseModule):
    def __init__(self, embed_dim, num_chans=8, expan_att_chans=1, fusion=True):
        super(CnnFusionRs, self).__init__()

        self.cnn = nn.Sequential(
            *base_block(embed_dim * 2, embed_dim, stride=1, normalization=True),
        )
        self.block1 = nn.Sequential(
            *base_block(embed_dim, embed_dim * 2, stride=1, normalization=True),
            *base_block(embed_dim * 2, embed_dim, stride=1, normalization=True),
        )
        self.block2 = nn.Sequential(
            *base_block(embed_dim, embed_dim * 2, stride=1, normalization=True),
            *base_block(embed_dim * 2, embed_dim, stride=1, normalization=True),
        )

    def init_weight(self):
        for l in self.cnn.modules():
            if isinstance(l, nn.Conv2d):
                xavier_init(l, gain=0.01)

    def forward(self, x, text):
        x_ = torch.cat([x, text], dim=1)
        x_ = self.cnn(x_)
        x = self.block1(x_) + x_
        x = self.block2(x) + x
        return x
@MODULECONV.register_module()
class BevFusionRs(BaseModule):
    def __init__(self, embed_dim, num_chans=8, expan_att_chans=1, fusion=True):
        super(BevFusionRs, self).__init__()
        self.embed_dim=embed_dim

        self.isFution = fusion
        self.funet=nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3,1, 1),
        )

        self.adaptive_feature_selection=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )

    def init_weight(self):
        for l in self.funet.modules():
            if isinstance(l, nn.Conv2d):
                xavier_init(l, gain=0.01)
        for l in self.adaptive_feature_selection.modules():
            if isinstance(l, nn.Conv2d):
                xavier_init(l, gain=0.01)
    def forward(self, x, text):
        if self.isFution:
           x=torch.cat((x,text),dim=1)
           x=self.funet(x)
        x=F.sigmoid(self.adaptive_feature_selection(x))*x

        return x


@MODULECONV.register_module()
class ChannelAttentionRs(nn.Module):
    def __init__(self, embed_dim, num_chans=8, expan_att_chans=2, fusion=True):
        super(ChannelAttentionRs, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))


        self.isFution = fusion
        if self.isFution:
            self.group_qkv_text = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim*expan_att_chans * 2, 1, groups=embed_dim),
                nn.Conv2d(embed_dim *expan_att_chans* 2, embed_dim*expan_att_chans, 1, groups=embed_dim),
                nn.Conv2d(embed_dim*expan_att_chans , embed_dim*expan_att_chans * 2, 1, groups=embed_dim),
                Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans *2),
            )
            self.group_qkv_rgb = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim*expan_att_chans * 2, 1, groups=embed_dim),
                nn.Conv2d(embed_dim*expan_att_chans * 2, embed_dim*expan_att_chans, 1, groups=embed_dim),
                nn.Conv2d( embed_dim*expan_att_chans, embed_dim *expan_att_chans, 1, groups=embed_dim),
                Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans),
            )
        else:
            self.group_qkv_rgb = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 2, 1, groups=embed_dim),
                nn.Conv2d(embed_dim * expan_att_chans * 2, embed_dim * expan_att_chans, 1, groups=embed_dim),
                nn.Conv2d(embed_dim * expan_att_chans, embed_dim * expan_att_chans, 1, groups=embed_dim),
                nn.Conv2d(embed_dim * expan_att_chans, embed_dim* expan_att_chans * 3, 1, groups=embed_dim),
                Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans *3),
            )
        self.group_fus = nn.Sequential(*base_block(embed_dim*expan_att_chans, embed_dim, stride=1))

    def init_weight(self):
        for l in self.group_qkv_text.modules():
            if isinstance(l, nn.Conv2d):
                xavier_init(l, gain=0.01)
        for l in self.group_qkv_rgb.modules():
            if isinstance(l, nn.Conv2d):
                xavier_init(l, gain=0.01)
        xavier_init(self.group_fus, gain=0.01)

    def attnfun(self, q, k, v, t):
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * t
        return attn.softmax(dim=-1) @ v

    def forward(self, x, to):
        B, C, H, W = x.size()
        C_exp=self.expan_att_chans*C
        if self.isFution:
            q= self.group_qkv_rgb(x).contiguous()
            k, v = self.group_qkv_text(to).contiguous().chunk(2, dim=1)

            q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
            k =  k.view(B, self.num_heads,   C_exp // self.num_heads, H * W)
            v =v.view(B, self.num_heads,  C_exp // self.num_heads,  H * W)
            x_ = self.attnfun(q, k, v, self.t)
        else:
            qRGB, kRGB, vRGB = self.group_qkv_rgb(x).contiguous().chunk(3, dim=1)
            qRGB = qRGB.view(B, self.num_heads, C_exp // self.num_heads, H * W)
            kRGB = kRGB.view(B, self.num_heads, C_exp // self.num_heads, H * W)
            vRGB = vRGB.view(B, self.num_heads, C_exp // self.num_heads, H * W)
            x_ = self.attnfun(qRGB, kRGB, vRGB, self.t)
        x_ = rearrange(x_, "B C X (H W)-> B (X C) H W", B=B, W=W, H=H, C=self.num_heads).contiguous()
        x_ = x + self.group_fus(x_)
        return x_


@MODULECONV.register_module()
class FGTransformer(nn.Module):
    def __init__(self, embed_dim):
        super(FGTransformer, self).__init__()
        self.embed_dim = embed_dim

        self.FCNet = ChannelAttentionRs(embed_dim)

        self.FGNet = DualAdaptiveNeuralBlock(embed_dim)

    def initweight(self):
        self.FGNet.init_weight()
        self.FCNet.init_weight()

    def forward(self, batch):
        x, inf = batch
        x = self.FCNet(x, inf) + x
        x = self.FGNet(x, inf) + x
        return x, inf


def write_losses(file_name, losses_accu, epoch):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 0:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


def write_val(file_name, losses_accu, epoch):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 0:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


@MODULECONV.register_module()
class IAMBlock(BaseModule):
    def __init__(self, num_group, num_mask, embed_dims, sparse_num_group):
        super(IAMBlock, self).__init__()
        self.num_group = num_group
        self.num_mask = num_mask
        self.sparse_num_group = sparse_num_group
        self.inst_convs = _make_stack_3x3_convs(
            num_convs=3,
            in_channels=embed_dims + 2,
            out_channels=embed_dims)
        self.iam_conv = nn.Conv2d(
            embed_dims * num_group,
            num_group * num_mask * sparse_num_group,
            3, padding=1, groups=num_group * sparse_num_group)
        self.fc = nn.Linear(embed_dims * sparse_num_group, embed_dims)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, seg_features, is_training=True):
        coord_features = self.compute_coordinates(seg_features)
        seg_features = torch.cat([coord_features, seg_features], dim=1)
        seg_features = self.inst_convs(seg_features)
        iam = self.iam_conv(seg_features.tile(
            (1, self.num_group, 1, 1)))
        num_group = self.num_group
        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]
        C = seg_features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob_norm_hw = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        # (B x N x HW) @ (B x HW x C) -> B x N x C
        all_inst_features = torch.bmm(
            iam_prob_norm_hw,
            seg_features.view(B, C, -1).permute(0, 2, 1))  # BxNxC

        # concat sparse group features
        inst_features = all_inst_features.reshape(
            B, num_group,
            self.sparse_num_group,
            self.num_mask, -1
        ).permute(0, 1, 3, 2, 4).reshape(
            B, num_group,
            self.num_mask, -1)
        inst_features = F.relu_(
            self.fc(inst_features))
        inst_features = inst_features.flatten(1, 2)
        return inst_features


class CondensedAttentionNeuralBlock(nn.Module):
    def __init__(self, embed_dim, squeezes, shuffle, expan_att_chans, caOpt):
        '''

        :param embed_dim: 嵌入维度
        :param squeezes: 压缩因子列表
        :param shuffle: 像素混洗因子
        :param expan_att_chans: 扩展注意力通道数
        :param caOpt:
        '''
        super(CondensedAttentionNeuralBlock, self).__init__()
        self.embed_dim = embed_dim  # 16
        self.caOpt = caOpt
        sque_ch_dim = embed_dim // squeezes[0]  # 16/4
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))  # 4*16**2=1024
        sque_sp_dim = shuf_sp_dim // squeezes[1]  # 128

        self.sque_ch_dim = sque_ch_dim  # 4
        self.shuffle = shuffle  # 16
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        # 将输入张量进行通道和空间维度的压缩
        self.ch_sp_squeeze = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, sque_ch_dim, 1),
                nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
            ) for _ in range(2)
        ])

        self.g1 = MODEL_REGISTRY.get(caOpt.name)(sque_sp_dim, sque_ch_dim, expan_att_chans, caOpt)
        # 其中包含两个卷积层和一个像素混洗层。这些层用于将输入张量进行空间和通道维度的展开。
        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x, text):
        x = self.ch_sp_squeeze[0](x)
        text = self.ch_sp_squeeze[1](text)

        # group_num = self.sque_ch_dim
        # each_group = self.sque_sp_dim // self.sque_ch_dim
        # idx = [i + j * group_num for i in range(group_num) for j in range(
        #     each_group)]  # 外层循环range(group_num)控制了组的数量，内层循环range(each_group)控制了每个组中的元素数量。通过这两个循环，生成了一个长度为group_num * each_group的索引列表。
        # x = x[:, idx, :, :]
        # text = text[:, idx, :, :]

        x = self.g1(x, text)
        # nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        # x = x[:, nidx, :, :]

        x = self.sp_ch_unsqueeze(x)
        return x


@MODULECONV.register_module()
class TransformerBlock(BaseModule):
    def __init__(self,
                 embed_dim,
                 squeezes,
                 shuffle,
                 expan_att_chans,
                 ca_cfg,
                 da_cfg
                 ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.ModuleList([
            nn.Sequential(
                Rearrange('B C H W -> B (H W) C'),
                nn.LayerNorm(embed_dim)
            ) for _ in range(2)
        ])
        # ChannelAttentionRs
        sque_ch_dim = embed_dim // squeezes[0]  # 16/4
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))  # 4*16**2=1024
        sque_sp_dim = shuf_sp_dim // squeezes[1]  # 128
        self.sque_ch_dim = sque_ch_dim  # 4
        self.shuffle = shuffle  # 16
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim
        self.ch_sp_squeeze = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, sque_ch_dim, 1),
                nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
            ) for _ in range(2)
        ])
        self.ca = MODULECONV.get(ca_cfg.type)(sque_sp_dim, sque_ch_dim, expan_att_chans, ca_cfg.fusion)
        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )
        # self.ca = CondensedAttentionNeuralBlock(embed_dim, squeezes, shuffle, expan_att_chans, transformerOpt.ca)
        self.norm2 = nn.ModuleList([
            nn.Sequential(
                Rearrange('B C H W -> B (H W) C'),
                nn.LayerNorm(embed_dim)
            ) for _ in range(2)
        ])
        # MODULECONV.get()
        self.da = MODULECONV.get(da_cfg.type)(embed_dim, da_cfg.fusion)

    def forward(self, batch):
        x, text, inf=batch
        B, C, H, W = x.size()
        x_ = rearrange(self.norm1[0](x), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_text = rearrange(self.norm1[1](text), "B (H W) C -> B C H W", H=H, W=W).contiguous()

        x_ = self.ch_sp_squeeze[0](x_)
        x_text = self.ch_sp_squeeze[1](x_text)
        x_ = self.ca(x_, x_text)
        x_ = self.sp_ch_unsqueeze(x_)

        x = x + x_

        x_ = rearrange(self.norm2[0](x), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_inf = rearrange(self.norm2[1](inf), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x = x + self.da(x_, x_inf)
        return x, text, inf
@MODULECONV.register_module()
class FilteNet(BaseModule):
    def __init__(self,fre_backone=None,gf=None):
        super(FilteNet, self).__init__()
        self.filterNet = build_backbone(fre_backone)
        self.gf = MODULECONV.build(gf)
        self.filterHead = nn.Sequential(
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            *base_block(512, 2),
            nn.Sigmoid()
        )
        self.lNet=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.hNet=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )


    def forward(self, image):
        filterWeight = self.filterNet(image)[-1]
        filterWeight = self.filterHead(filterWeight)
        filterWeight = filterWeight.squeeze()
        _, img_Low, img_High = self.gf(image, filterWeight)
        img_Low=self.lNet(image)*torch.sigmoid(img_Low)+image*torch.sigmoid(-img_Low)
        img_High=self.hNet(image)*torch.sigmoid(img_High)

        return img_Low,img_High

@MODULECONV.register_module()
class DTFBackbone(BaseModule):
    def __init__(self, pts_backbone=None,ms2one=None,neck=None,pts_neck=None,img_backbone=None):
        super(DTFBackbone, self).__init__()

        # checkpoint = torch.load('pretrained_models/feq.pth.tar', map_location='cpu')
        # self.feq.load_state_dict(checkpoint['state_dict'])
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.encoder = MODULECONV.build(img_backbone)
        self.neck = build_neck(neck)
        self.ms2one = build_ms2one(ms2one)
        self.encoder.init_weights()

        self.dep_encoder =MODULECONV.build(img_backbone)
        self.dep_neck = build_neck(neck)
        self.dep_ms2one = build_ms2one(ms2one)
        self.dep_encoder.init_weights()

        self.pt_encoder = MODULECONV.build(pts_backbone)
        self.pt_neck = MODULECONV.build(pts_neck)
        self.pt_ms2one = build_ms2one(ms2one)
        self.pt_encoder.init_weights()


    def transform_to_lidar(self,point_cloud_cam, lidar_to_cam):
        # Convert to homogeneous coordinates
        point_cloud_cam_hom = torch.cat([point_cloud_cam, torch.ones(point_cloud_cam.shape[0], 1)], dim=1)

        # Convert lidar_to_cam to torch tensor
        lidar_to_cam = torch.from_numpy(lidar_to_cam)

        # Do the transformation, the torch.inverse function is used to get the inverse of lidar_to_cam
        point_cloud_lidar = torch.matmul(torch.inverse(lidar_to_cam), point_cloud_cam_hom.t()).t()

        # Convert back to Euclidean coordinates from homogeneous coordinates
        point_cloud_lidar = point_cloud_lidar[:, :3]

        return point_cloud_lidar




    def forward(self, image,dep_image,point):
        image=self.grid_mask(image)

        out_featList = self.encoder(image)
        neck_out = self.neck(out_featList)
        neck_out = self.ms2one(neck_out)

        dep_image = self.grid_mask(dep_image)
        dep_image = self.dep_encoder(dep_image)
        dep_image = self.dep_neck(dep_image)
        dep_image = self.dep_ms2one(dep_image)

        point = self.grid_mask(point)
        point = self.pt_encoder(point)
        point = self.pt_neck(point)

        return neck_out,dep_image,point
@MODULECONV.register_module()
class DTFBackbone2(BaseModule):
    def __init__(self, encoder=None,ms2one=None,neck=None,h=960,w=640,dim=128):
        super(DTFBackbone, self).__init__()
        self.filterNet=build_backbone(encoder)
        self.filterHead=nn.Sequential(
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            *base_block(512, 2),
            nn.Sigmoid()
        )
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.encoder = build_backbone(encoder)
        self.encoderH = build_backbone(encoder)
        self.encoderL = build_backbone(encoder)
        self.neck = build_neck(neck)
        self.neckH = build_neck(neck)
        self.neckL = build_neck(neck)
        self.encoder.init_weights()
        self.filterNet.init_weights()
        self.encoderH.init_weights()
        self.encoderL.init_weights()

        self.ms2one = build_ms2one(ms2one)
        self.ms2oneH = build_ms2one(ms2one)
        self.ms2oneL = build_ms2one(ms2one)
    @torch.no_grad()
    def low_pass_filter(self, radiusRadios, img, isLow=True):
        assert len(radiusRadios.shape) == 1
        assert len(img.shape) == 4  # batch_size, num_channels, height, width
        device = img.device

        # Get image size
        batch_size, num_channels, h, w = img.shape

        # Output image
        img_back = torch.zeros_like(img)

        for b in range(batch_size):
            radiusRadio = radiusRadios[b]

            # Create mask for filtering
            radius = torch.round(min(h, w) * radiusRadio).int()
            y, x = torch.meshgrid(torch.arange(-h // 2, h // 2), torch.arange(-w // 2, w // 2))
            # Move x and y tensors to the same device as mask
            x = x.to(device).contiguous()
            y = y.to(device).contiguous()
            if isLow:
                mask = ((x * x + y * y <= radius * radius)).float().to(device)  # low filter
            else:
                mask = ((x * x + y * y > radius * radius)).float().to(device)  # high filter
            # 获取张量的最大值和最小值
            for c in range(num_channels):
                # Apply FFT to each channel
                img_fft = torch.fft.fftn(img[b, c], dim=(0, 1))

                # Apply low pass filter
                img_fft_shift = torch.fft.fftshift(img_fft, dim=(0, 1))
                img_low_pass = img_fft_shift * mask

                img_low_pass = torch.fft.ifftshift(img_low_pass, dim=(0, 1))

                # Inverse FFT to get filtered image and copy to output tensor
                img_back[b, c] = torch.fft.ifftn(img_low_pass, dim=(0, 1))

        # Take absolute value
        img_back = torch.abs(img_back).float().contiguous()

        return img_back

    def forward(self, image):
        filterWeight = self.filterNet(image)[-1]
        filterWeight=self.filterHead(filterWeight)
        filterWeight = torch.clamp(filterWeight, min=0.35, max=0.65)
        filterWeight = filterWeight.squeeze()
        img_Low = self.low_pass_filter(filterWeight[:, 0], image)
        img_High = self.low_pass_filter(filterWeight[:, 1], image, isLow=False)

        out_featList = self.encoder(image)
        neck_out = self.neck(out_featList)
        neck_out = self.ms2one(neck_out)

        out_img_Low = self.encoderL(img_Low)
        neck_low_out = self.neckL(out_img_Low)
        neck_low_out = self.ms2oneL(neck_low_out)

        out_img_High = self.encoderH(img_High)
        neck_high_out = self.neckH(out_img_High)
        neck_high_out = self.ms2oneH(neck_high_out)
        return neck_out,neck_low_out,neck_high_out
@MODULECONV.register_module()
class DTFormer(BaseModule):
    def __init__(self,
                 in_chans,
                 embed_dim,
                 num_blocks,
                 ch_sp_squeeze,
                 num_shuffles,
                 expan_att_chans,
                 filter_cfg=dict(type='CustomModel'),
                 gf_cfg=None,
                 block=None
                 ):
        super(DTFormer, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
        self.patch_embed2 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
        embed_dim = int(embed_dim * 4)
        self.filterNet = MODULECONV.build(filter_cfg)
        if gf_cfg:
            self.gf=MODULECONV.build(gf_cfg)
        self.encoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans, block.hf_cfg,block.lf_cfg) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))])
        self.downsampler = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i - 1)), 3, 1, 1),
                    nn.PixelUnshuffle(2)
                ) for i in range(len(num_blocks) - 1)
            ]).append(nn.Identity()) for _ in range(3)
        ])

    @torch.no_grad()
    def low_pass_filter(self, radiusRadios, img, isLow=True):
        assert len(radiusRadios.shape) == 1
        assert len(img.shape) == 4  # batch_size, num_channels, height, width
        device = img.device

        # Get image size
        batch_size, num_channels, h, w = img.shape

        # Output image
        img_back = torch.zeros_like(img)

        for b in range(batch_size):
            radiusRadio = radiusRadios[b]

            # Create mask for filtering
            radius = torch.round(min(h, w) * radiusRadio).int()
            y, x = torch.meshgrid(torch.arange(-h // 2, h // 2), torch.arange(-w // 2, w // 2))
            # Move x and y tensors to the same device as mask
            x = x.to(device).contiguous()
            y = y.to(device).contiguous()
            if isLow:
                mask = ((x * x + y * y <= radius * radius)).float().to(device)  # low filter
            else:
                mask = ((x * x + y * y > radius * radius)).float().to(device)  # high filter
            # 获取张量的最大值和最小值
            for c in range(num_channels):
                # Apply FFT to each channel
                img_fft = torch.fft.fftn(img[b, c], dim=(0, 1))

                # Apply low pass filter
                img_fft_shift = torch.fft.fftshift(img_fft, dim=(0, 1))
                img_low_pass = img_fft_shift * mask

                img_low_pass = torch.fft.ifftshift(img_low_pass, dim=(0, 1))

                # Inverse FFT to get filtered image and copy to output tensor
                img_back[b, c] = torch.fft.ifftn(img_low_pass, dim=(0, 1))

        # Take absolute value
        img_back = torch.abs(img_back).float().contiguous()

        return img_back

    def forward(self, image):
        filterWeight = self.filterNet(image)
        filterWeight = filterWeight.squeeze()
        img_Low = self.low_pass_filter(filterWeight[:, 0], image)
        img_High = self.low_pass_filter(filterWeight[:, 1], image, isLow=False)
        x_emb = self.patch_embed(image)
        l_emb = self.patch_embed1(img_Low)
        h_emb = self.patch_embed2(img_High)
        x_ = x_emb
        x_l = l_emb
        x_h = h_emb
        x_ms = []

        for layer, sampler, sampler1, sampler2 in zip(
                self.encoder, self.downsampler[0], self.downsampler[1], self.downsampler[2]
        ):
            x_, _, _ = layer((x_, x_l, x_h))
            x_ms.append((x_, x_l, x_h))
            x_, x_l, x_h = sampler(x_), sampler1(x_l), sampler2(x_h)
        return x_ms
