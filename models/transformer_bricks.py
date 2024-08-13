import numpy as np
import math
import warnings

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init, constant_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from .builder import MODULECONV
from .scatter_utils import scatter_mean
from .utils import inverse_sigmoid
import copy
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def generate_ref_pt(minx, miny, maxx, maxy, z, nx, ny, device='cuda'):
    if isinstance(z, list):
        nz = z[-1]
        # minx, miny, maxx, maxy : in ground coords
        xs = torch.linspace(minx, maxx, nx, dtype=torch.float, device=device
                            ).view(1, -1, 1).expand(ny, nx, nz)
        ys = torch.linspace(miny, maxy, ny, dtype=torch.float, device=device
                            ).view(-1, 1, 1).expand(ny, nx, nz)
        zs = torch.linspace(z[0], z[1], nz, dtype=torch.float, device=device
                            ).view(1, 1, -1).expand(ny, nx, nz)
        ref_3d = torch.stack([xs, ys, zs], dim=-1)
        ref_3d = ref_3d.flatten(1, 2)
    else:
        # minx, miny, maxx, maxy : in ground coords
        xs = torch.linspace(minx, maxx, nx, dtype=torch.float, device=device
                            ).view(1, -1, 1).expand(ny, nx, 1)
        ys = torch.linspace(miny, maxy, ny, dtype=torch.float, device=device
                            ).view(-1, 1, 1).expand(ny, nx, 1)
        ref_3d = F.pad(torch.cat([xs, ys], dim=-1), (0, 1), mode='constant', value=z)
    return ref_3d


def ground2img(coords3d, H, W, lidar2img, ori_shape, mask=None, return_img_pts=False):
    coords3d = coords3d.clone()
    img_pt = coords3d.flatten(1, 2) @ lidar2img.permute(0, 2, 1)
    img_pt = torch.cat([
        img_pt[..., :2] / torch.maximum(
            img_pt[..., 2:3], torch.ones_like(img_pt[..., 2:3]) * 1e-5),
        img_pt[..., 2:]
    ], dim=-1)

    # rescale to feature_map size
    x = img_pt[..., 0] / ori_shape[0][1] * (W - 1)
    y = img_pt[..., 1] / ori_shape[0][0] * (H - 1)
    valid = (x >= 0) * (y >= 0) * (x <= (W - 1)) \
            * (y <= (H - 1)) * (img_pt[..., 2] > 0)
    if return_img_pts:
        return x, y, valid

    if mask is not None:
        valid = valid * mask.flatten(1, 2).float()

    # B, C, H, W = img_feats.shape
    B = coords3d.shape[0]
    canvas = torch.zeros((B, H, W, 3 + 1),
                         dtype=torch.float32,
                         device=coords3d.device)
    x = x.long()
    y = y.long()
    ind = (x + y * W) * valid.long()
    # ind = torch.clamp(ind, 0, H * W - 1)
    ind = ind.long().unsqueeze(-1).repeat(1, 1, canvas.shape[-1])
    canvas = canvas.flatten(1, 2)
    target = coords3d.flatten(1, 2).clone()
    scatter_mean(target, ind, out=canvas, dim=1)
    canvas = canvas.view(B, H, W, canvas.shape[-1]
                         ).permute(0, 3, 1, 2).contiguous()
    canvas[:, :, 0, 0] = 0
    return canvas
def ground2img2(coords3d, H, W, lidar2img, ori_shape, mask=None, return_img_pts=False):
    coords3d = coords3d.clone().sigmoid()
    batch_size, num_points_x, num_points_y, _ = coords3d.shape
    ori_H, ori_W = ori_shape[0][0], ori_shape[0][1]
    output_features = torch.zeros((batch_size, 4, H, W), dtype=torch.float32, device=coords3d.device)

    for b in range(batch_size):
        coords = coords3d[b].view(-1, 4)  # Flatten the 3D points to a 2D array (num_points, 4)
        proj_matrix = lidar2img[b]  # Get the projection matrix for this batch
        proj_coords = proj_matrix @ coords.t()  # Apply the projection matrix
        proj_coords = proj_coords.t()  # Transpose back to (num_points, 4)

        # Normalize the projected coordinates
        proj_coords = proj_coords / proj_coords[:, 3].unsqueeze(-1)

        # Map the projected coordinates to the image plane
        x = proj_coords[:, 0]
        y = proj_coords[:, 1]

        # Convert from homogeneous coordinates to image coordinates
        x_img = ((x + 1) * ori_W) / 2
        y_img = ((y + 1) * ori_H) / 2

        # Normalize the coordinates to the range [0, 1] for grid_sample
        x_img = x_img / ori_W
        y_img = y_img / ori_H

        # Stack the coordinates and add a batch dimension
        grid = torch.stack((x_img, y_img), dim=1)

        # Ensure the coordinates are within the image bounds
        grid = torch.clamp(grid, 0, 1)

        # Normalize grid values from [0, 1] to [-1, 1] for grid_sample
        grid = grid * 2 - 1

        # Reshape the grid to (1, num_points_x * num_points_y, 1, 2)
        grid = grid.view(1, num_points_x * num_points_y, 1, 2)

        # Sample the features using the grid
        sampled_features = F.grid_sample(
            coords3d[b].permute(2, 0, 1).unsqueeze(0),
            grid,
            mode='bilinear',
            align_corners=False
        )

        # Reshape sampled features to match output shape
        sampled_features = sampled_features.view(4, num_points_x, num_points_y)

        # Interpolate to the target size
        sampled_features = F.interpolate(sampled_features.unsqueeze(0), size=(H, W), mode='bilinear',
                                         align_corners=False)

        output_features[b] = sampled_features.squeeze(0)

    return output_features

@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 d3type='dc',
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.d3type = d3type
        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.register_buffer('anchor_y_steps',
                             torch.from_numpy(anchor_y_steps).float())
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points * 2 * self.num_points_per_anchor)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points * self.num_points_per_anchor)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1,
            2).repeat(1, self.num_points_per_anchor, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[..., i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def ref_to_lidar(self, reference_points, pc_range, not_y=True):
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        if not not_y:
            reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                         (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]
        return reference_points

    def point_sampling(self, reference_points, lidar2img, ori_shape):
        x, y, mask = ground2img(
            reference_points, H=2, W=2,
            lidar2img=lidar2img, ori_shape=ori_shape,
            mask=None, return_img_pts=True)
        return torch.stack([x, y], -1), mask

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pc_range=None,
                lidar2img=None,
                pad_shape=None,
                key_pos=None,
                **kwargs):
        if value is None:
            assert False
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            value = value + key_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_points_per_anchor,
            self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_points_per_anchor,
            self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_points_per_anchor,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        not_y = True
        if self.d3type!='petr':
            reference_points = reference_points.view(
                bs, self.num_query, self.num_anchor_per_query, -1, 2)
            ref_pt3d = torch.cat([
                reference_points[..., 0:1],  # x
                self.anchor_y_steps.view(1, 1, self.num_anchor_per_query, -1, 1
                                         ).expand_as(reference_points[..., 0:1]),  # y
                reference_points[..., 1:2]  # z
            ], dim=-1)
        else:
            ref_pt3d=reference_points.view(
                bs, self.num_query, self.num_anchor_per_query, -1, 3)

        sampling_locations = self.ref_to_lidar(ref_pt3d, pc_range, not_y=not_y)
        sampling_locations2d, _ = self.point_sampling(
            F.pad(sampling_locations.flatten(1, 2), (0, 1), value=1),
            lidar2img=lidar2img, ori_shape=pad_shape,
        )
        sampling_locations2d = sampling_locations2d.view(
            *sampling_locations.shape[:-1], 2)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_offsets = sampling_offsets / \
                           offset_normalizer[None, None, None, :, None, :]

        sampling_locations2d = sampling_locations2d.view(
            bs, self.num_query, self.num_anchor_per_query, -1, 1, 1, 1, 2) \
                               + sampling_offsets.view(
            bs, self.num_query, self.num_anchor_per_query, self.num_points_per_anchor,
            *sampling_offsets.shape[3:]
        )

        # reshape, move self.num_anchor_per_query to last axis
        sampling_locations2d = sampling_locations2d.permute(0, 1, 2, 4, 5, 6, 3, 7)
        attention_weights = attention_weights.permute(0, 1, 3, 4, 5, 2)
        sampling_locations2d = sampling_locations2d.flatten(-3, -2)
        attention_weights = attention_weights.flatten(-2) / self.num_points_per_anchor

        xy = 2
        num_all_points = sampling_locations2d.shape[-2]

        sampling_locations2d = sampling_locations2d.view(
            bs, num_query, self.num_heads, self.num_levels, num_all_points, xy)

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations2d,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations2d, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output


@TRANSFORMER_LAYER.register_module()
class DCGDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DCGDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        query = super().forward(
            query=query, key=key, value=value,
            query_pos=query_pos, key_pos=key_pos,
            attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask, **kwargs)
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DCGTransformerDecoder(TransformerLayerSequence):
    def __init__(self,
                 *args, embed_dims=None,
                 post_norm_cfg=dict(type='LN'),
                 enlarge_length=10,
                 M_decay_ratio=10,
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 **kwargs):
        super(DCGTransformerDecoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query

        self.embed_dims = embed_dims
        self.gflat_pred_layer = nn.Sequential(
            nn.Conv2d(embed_dims + 4, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims // 4, 1),
            nn.BatchNorm2d(embed_dims // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dims // 4, 2, 1))

        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.M_decay_ratio = M_decay_ratio
        self.enlarge_length = enlarge_length

    def init_weights(self):
        super().init_weights()
        for l in self.gflat_pred_layer:
            xavier_init(l, gain=0.01)

    def pred2M(self, pitch_z):
        pitch_z = pitch_z / self.M_decay_ratio
        t = pitch_z[:, 0] / 100
        z = pitch_z[:, 1]
        one = torch.ones_like(t)
        zero = torch.zeros_like(t)

        # rot first, then translate
        M = torch.stack([
            one, zero, zero, zero,
            zero, t.cos(), -t.sin(), zero,
            zero, t.sin(), t.cos(), z,
            zero, zero, zero, one], dim=-1).view(t.shape[0], 4, 4)
        return M

    def forward(self, query, key, value,
                top_view_region=None, z_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,

                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None,
                query_pos=None,
                **kwargs):

        # init pts and M to generate pos embed for key/value
        batch_size = query.shape[1]
        xmin = top_view_region[0][0] - self.enlarge_length
        xmax = top_view_region[1][0] + self.enlarge_length
        ymin = top_view_region[2][1] - self.enlarge_length
        ymax = top_view_region[0][1] + self.enlarge_length
        zmin = z_region[0]
        zmax = z_region[1]
        ref_3d_homo = generate_ref_pt(
            xmin, ymin, xmax, ymax, init_z,
            bev_w, bev_h, query.device)
        ref_3d_homo = ref_3d_homo[None, ...].repeat(batch_size, 1, 1, 1)
        ref_3d_homo = F.pad(ref_3d_homo, (0, 1), value=1)
        init_ref_3d_homo = ref_3d_homo.clone()
        M = torch.eye(4, device=query.device).float()
        M = M[None, ...].repeat(batch_size, 1, 1)

        intermediate = []
        project_results = []
        outputs_classes = []
        outputs_coords = []
        for layer_idx, layer in enumerate(self.layers):
            coords_img = ground2img(
                ref_3d_homo, *img_feats[0].shape[-2:],
                lidar2img, pad_shape)
            if layer_idx > 0:
                project_results.append(coords_img.clone())
            coords_img_key_pos = coords_img.clone()
            ground_coords = coords_img_key_pos[:, :3, ...]

            ground_coords[:, 0, ...] = (ground_coords[:, 0, ...] - xmin) / (xmax - xmin)
            ground_coords[:, 1, ...] = (ground_coords[:, 1, ...] - ymin) / (ymax - ymin)
            ground_coords[:, 2, ...] = (ground_coords[:, 2, ...] - zmin) / (zmax - zmin)
            ground_coords = inverse_sigmoid(ground_coords)
            key_pos = self.position_encoder(ground_coords)

            query = layer(query, key=key, value=value,
                          key_pos=(key_pos + sin_embed
                                   ).flatten(2, 3).permute(2, 0, 1).contiguous(),
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)

            # update M
            if layer_idx < len(self.layers) - 1:
                input_feat = torch.cat([img_feats[0], coords_img], dim=1)
                M = M.detach() @ self.pred2M(self.gflat_pred_layer(input_feat).squeeze(-1).squeeze(-1))
                ref_3d_homo = (init_ref_3d_homo.flatten(1, 2) @ M.permute(0, 2, 1)).view(*ref_3d_homo.shape)

            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

            query = query.permute(1, 0, 2)
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                           self.num_anchor_per_query, -1, 3)
            reference_points = reference_points.view(
                bs, self.num_query, self.num_anchor_per_query,
                self.num_points_per_anchor, 2
            )
            reference_points = inverse_sigmoid(reference_points)
            new_reference_points = torch.stack([
                reference_points[..., 0] + tmp[..., 0],
                reference_points[..., 1] + tmp[..., 1],
            ], dim=-1)
            reference_points = new_reference_points.sigmoid()

            cls_feat = query.view(bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)
            outputs_coords.append(torch.cat([
                reference_points, tmp[..., -1:]
            ], dim=-1))

            reference_points = reference_points.view(
                bs, self.num_query * self.num_anchor_per_query,
                    self.num_points_per_anchor * 2
            ).detach()
            query = query.permute(1, 0, 2)
        return torch.stack(intermediate), project_results, outputs_classes, outputs_coords


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DCGTransformerDecoderV2(TransformerLayerSequence):
    def __init__(self,
                 *args, embed_dims=None,
                 post_norm_cfg=dict(type='LN'),
                 enlarge_length=10,
                 M_decay_ratio=10,
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 **kwargs):
        super(DCGTransformerDecoderV2, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query

        self.embed_dims = embed_dims
        self.gflat_pred_layer = nn.Sequential(
            nn.Conv2d(embed_dims + 4, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims // 4, 1),
            nn.BatchNorm2d(embed_dims // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dims // 4, 2, 1))

        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.M_decay_ratio = M_decay_ratio
        self.enlarge_length = enlarge_length

    def init_weights(self):
        super().init_weights()
        for l in self.gflat_pred_layer:
            xavier_init(l, gain=0.01)

    def pred2M(self, pitch_z):
        pitch_z = pitch_z / self.M_decay_ratio
        t = pitch_z[:, 0] / 100
        z = pitch_z[:, 1]
        one = torch.ones_like(t)
        zero = torch.zeros_like(t)

        # rot first, then translate
        M = torch.stack([
            one, zero, zero, zero,
            zero, t.cos(), -t.sin(), zero,
            zero, t.sin(), t.cos(), z,
            zero, zero, zero, one], dim=-1).view(t.shape[0], 4, 4)
        return M

    def forward(self, query, key, value,
                top_view_region=None, z_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,

                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None,
                query_pos=None,
                point_org=None,
                **kwargs):

        xmin = top_view_region[0][0] - self.enlarge_length
        xmax = top_view_region[1][0] + self.enlarge_length
        ymin = top_view_region[2][1] - self.enlarge_length
        ymax = top_view_region[0][1] + self.enlarge_length
        zmin = z_region[0]
        zmax = z_region[1]

        intermediate = []

        outputs_classes = []
        outputs_coords = []
        for layer_idx, layer in enumerate(self.layers):

            ground_coords_d3 = point_org.clone()

            ground_coords_d3[:, 1, ...] = (ground_coords_d3[:, 1, ...] - xmin) / (xmax - xmin)
            ground_coords_d3[:, 2, ...] = (ground_coords_d3[:, 2, ...] - ymin) / (ymax - ymin)
            ground_coords_d3[:, 0, ...] = (ground_coords_d3[:, 0, ...] - zmin) / (zmax - zmin)
            ground_coords_d3 = F.interpolate(ground_coords_d3, img_feats[0].shape[-2:], mode='bilinear',
                                             align_corners=False)
            ground_coords_d3 = inverse_sigmoid(ground_coords_d3)

            key_pos = img_feats[0]

            # ground_coords = inverse_sigmoid(ground_coords)
            # key_pos = self.position_encoder(ground_coords)

            query = layer(query, key=key, value=value,
                          key_pos=(key_pos + sin_embed
                                   ).flatten(2, 3).permute(2, 0, 1).contiguous(),
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)

            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

            query = query.permute(1, 0, 2)
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                           self.num_anchor_per_query, -1, 3)
            reference_points = reference_points.view(
                bs, self.num_query, self.num_anchor_per_query,
                self.num_points_per_anchor, 2
            )
            reference_points = inverse_sigmoid(reference_points)
            new_reference_points = torch.stack([
                reference_points[..., 0] + tmp[..., 0],
                reference_points[..., 1] + tmp[..., 1],
            ], dim=-1)
            reference_points = new_reference_points.sigmoid()

            cls_feat = query.view(bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)
            outputs_coords.append(torch.cat([
                reference_points, tmp[..., -1:]
            ], dim=-1))

            reference_points = reference_points.view(
                bs, self.num_query * self.num_anchor_per_query,
                    self.num_points_per_anchor * 2
            ).detach()
            query = query.permute(1, 0, 2)
        return torch.stack(intermediate), None, outputs_classes, outputs_coords


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransformerLayerSequenceV2(BaseModule):
    """Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super(TransformerLayerSequenceV2, self).__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.layers2 = ModuleList()
        for i in range(num_layers):
            self.layers2.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DCGTransformerDecoderV3(TransformerLayerSequenceV2):
    def __init__(self,
                 *args, embed_dims=None,
                 post_norm_cfg=dict(type='LN'),
                 enlarge_length=10,
                 M_decay_ratio=10,
                 num_query=None,
                 num_classes=None,
                 d3type=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 **kwargs):
        super(DCGTransformerDecoderV3, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.d3type = d3type
        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query



        self.embed_dims = embed_dims
        self.gflat_pred_layer = nn.Sequential(
            nn.Conv2d(embed_dims*2 + 4, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims // 4, 1),
            nn.BatchNorm2d(embed_dims // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dims // 4, 2, 1))
        self.position_encoder_ground = nn.Sequential(
            nn.Conv2d(3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.position_encoder2 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.ref_layer = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 4),
            nn.ReLU(True)
        )
        self.padding_layer = nn.ZeroPad2d((0, 0, 0, 1))
        self.M_decay_ratio = M_decay_ratio
        self.enlarge_length = enlarge_length
        self.cls_score = nn.Linear(embed_dims, num_classes)
        self.mask_kernel = nn.Linear(embed_dims, embed_dims)

    def init_weights(self):
        super().init_weights()
        for l in self.gflat_pred_layer:
            xavier_init(l, gain=0.01)

    def pred2M(self, pitch_z):
        pitch_z = pitch_z / self.M_decay_ratio
        t = pitch_z[:, 0] / 100
        z = pitch_z[:, 1]
        one = torch.ones_like(t)
        zero = torch.zeros_like(t)

        # rot first, then translate
        M = torch.stack([
            one, zero, zero, zero,
            zero, t.cos(), -t.sin(), zero,
            zero, t.sin(), t.cos(), z,
            zero, zero, zero, one], dim=-1).view(t.shape[0], 4, 4)
        return M

    def forward(self, query, key, value,
                top_view_region=None, z_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,
                pt_feats=None,
                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None, seg_branches=None, scores_braches=None,
                query_pos=None,
                point_org=None,
                **kwargs):
        batch_size = query.shape[1]
        xmin = top_view_region[0][0] - self.enlarge_length
        xmax = top_view_region[1][0] + self.enlarge_length
        ymin = top_view_region[2][1] - self.enlarge_length
        ymax = top_view_region[0][1] + self.enlarge_length
        zmin = z_region[0]
        zmax = z_region[1]
        # init_ref_3d = generate_ref_pt(
        #     xmin, ymin, xmax, ymax, init_z,
        #     bev_w, bev_h, query.device)
        # init_ref_3d = init_ref_3d[None, ...].repeat(batch_size, 1, 1, 1)
        pt = point_org.clone()
        init_ref_3d=F.interpolate(pt,size=(bev_w,bev_h),mode='bilinear',align_corners=False).permute(0,2,3,1).contiguous()
        ref_3d_homo = F.pad(init_ref_3d, (0, 1), value=1)
        init_ref_3d_homo = ref_3d_homo.clone()
        init_ref_3d_homo=self.ref_layer(init_ref_3d_homo)
        init_M = torch.eye(4, device=query.device).float()
        M = init_M[None, ...].repeat(batch_size, 1, 1)


        # sizeint = img_feats[0].shape[-2:]
        # pt = F.interpolate(pt, size=sizeint, mode='bilinear')
        # pt[:, 0, ...] = (pt[:, 0, ...] - xmin) / (xmax - xmin)
        # pt[:, 1, ...] = (pt[:, 1, ...] - ymin) / (ymax - ymin)
        # pt[:, 2, ...] = (pt[:, 2, ...] - zmin) / (zmax - zmin)



        project_results = []
        intermediate = []
        outputs_seg = []
        outputs_scores = []
        outputs_classes = []
        outputs_coords = []
        for layer_idx, layer in enumerate(self.layers):

            coords_img = ground2img(
                ref_3d_homo, *img_feats[0].shape[-2:],
                lidar2img, pad_shape)
            if layer_idx > 0:
                project_results.append(coords_img.clone())
            coords_img_key_pos = coords_img.clone()
            ground_coords = coords_img_key_pos[:, :3, ...]
            img_mask = coords_img_key_pos[:, -1, ...]

            ground_coords[:, 0, ...] = (ground_coords[:, 0, ...] - xmin) / (xmax - xmin)
            ground_coords[:, 1, ...] = (ground_coords[:, 1, ...] - ymin) / (ymax - ymin)
            ground_coords[:, 2, ...] = (ground_coords[:, 2, ...] - zmin) / (zmax - zmin)


            key_pos = img_feats[0]
            key_pos = self.position_encoder(key_pos)

            key_pos2 = pt_feats[0]
            ground_coords = inverse_sigmoid(ground_coords)
            key_pos = inverse_sigmoid(key_pos)
            key_pos2 = inverse_sigmoid(key_pos2)
            # ground_coords = inverse_sigmoid(ground_coords)
            key_ground = self.position_encoder_ground(ground_coords)
            key_pos = self.position_encoder(key_pos)
            key_pos2 = self.position_encoder2(key_pos2)
            key_pos = ( sin_embed+key_pos
                       ).flatten(2, 3).permute(2, 0, 1).contiguous()
            key_pos2 = (sin_embed+key_pos2
                       ).flatten(2, 3).permute(2, 0, 1).contiguous()

            query1 = layer(query, key=key, value=value,
                          key_pos=key_pos,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)
            pt_featsnew=pt_feats[0].view( batch_size, self.embed_dims,-1).permute(2, 0, 1).contiguous()
            query2 = self.layers2[layer_idx](query, key=pt_featsnew, value=pt_featsnew,
                          key_pos=key_pos2,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)
            query=query2+query1
            # update M
            if layer_idx < len(self.layers) - 1:
                input_feat = torch.cat([img_feats[0],pt_feats[0], coords_img], dim=1)
                M = M.detach() @ self.pred2M(self.gflat_pred_layer(input_feat).squeeze(-1).squeeze(-1))
                ref_3d_homo = (init_ref_3d_homo.flatten(1, 2) @ M.permute(0, 2, 1)
                               ).view(*ref_3d_homo.shape)
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

            query = query.permute(1, 0, 2)
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                           self.num_anchor_per_query, -1, 3)
            if self.d3type != 'petr':
                reference_points = reference_points.view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 2
                )
                reference_points = inverse_sigmoid(reference_points)
                new_reference_points = torch.stack([
                    reference_points[..., 0] + tmp[..., 0],
                    reference_points[..., 1] + tmp[..., 1],
                ], dim=-1)
                reference_points = new_reference_points.sigmoid()
                outputs_coords.append(torch.cat([
                    reference_points, tmp[..., -1:]
                ], dim=-1))
                reference_points = reference_points.view(
                    bs, self.num_query * self.num_anchor_per_query,
                        self.num_points_per_anchor * 2
                ).detach()
            else:
                reference_points = reference_points.view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 3
                )
                reference_points = inverse_sigmoid(reference_points)
                new_reference_points = torch.stack([
                    reference_points[..., 0] + tmp[..., 0],
                    reference_points[..., 1] + tmp[..., 1],
                ], dim=-1)
                reference_points = new_reference_points.sigmoid()
                outputs_coords.append(torch.cat([
                    reference_points, tmp[..., -1:]
                ], dim=-1))
                reference_points = reference_points.view(
                    bs, self.num_query * self.num_anchor_per_query,
                        self.num_points_per_anchor * 3
                ).detach()

            seg_feat = query.permute(0, 2, 1)
            score = scores_braches[layer_idx](seg_feat)
            outputs_scores.append(score)
            seg_br = seg_branches[layer_idx](seg_feat)
            outputs_seg.append(seg_br.permute(0, 2, 1))

            cls_feat = query.view(bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)

            query = query.permute(1, 0, 2)
        return (outputs_seg, outputs_scores),project_results, outputs_classes, outputs_coords

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DCGTransformerDecoderV4(TransformerLayerSequenceV2):
    def __init__(self,
                 *args, embed_dims=None,
                 post_norm_cfg=dict(type='LN'),
                 enlarge_length=10,
                 M_decay_ratio=10,
                 num_query=None,
                 num_classes=None,
                 d3type=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 **kwargs):
        super(DCGTransformerDecoderV4, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.d3type = d3type
        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query



        self.embed_dims = embed_dims
        self.gflat_pred_layer = nn.Sequential(
            nn.Conv2d(embed_dims*2 + 4, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims // 4, 1),
            nn.BatchNorm2d(embed_dims // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dims // 4, 2, 1))
        self.position_encoder_ground = nn.Sequential(
            nn.Conv2d(3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.position_encoder2 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),

            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.ref_layer = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 4),
            nn.ReLU(True)
        )
        self.padding_layer = nn.ZeroPad2d((0, 0, 0, 1))
        self.M_decay_ratio = M_decay_ratio
        self.enlarge_length = enlarge_length
        self.cls_score = nn.Linear(embed_dims, num_classes)
        self.mask_kernel = nn.Linear(embed_dims, embed_dims)

    def init_weights(self):
        super().init_weights()
        for l in self.gflat_pred_layer:
            xavier_init(l, gain=0.01)

    def pred2M(self, pitch_z):
        pitch_z = pitch_z / self.M_decay_ratio
        t = pitch_z[:, 0] / 100
        z = pitch_z[:, 1]
        one = torch.ones_like(t)
        zero = torch.zeros_like(t)

        # rot first, then translate
        M = torch.stack([
            one, zero, zero, zero,
            zero, t.cos(), -t.sin(), zero,
            zero, t.sin(), t.cos(), z,
            zero, zero, zero, one], dim=-1).view(t.shape[0], 4, 4)
        return M

    def forward(self, query, key, value,
                top_view_region=None, z_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,
                pt_feats=None,
                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None, seg_branches=None, scores_braches=None,
                query_pos=None,
                point_org=None,
                **kwargs):
        # init pts and M to generate pos embed for key/value
        batch_size = query.shape[1]
        xmin = top_view_region[0][0] - self.enlarge_length
        xmax = top_view_region[1][0] + self.enlarge_length
        ymin = top_view_region[2][1] - self.enlarge_length
        ymax = top_view_region[0][1] + self.enlarge_length
        zmin = z_region[0]
        zmax = z_region[1]
        ref_3d_homo = generate_ref_pt(
            xmin, ymin, xmax, ymax, init_z,
            bev_w, bev_h, query.device)
        ref_3d_homo = ref_3d_homo[None, ...].repeat(batch_size, 1, 1, 1)
        ref_3d_homo = F.pad(ref_3d_homo, (0, 1), value=1)
        init_ref_3d_homo = ref_3d_homo.clone()
        M = torch.eye(4, device=query.device).float()
        M = M[None, ...].repeat(batch_size, 1, 1)



        project_results = []
        intermediate = []
        outputs_seg = []
        outputs_scores = []
        outputs_classes = []
        outputs_coords = []
        for layer_idx, layer in enumerate(self.layers):
            coords_img = ground2img(
                ref_3d_homo, *img_feats[0].shape[-2:],
                lidar2img, pad_shape)
            if layer_idx > 0:
                project_results.append(coords_img.clone())
            coords_img_key_pos = coords_img.clone()
            ground_coords = coords_img_key_pos[:, :3, ...]

            ground_coords[:, 0, ...] = (ground_coords[:, 0, ...] - xmin) / (xmax - xmin)
            ground_coords[:, 1, ...] = (ground_coords[:, 1, ...] - ymin) / (ymax - ymin)
            ground_coords[:, 2, ...] = (ground_coords[:, 2, ...] - zmin) / (zmax - zmin)
            ground_coords = inverse_sigmoid(ground_coords)
            key_postmp = self.position_encoder_ground(ground_coords)

            key_w1 = img_feats[0].clone().detach()
            key_w1 = self.position_encoder(key_w1)

            key_w2 = pt_feats[0].clone().detach()

            key_w2 = self.position_encoder2(key_w2)
            key_pos = (sin_embed+key_postmp*key_w1
                       ).flatten(2, 3).permute(2, 0, 1).contiguous()
            key_pos2 = (sin_embed+key_postmp*key_w2
                       ).flatten(2, 3).permute(2, 0, 1).contiguous()

            query1 = layer(query, key=key, value=value,
                          key_pos=key_pos,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)
            pt_featsnew=pt_feats[0].view( batch_size, self.embed_dims,-1).permute(2, 0, 1).contiguous()
            query2 = self.layers2[layer_idx](query, key=pt_featsnew, value=pt_featsnew,
                          key_pos=key_pos2,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)
            query=(query2.permute(1,2,0)*key_w2.squeeze(2)+query1.permute(1,2,0)*key_w1.squeeze(2)).permute(2,0,1).contiguous()
            # update M
            if layer_idx < len(self.layers) - 1:
                input_feat = torch.cat([img_feats[0],pt_feats[0], coords_img], dim=1)
                M = M.detach() @ self.pred2M(self.gflat_pred_layer(input_feat).squeeze(-1).squeeze(-1))
                ref_3d_homo = (init_ref_3d_homo.flatten(1, 2) @ M.permute(0, 2, 1)
                               ).view(*ref_3d_homo.shape)
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

            query = query.permute(1, 0, 2)
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                           self.num_anchor_per_query, -1, 3)
            if self.d3type != 'petr':
                reference_points = reference_points.view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 2
                )
                reference_points = inverse_sigmoid(reference_points)
                new_reference_points = torch.stack([
                    reference_points[..., 0] + tmp[..., 0],
                    reference_points[..., 1] + tmp[..., 1],
                ], dim=-1)
                reference_points = new_reference_points.sigmoid()
                outputs_coords.append(torch.cat([
                    reference_points, tmp[..., -1:]
                ], dim=-1))
                reference_points = reference_points.view(
                    bs, self.num_query * self.num_anchor_per_query,
                        self.num_points_per_anchor * 2
                ).detach()
            else:
                reference_points = reference_points.view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 3
                )
                reference_points = inverse_sigmoid(reference_points)
                new_reference_points = torch.stack([
                    reference_points[..., 0] + tmp[..., 0],
                    reference_points[..., 1] + tmp[..., 1],
                ], dim=-1)
                reference_points = new_reference_points.sigmoid()
                outputs_coords.append(torch.cat([
                    reference_points, tmp[..., -1:]
                ], dim=-1))
                reference_points = reference_points.view(
                    bs, self.num_query * self.num_anchor_per_query,
                        self.num_points_per_anchor * 3
                ).detach()

            seg_feat = query.permute(0, 2, 1)
            score = scores_braches[layer_idx](seg_feat)
            outputs_scores.append(score)
            seg_br = seg_branches[layer_idx](seg_feat)
            outputs_seg.append(seg_br.permute(0, 2, 1))

            cls_feat = query.view(bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)

            query = query.permute(1, 0, 2)
        return (outputs_seg, outputs_scores),project_results, outputs_classes, outputs_coords

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DCGTransformerDecoderV5(TransformerLayerSequenceV2):
    def __init__(self,
                 *args, embed_dims=None,
                 post_norm_cfg=dict(type='LN'),
                 enlarge_length=10,
                 M_decay_ratio=10,
                 num_query=None,
                 num_classes=None,
                 d3type=None,
                 pt_cfg=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 **kwargs):
        super(DCGTransformerDecoderV5, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.d3type = d3type
        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query
        self.pt_cfg=pt_cfg
        if self.pt_cfg.fusion:
            self.d3fuse2 = MODULECONV.build(self.pt_cfg)

        self.embed_dims = embed_dims
        self.gflat_pred_layer = nn.Sequential(
            nn.Conv2d(embed_dims*2 + 4, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(True),
            nn.Conv2d(embed_dims, embed_dims // 4, 1),
            nn.BatchNorm2d(embed_dims // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dims // 4, 2, 1))
        self.position_encoder_ground = nn.Sequential(
            nn.Conv2d(3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.position_encoder2 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),

            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.num_query*self.num_anchor_per_query, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.ref_layer = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 4),
            nn.ReLU(True)
        )
        self.padding_layer = nn.ZeroPad2d((0, 0, 0, 1))
        self.M_decay_ratio = M_decay_ratio
        self.enlarge_length = enlarge_length
        self.cls_score = nn.Linear(embed_dims, num_classes)
        self.mask_kernel = nn.Linear(embed_dims, embed_dims)

    def init_weights(self):
        super().init_weights()
        for l in self.gflat_pred_layer:
            xavier_init(l, gain=0.01)

    def pred2M(self, pitch_z):
        pitch_z = pitch_z / self.M_decay_ratio
        t = pitch_z[:, 0] / 100
        z = pitch_z[:, 1]
        one = torch.ones_like(t)
        zero = torch.zeros_like(t)

        # rot first, then translate
        M = torch.stack([
            one, zero, zero, zero,
            zero, t.cos(), -t.sin(), zero,
            zero, t.sin(), t.cos(), z,
            zero, zero, zero, one], dim=-1).view(t.shape[0], 4, 4)
        return M

    def forward(self, query, key, value,
                top_view_region=None, z_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,
                pt_feats=None,
                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None, seg_branches=None, scores_braches=None,
                query_pos=None,
                point_org=None,
                pt_weight=None,
                img_dep=None,
                **kwargs):
        # init pts and M to generate pos embed for key/value
        batch_size = query.shape[1]
        xmin = top_view_region[0][0] - self.enlarge_length
        xmax = top_view_region[1][0] + self.enlarge_length
        ymin = top_view_region[2][1] - self.enlarge_length
        ymax = top_view_region[0][1] + self.enlarge_length
        zmin = z_region[0]
        zmax = z_region[1]
        ref_3d_homo = generate_ref_pt(
            xmin, ymin, xmax, ymax, init_z,
            bev_w, bev_h, query.device)
        ref_3d_homo = ref_3d_homo[None, ...].repeat(batch_size, 1, 1, 1)
        ref_3d_homo = F.pad(ref_3d_homo, (0, 1), value=1)
        init_ref_3d_homo = ref_3d_homo.clone()
        M = torch.eye(4, device=query.device).float()
        M = M[None, ...].repeat(batch_size, 1, 1)

        # if self.pt_cfg.fusion:
        #     a=img_feats[0].clone().detach()
        #     b=pt_feats[0].clone().detach()
        #     w_feat = self.d3fuse2(a,b)
        # else:
        #     w_feat = img_feats[0]

        project_results = []
        intermediate = []
        outputs_seg = []
        outputs_scores = []
        outputs_classes = []
        outputs_coords = []
        for layer_idx, layer in enumerate(self.layers):
            coords_img = ground2img(
                ref_3d_homo, *img_feats[0].shape[-2:],
                lidar2img, pad_shape)
            if layer_idx > 0:
                project_results.append(coords_img.clone())
            coords_img_key_pos = coords_img.clone()
            ground_coords = coords_img_key_pos[:, :3, ...]

            ground_coords[:, 0, ...] = (ground_coords[:, 0, ...] - xmin) / (xmax - xmin)
            ground_coords[:, 1, ...] = (ground_coords[:, 1, ...] - ymin) / (ymax - ymin)
            ground_coords[:, 2, ...] = (ground_coords[:, 2, ...] - zmin) / (zmax - zmin)
            ground_coords = inverse_sigmoid(ground_coords)
            key_postmp = self.position_encoder_ground(ground_coords)

            # key_w1 = self.position_encoder2(w_feat).flatten(2,3).permute(1,0,2).contiguous().sigmoid()



            key_pos = (sin_embed+key_postmp
                       ).flatten(2, 3).permute(2, 0, 1).contiguous()
            key_pos2 = (sin_embed+key_postmp
                       ).flatten(2, 3).permute(2, 0, 1).contiguous()

            query1 = layer(query, key=key, value=value,
                          key_pos=key_pos,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)
            pt_featsnew=pt_feats[0].view( batch_size, self.embed_dims,-1).permute(2, 0, 1).contiguous()
            query2 = self.layers2[layer_idx](query, key=pt_featsnew, value=pt_featsnew,
                          key_pos=key_pos2,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          **kwargs)
            query=query1*pt_weight+query2*(1-pt_weight)

            # update M
            if layer_idx < len(self.layers) - 1:
                input_feat = torch.cat([img_feats[0],pt_feats[0], coords_img], dim=1)
                M = M.detach() @ self.pred2M(self.gflat_pred_layer(input_feat).squeeze(-1).squeeze(-1))
                ref_3d_homo = (init_ref_3d_homo.flatten(1, 2) @ M.permute(0, 2, 1)
                               ).view(*ref_3d_homo.shape)
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

            query = query.permute(1, 0, 2)
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                           self.num_anchor_per_query, -1, 3)
            if self.d3type != 'petr':
                reference_points = reference_points.view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 2
                )
                reference_points = inverse_sigmoid(reference_points)
                new_reference_points = torch.stack([
                    reference_points[..., 0] + tmp[..., 0],
                    reference_points[..., 1] + tmp[..., 1],
                ], dim=-1)
                reference_points = new_reference_points.sigmoid()
                outputs_coords.append(torch.cat([
                    reference_points, tmp[..., -1:]
                ], dim=-1))
                reference_points = reference_points.view(
                    bs, self.num_query * self.num_anchor_per_query,
                        self.num_points_per_anchor * 2
                ).detach()
            else:
                reference_points = reference_points.view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 3
                )
                reference_points = inverse_sigmoid(reference_points)
                new_reference_points = torch.stack([
                    reference_points[..., 0] + tmp[..., 0],
                    reference_points[..., 1] + tmp[..., 1],
                ], dim=-1)
                reference_points = new_reference_points.sigmoid()
                reference_points=torch.cat([
                    reference_points, tmp[..., -1:]
                ], dim=-1)
                outputs_coords.append(reference_points)
                reference_points = reference_points.view(
                    bs, self.num_query * self.num_anchor_per_query,
                        self.num_points_per_anchor * 3
                ).detach()

            seg_feat = query.permute(0, 2, 1)
            score = scores_braches[layer_idx](seg_feat)
            outputs_scores.append(score)
            seg_br = seg_branches[layer_idx](seg_feat)
            outputs_seg.append(seg_br.permute(0, 2, 1))

            cls_feat = query.view(bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)

            query = query.permute(1, 0, 2)
        return (outputs_seg, outputs_scores),project_results, outputs_classes, outputs_coords

@TRANSFORMER.register_module()
class DCGTransformer(BaseModule):
    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(DCGTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @property
    def with_encoder(self):
        return hasattr(self, 'encoder') and self.encoder

    def forward(self, x, query,
                query_embed,
                reference_points=None,
                reg_branches=None, cls_branches=None, seg_branches=None, scores_braches=None,
                spatial_shapes=None,
                level_start_index=None,

                **kwargs):

        memory = x
        # encoder

        query_embed = query_embed.permute(1, 0, 2)

        target = query.permute(1, 0, 2)

        # out_dec: [num_layers, num_query, bs, dim]
        querys,project_results, outputs_classes, outputs_coords = \
            self.decoder(
                query=target,
                key=memory,
                value=memory,

                query_pos=query_embed,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                seg_branches=seg_branches,
                scores_braches=scores_braches,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
        return querys,project_results, outputs_classes, outputs_coords
