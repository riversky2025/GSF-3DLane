import numpy as np
from mmcv.utils import Config

_base_ = [
    '../_base_/base_res101_bs16xep100.py',
    '../_base_/optimizer.py'
]

mod = 'release_iccv/latr_1000_baseline_rio'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

dataset = '1000'
dataset_dir = '/data02/OpenLane/images/'
data_dir = '/data02/OpenLane/lane3d_1000/'

batch_size = 6
nworkers = 10
num_category = 21
pos_threshold = 0.3

clip_grad_norm = 20

top_view_region = np.array([
    [-10, 103], [10, 103], [-10, 3], [10, 3]])
enlarge_length = 20
position_range = [
    top_view_region[0][0] - enlarge_length,
    top_view_region[2][1] - enlarge_length,
    -5,
    top_view_region[1][0] + enlarge_length,
    top_view_region[0][1] + enlarge_length,
    5.]
anchor_y_steps = np.linspace(3, 103, 20)
num_y_steps = len(anchor_y_steps)

# extra aug
photo_aug = dict(
    brightness_delta=32 // 2,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=9)

_dim_ = 256
num_query = 40
num_pt_per_line = 20
em_bed_type='dc'
dcg_cfg = dict(
    fpn_dim=_dim_,
    num_query=num_query,
    num_group=1,
    sparse_num_group=4,
    encoder=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, False, False),
    ),
    neck=dict(
        type='FPN',
       in_channels=[128, 256, 512],
         # in_channels=[512, 768, 1024],
        # in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=3,
        out_channels=[128, 256, 512],
        layer_nums=[5, 5, 5],
        layer_strides=[8, 2, 1],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256, 512],
        out_channels=[_dim_, _dim_, _dim_],
        upsample_strides=[1, 2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    ms2one=dict(
        type='DilateNaive',
        inc=_dim_, outc=_dim_, num_scales=3,
        dilations=(1, 2, 5)),
    head=dict(
        xs_loss_weight=2.0,
        zs_loss_weight=10.0,
        vis_loss_weight=1.0,
        cls_loss_weight=10,
        project_loss_weight=1.0,
        pt_as_query=True,
        num_pt_per_line=num_pt_per_line,
    ),
    trans_params=dict(init_z=0, bev_h=150, bev_w=70),
)
# DualAdaptiveNeuralBlock CnnFusionRs ChannelAttentionRs CondensedAttentionNeuralBlockAtt,BevFusionRs
fusion_cfg = dict(
    d3_dim=num_query * num_pt_per_line,
    dep_cfg=dict(
        type='SCC',
        embed_dim=_dim_,
        fusion=True
    ),
    pt_cfg=dict(
        type='SCC',
        embed_dim=_dim_,
        fusion=True
    ),
    iam_cfg=dict(
        type='IAMBlock',
        num_mask=num_query,
        num_group=dcg_cfg['num_group'],
        sparse_num_group=dcg_cfg['sparse_num_group'],
        embed_dims=_dim_
    )
)

sparse_ins_decoder=Config(
    dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            num_query=dcg_cfg['num_query'],
            num_group=dcg_cfg['num_group'],
            sparse_num_group=dcg_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1.,
            ce_weight=2.0,
            mask_weight=5.0,
            dice_weight=2.0,
            objectness_weight=1.0,
        ),
        sparse_decoder_weight=5.0,
))
seg_cfg=dict(
    in_dim=_dim_,
    decoder=dict(
            num_query=dcg_cfg['num_query'],
            num_group=dcg_cfg['num_group'],
            sparse_num_group=dcg_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1.,
            ce_weight=2.0,
            mask_weight=5.0,
            dice_weight=2.0,
            objectness_weight=1.0,
        ),
)
transformer=dict(
    type='DCGTransformer',
    decoder=dict(
        type='DCGTransformerDecoderV5',
        d3type=em_bed_type,
        embed_dims=_dim_,
        num_layers=4,
        enlarge_length=enlarge_length,
        M_decay_ratio=1,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        num_classes=num_category,
            pt_cfg=dict(
        type='SCC',
        embed_dim=_dim_,
        fusion=True
    ),
        transformerlayers=dict(
            type='DCGDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    dropout=0.1),
                dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    d3type=em_bed_type,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    num_query=num_query,
                    num_anchor_per_query=num_pt_per_line,
                    anchor_y_steps=anchor_y_steps,
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))



resize_h = 720
resize_w = 960

nepochs = 300
eval_freq = 1
optimizer_cfg = dict(
    type='AdamW',
    lr=2e-5,
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)