import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from mmdet3d.models import build_backbone, build_neck
from .dcg_head import DCGHead
from .ms2one import build_ms2one
from mmcv.utils import Config
from models.builder import MODULECONV, LOSSES
from einops import rearrange
from mmdet.models.builder import BACKBONES
from models.grid_mask import GridMask

class PointFeatureNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)

def base_block(in_filters, out_filters, normalization=False, kernel_size=3, stride=2, padding=1):
    layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class NarrowResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 32
        
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.relu1_1 = nn.ReLU(inplace=True)
        
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.relu1_2 = nn.ReLU(inplace=True)
        
        self.conv1_3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(32)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(NarrowBasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(NarrowBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.relu1_3(self.bn1_3(self.conv1_3(x)))
        
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return (c2, c3, c4)

class NarrowBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

@torch.no_grad()
def filter_points_by_y_slice_quantile_torch(points, num_slices, quantile):
    batch_size, num_points, _ = points.shape
    processed_points = torch.clone(points)
    valid_mask = torch.zeros(batch_size, num_points, dtype=torch.bool, device=points.device)

    def process_slices(pts, y_slices, quantile, valid_mask):
        y_values = pts[:, 1]
        z_values = pts[:, 2]

        for j in range(len(y_slices) - 1):
            y_low, y_high = y_slices[j], y_slices[j + 1]
            slice_mask = (y_values >= y_low) & (y_values < y_high)
            slice_points = pts[slice_mask]

            if slice_points.size(0) == 0:
                continue

            heights = slice_points[:, 2]
            q1_height = torch.quantile(heights, quantile.item())
            valid_slice_mask = slice_mask & (z_values <= q1_height)
            valid_mask |= valid_slice_mask

        return valid_mask

    for i in range(batch_size):
        pts = points[i]
        y_values = pts[:, 1]

        y_min, y_max = torch.min(y_values), torch.max(y_values)

        y_slices1 = torch.linspace(y_min.item(), y_max.item(), num_slices[i].item() + 1, device=points.device)
        y_slices2 = torch.linspace(
            y_min.item() + (y_max.item() - y_min.item()) / num_slices[i].item() / 3,
            y_max.item() + (y_max.item() - y_min.item()) / num_slices[i].item() / 3,
            num_slices[i].item() + 1, device=points.device)
        y_slices3 = torch.linspace(
            y_min.item() + (y_max.item() - y_min.item()) * 2 / num_slices[i].item() / 3,
            y_max.item() + (y_max.item() - y_min.item()) * 2 / num_slices[i].item() / 3,
            num_slices[i].item() + 1, device=points.device)

        valid_mask1 = process_slices(pts, y_slices1, quantile[i], torch.zeros(num_points, dtype=torch.bool, device=points.device))
        valid_mask2 = process_slices(pts, y_slices2, quantile[i], torch.zeros(num_points, dtype=torch.bool, device=points.device))
        valid_mask3 = process_slices(pts, y_slices3, quantile[i], torch.zeros(num_points, dtype=torch.bool, device=points.device))

        valid_mask[i] = valid_mask1 | valid_mask2 | valid_mask3

    processed_points[~valid_mask] = 0.0
    return processed_points

@torch.no_grad()
def create_point_cloud_from_rgb_depth_torch(imgDep, K, cam_pitch, cam_height, weight):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depth = imgDep[:, 0, :, :]

    batch_size, w, h = depth.shape
    u, v = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device))
    u = u.t().float()
    v = v.t().float()
    d = depth.float()

    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    pitch = cam_pitch
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    x_rot = x_cam
    y_rot = cos_p * y_cam - sin_p * z_cam
    z_rot = sin_p * y_cam + cos_p * z_cam

    x = x_rot
    y = z_rot
    z = -y_rot + cam_height

    points = torch.stack([x, y, z], dim=-1)
    batch_size, width, height, _ = points.shape

    points = points.reshape(batch_size, width * height, 3)

    threshold = (weight[:, 0]*10)
    num_slices = torch.full(weight[:, 0].shape, 5)
    quantile = torch.full(weight[:, 0].shape, 0.5)
    points = filter_points_by_y_slice_quantile_torch(points, num_slices=num_slices, quantile=quantile)
    return points

class DCG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        
        _dim_ = args.dcg_cfg.fpn_dim
        num_query = args.dcg_cfg.num_query
        num_group = args.dcg_cfg.num_group
        
        if args.dcg_cfg.encoder.type == 'ResNet':
            self.encoder = build_backbone(args.dcg_cfg.encoder)
        else:
            self.encoder = MODULECONV.build(args.dcg_cfg.encoder)
            
        self.enc_proj = nn.Sequential(
           nn.Conv2d(512, _dim_ * (2 ** 2), kernel_size=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(_dim_),
            nn.ReLU(True),
            
            nn.Conv2d(_dim_, _dim_ * (2 ** 2), kernel_size=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(_dim_),
            nn.ReLU(True)
        )
        
        self.filterHead = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            *base_block(_dim_, 2),
            nn.Sigmoid()
        )
        
        self.dep_conv_dim = nn.Sequential(
            nn.Conv2d(256, _dim_, kernel_size=1, bias=False),
            nn.BatchNorm2d(_dim_),
            nn.ReLU(True),
        )
        
        self.pt_conv_dim = nn.Sequential(
            nn.Conv2d(512, _dim_, kernel_size=1, bias=False),
            nn.BatchNorm2d(_dim_),
            nn.ReLU(True),
        )
        
        self.encoder.init_weights()
        self.fusion_cfg = args.fusion_cfg

        self.dep_encoder = NarrowResNet18()
        self.dep_encoder.init_weights()
        
        self.pt_feature = PointFeatureNet(in_channels=3, out_channels=32)
        self.pt_encoder = MODULECONV.build(args.dcg_cfg.pts_backbone)
        self.pt_encoder.init_weights()

        if 'openlane' in args.dataset_name:
            self.register_buffer('K', torch.tensor([[2081.5212, 0.0, 934.7111],
                                                    [0.0, 2081.5212, 646.3389],
                                                    [0., 0., 1.]], dtype=torch.float32))
            
            vc_extrinsics = np.array([
                [-0.00212216,  0.01069750,  0.99994053,  1.54410395],
                [-0.99993783, -0.01096862, -0.00200481, -0.02377403],
                [ 0.01094652, -0.99988262,  0.01072011,  2.11573979],
                [ 0.0,         0.0,         0.0,         1.0       ]
            ])
            self.cam_height = float(vc_extrinsics[2, 3])
            pitch_rad = float(np.arcsin(vc_extrinsics[2, 2]))
            self.register_buffer('cam_pitch', torch.tensor(pitch_rad, dtype=torch.float32))
            
            self.max_depth = 104.0 
            
        elif 'apollo' in args.dataset_name:
            self.register_buffer('K', torch.tensor([[2015., 0., 960.],
                                                    [0., 2015., 540.],
                                                    [0., 0., 1.]], dtype=torch.float32))
            self.cam_height = 1.786
            self.register_buffer('cam_pitch', torch.tensor(0.043250839, dtype=torch.float32))
            self.max_depth = 104.0 
        else:
            assert False, "Unsupported dataset"

        self.head = DCGHead(
            args=args,
            dim=_dim_,
            num_group=num_group,
            num_convs=4,
            in_channels=_dim_,
            kernel_dim=_dim_,
            position_range=args.position_range,
            top_view_region=args.top_view_region,
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=_dim_ // 2, normalize=True),
            num_query=num_query,
            pred_dim=self.num_y_steps,
            num_classes=args.num_category,
            embed_dims=_dim_,
            transformer=args.transformer,
            sparse_ins_decoder=args.sparse_ins_decoder,
            **args.dcg_cfg.get('head', {}),
            trans_params=args.dcg_cfg.get('trans_params', {})
        )

    def forward(self, image, _M_inv=None, is_training=True, extra_dict=None):
        enc_out = self.encoder(image)
        _, _, w, h = image.shape
        
        if isinstance(enc_out, (list, tuple)):
            enc_out = enc_out[-1]
            
        enc_out = self.enc_proj(enc_out)
        
        dep_image_raw = extra_dict['dep_image']
        
        dep_image_norm = torch.clamp(dep_image_raw / self.max_depth, min=0.0, max=1.0)
        
        dep_enc_out = self.dep_encoder(dep_image_norm)
        dep_out1 = self.dep_conv_dim(dep_enc_out[-1])

        filterWeight = self.filterHead(dep_out1)
        filterWeight = filterWeight.squeeze(3).squeeze(2)

        batch, _, w, h = dep_image_raw.shape
        with torch.no_grad():
            pt = create_point_cloud_from_rgb_depth_torch(
                dep_image_raw.detach(), 
                self.K, 
                self.cam_pitch, 
                self.cam_height, 
                filterWeight
            )

        extra_dict['point'] = pt
        point = pt
        point = rearrange(point, "b (w h) c -> b c w h", w=w, h=h)
        point = self.pt_feature(point)
        point_enc_out = self.pt_encoder(point)
        point = self.pt_conv_dim(point_enc_out[-1])
        point = F.interpolate(point, size=enc_out.shape[2:], mode='nearest')
        
        output = self.head(
            dict(
                x=enc_out,
                dep_x=dep_out1 if 'dep_image' in locals() else None,
                pt_x=point if 'point' in locals() else None,
                lane_idx=extra_dict['seg_idx_label'],
                seg=extra_dict['seg_label'],
                lidar2img=extra_dict['lidar2img'],
                pad_shape=extra_dict['pad_shape'],
                ground_lanes=extra_dict['ground_lanes'] if is_training else None,
                ground_lanes_dense=extra_dict['ground_lanes_dense'] if is_training else None,
                image=image,
                point_org=extra_dict['point'],
                pt_weight=filterWeight[:, -1].unsqueeze(1)
            ),
            is_training=is_training,
        )
        return output
