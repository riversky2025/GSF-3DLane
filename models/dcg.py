import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from mmdet3d.models import build_backbone, build_neck
from .dcg_head import DCGHead
from mmcv.utils import Config
from .ms2one import build_ms2one
from .utils import deepFeatureExtractor_EfficientNet
from models.builder import MODULECONV, LOSSES
from einops import rearrange
from mmdet.models.builder import BACKBONES
from models.grid_mask import GridMask
from .effnet import efficientnet_feature


def base_block(in_filters, out_filters, normalization=False, kernel_size=3, stride=2, padding=1):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))

    return layers


def apply_camera_pose_adjustment(x, y, z, camera_height=1.7860000133514404, pitch_angle=0.04325083977888603):
    # Apply camera pose adjustment based on height and pitch angle
    x_adj = x
    y_adj = y + camera_height * np.sin(np.radians(pitch_angle))
    z_adj = z + camera_height * np.cos(np.radians(pitch_angle))
    return x_adj, y_adj, z_adj

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
def ransac_quadratic_surface_and_filter(point_cloud,threshold=0.01, num_iterations=5 ):
    batch_size, num_points, _ = point_cloud.shape
    device = point_cloud.device
    best_coeffs = torch.zeros(batch_size, 6, device=device)
    best_inliers = torch.zeros(batch_size, num_points, dtype=torch.bool, device=device)

    for batch_idx in range(batch_size):
        best_inlier_count = 0
        best_coeff = None

        for _ in range(num_iterations):
            # 随机选择六个点
            sample_indices = np.random.choice(num_points, 6, replace=False)
            sample_points = point_cloud[batch_idx, sample_indices]
            x = sample_points[:, 0]
            y = sample_points[:, 1]
            z = sample_points[:, 2]

            A = torch.stack([x**2, y**2, x*y, x, y, torch.ones_like(x)], dim=-1)
            try:
                A_pinv = torch.linalg.pinv(A)
                coeff = A_pinv @ z

                # 计算所有点到曲面的距离
                X = point_cloud[batch_idx, :, 0]
                Y = point_cloud[batch_idx, :, 1]
                Z = point_cloud[batch_idx, :, 2]
                Z_pred = coeff[0] * X**2 + coeff[1] * Y**2 + coeff[2] * X * Y + coeff[3] * X + coeff[4] * Y + coeff[5]
                distances = torch.abs(Z - Z_pred)

                # 找到内点
                inliers = distances < threshold[batch_idx]
                inlier_count = torch.sum(inliers)

                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_coeff = coeff
                    best_inliers[batch_idx] = inliers

            except RuntimeError as e:
                print(f"Error in pinverse computation: {e}")

        if best_coeff is not None:
            best_coeffs[batch_idx] = best_coeff

    # 生成与输入点云形状相同的点云
    X = point_cloud[..., 0]
    Y = point_cloud[..., 1]
    Z = point_cloud[..., 2]
    filtered_point_cloud = torch.zeros_like(point_cloud)

    for batch_idx in range(batch_size):
        a, b, c, d, e, f = best_coeffs[batch_idx]
        Z_pred = a * X[batch_idx]**2 + b * Y[batch_idx]**2 + c * X[batch_idx] * Y[batch_idx] + d * X[batch_idx] + e * Y[batch_idx] + f
        distances = torch.abs(Z[batch_idx] - Z_pred)
        inliers = distances < threshold[batch_idx]

        # 更新结果点云
        filtered_point_cloud[batch_idx][inliers] = point_cloud[batch_idx][inliers]

    return filtered_point_cloud
@torch.no_grad()
def fit_and_filter_surface_torch(point_cloud, threshold=0.1):
    """
    拟合二次曲面 z = ax^2 + by^2 + cxy + dx + ey + f，并过滤点云中不属于拟合曲面的点。

    参数:
    point_cloud (torch.Tensor): 点云数据，大小为 (batch_size, num_points, 3)
    threshold (float): 距离阈值，判断点是否属于曲面

    返回:
    torch.Tensor: 过滤后的点云，大小为 (batch_size, num_points, 3)
    """
    batch_size, num_points, _ = point_cloud.shape

    # 初始化结果
    coeffs = torch.zeros(batch_size, 6, device=point_cloud.device)
    distances = torch.zeros(batch_size, num_points, device=point_cloud.device)

    # 计算矩阵 A 和向量 b
    x = point_cloud[..., 0]
    y = point_cloud[..., 1]
    z = point_cloud[..., 2]
    A = torch.stack([x**2, y**2, x*y, x, y, torch.ones_like(x)], dim=-1)  # (batch_size, num_points, 6)
    b = z.unsqueeze(-1)  # (batch_size, num_points, 1)

    # 使用伪逆矩阵拟合曲面
    for i in range(batch_size):
        A_i = A[i]  # (num_points, 6)
        b_i = b[i]  # (num_points, 1)

        try:
            # 计算伪逆矩阵
            A_pinv = torch.linalg.pinv(A_i)
            coeffs[i] = (A_pinv @ b_i).squeeze()  # 从 (6, 1) 转换为 (6,)
        except RuntimeError as e:
            print(f"Error in pinverse computation: {e}")
            coeffs[i] = torch.zeros(6, device=point_cloud.device)

    # 提取拟合曲面的系数
    a, b, c, d, e, f = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3], coeffs[:, 4], coeffs[:, 5]

    # 计算拟合曲面的预测值
    z_pred = (a.unsqueeze(1) * x**2 +
              b.unsqueeze(1) * y**2 +
              c.unsqueeze(1) * x * y +
              d.unsqueeze(1) * x +
              e.unsqueeze(1) * y +
              f.unsqueeze(1))  # (batch_size, num_points)

    # 计算点到拟合曲面的距离
    distances = torch.abs(z - z_pred)

    # 找到内点
    mask = distances <= threshold.view(4,1)

    # 使用 mask 过滤点云
    filtered_point_cloud = torch.where(mask.unsqueeze(-1), point_cloud, torch.tensor(0.0, device=point_cloud.device))

    return filtered_point_cloud
@torch.no_grad()
def create_point_cloud_from_rgb_depth_torch(imgDep, K, height, patch,weight):
    """
    从深度图像创建点云，并根据Y坐标进行过滤。

    参数:
        imgDep (torch.Tensor): 深度图像，形状为 (batch, 1, w, h) 的 torch 张量。
        K (torch.Tensor): 相机内参矩阵。
        height (float): 相机姿态调整的高度。
        patch (torch.Tensor): 相机姿态调整的补丁。

    返回:
        points (torch.Tensor): 点云张量。
    """
    # 检查输入图像是否为torch张量
    if not isinstance(imgDep, torch.Tensor):
        raise TypeError("输入图像必须是torch张量。")

    # 检查输入图像的形状是否正确
    if imgDep.ndim != 4 :
        raise ValueError("输入张量必须具有形状 (batch, 3, w, h)。")

    # 提取相机内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 提取深度通道
    depth = imgDep[:, 0, :, :]

    # 计算每个点的XYZ值
    batch_size, w, h = depth.shape
    u, v = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device))
    u = u.t().float()
    v = v.t().float()
    d = depth.float()
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    # 应用相机姿态调整
    x, y, z = apply_camera_pose_adjustment(x, z, -y, height, patch)

    # 将XYZ值组合成一个张量
    points = torch.stack([x, y, z], dim=-1)  # 形状 (w, h, 3)
    batch_size, width, height, _ = points.shape

    # 使用 reshape 或 view 来调整张量的形状
    points = points.reshape(batch_size, width * height, 3)
    # num_slices = (weight[:, 0] * 10).long()
    # quantile = weight[:, 1]
    threshold = (weight[:, 0]*10)
    num_slices = torch.full(weight[:, 0].shape, 5)
    quantile = torch.full(weight[:, 0].shape, 0.5)
    # 50,0.6
    points = filter_points_by_y_slice_quantile_torch(points, num_slices=num_slices, quantile=quantile)
    # points1 = points
    # 2
    # points = fit_and_filter_surface_torch(points, threshold=threshold)
    # points = ransac_quadratic_surface_and_filter(points, threshold=threshold)
    return points



# overall network
class DCG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        _dim_ = args.dcg_cfg.fpn_dim
        num_query = args.dcg_cfg.num_query
        num_group = args.dcg_cfg.num_group
        sparse_num_group = args.dcg_cfg.sparse_num_group
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if args.dcg_cfg.encoder.type == 'ResNet':
            self.encoder = build_backbone(args.dcg_cfg.encoder)
        else:
            self.encoder = MODULECONV.build(args.dcg_cfg.encoder)

        self.filterHead = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            *base_block(_dim_, 2),
            nn.Sigmoid()
        )
        self.neck = build_neck(args.dcg_cfg.neck)
        self.ms2one = build_ms2one(args.dcg_cfg.ms2one)
        self.encoder.init_weights()

        self.fusion_cfg = args.fusion_cfg
        self.dep_grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if args.dcg_cfg.encoder.type == 'ResNet':
            self.dep_encoder = build_backbone(args.dcg_cfg.encoder)
        else:
            self.dep_encoder = MODULECONV.build(args.dcg_cfg.encoder)
        self.dep_neck = build_neck(args.dcg_cfg.neck)
        self.dep_ms2one = build_ms2one(args.dcg_cfg.ms2one)
        self.dep_encoder.init_weights()
        self.pt_encoder = MODULECONV.build(args.dcg_cfg.pts_backbone)
        self.pt_neck = build_neck(args.dcg_cfg.neck)
        self.pt_ms2one = build_ms2one(args.dcg_cfg.ms2one)
        self.pt_encoder.init_weights()

        if 'openlane' in args.dataset_name:
            self.K = np.array([[1000., 0., 960.],
                          [0., 1000., 640.],
                          [0., 0., 1.]])
            self.cam_height, self.cam_pitch = 1.55, np.array(3)
        elif 'apollo' in args.dataset_name:
            self.K = np.array([[2015., 0., 960.],
                          [0., 2015., 540.],
                          [0., 0., 1.]])
            self.cam_height, self.cam_pitch =   1.55, np.array(3)
        elif 'once' in args.dataset_name:
            self.cx, self.cy, self.fx, self.fy = 1000.0, 1000.0, 960.0, 640.0
        else:
            assert False

        # build 2d query-based instance seg
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
        image2 = self.grid_mask(image)
        neck_out1 = self.encoder(image2)
        _,_,w,h=image2.shape
        neck_out = self.neck(neck_out1)
        neck_out = self.ms2one(neck_out)

        # if self.fusion_cfg.dep_cfg.fusion:
        dep_image = extra_dict['dep_image']

        dep_out1 = self.dep_grid_mask(dep_image)
        dep_out1 = self.dep_encoder(dep_out1)
        dep_out1 = self.dep_neck(dep_out1)
        dep_out1 = self.dep_ms2one(dep_out1)

        # fu1=torch.cat((dep_image),dim=1)
        filterWeight = self.filterHead(dep_out1)
        # filterWeight = torch.clamp(filterWeight, max=0.65)

        filterWeight = filterWeight.squeeze(3).squeeze(2)

        batch, _, w, h = dep_image.shape
        pt = create_point_cloud_from_rgb_depth_torch(dep_image, self.K, self.cam_height, self.cam_pitch,filterWeight)

        extra_dict['point'] = pt

        # pt = extra_dict['point']
        # pt = filterdepty(pt, filterWeight,w,h)
        # extra_dict['point'] = pt
        # if self.fusion_cfg.pt_cfg.fusion:
        point = extra_dict['point']
        point = rearrange(point, "b (w h) c -> b c w h", w=w, h=h)
        point_out = self.pt_encoder(point)
        point = self.pt_neck(point_out)
        point = self.pt_ms2one(point)
        output = self.head(
            dict(
                x=neck_out,
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
                pt_weight=filterWeight[:,-1].unsqueeze(1)
            ),
            is_training=is_training,
        )
        return output
