import numpy as np

from scipy.spatial import cKDTree
class Calibration:
    def __init__(self, K, cam_height, pitch):
        # 设置相机内参
        # self.f_u = K[0, 0]
        # self.f_v = K[1, 0]
        # self.c_u = K[0, 2]
        # self.c_v = K[1, 2]
        self.c_u = K[0, 2]
        self.c_v = K[1, 2]
        self.f_u = K[0, 0]
        self.f_v = K[1, 1]
        # 相机高度
        self.cam_height = cam_height

        # 俯仰角
        self.pitch = pitch

        # 计算旋转矩阵 R_pitch
        pitch_rad = np.deg2rad(pitch)
        self.R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])

        # 设置外参中的平移分量
        self.b_x = 0  # 相机沿 x 轴没有平移
        self.b_y = 0  # 相机沿 y 轴没有平移

    def update_point_cloud_with_surface(self,original_points, surface_points, threshold=0.1):
        """
        在原始点云基础上，更新指定位置上的点

        参数:
        original_points (np.ndarray): 原始点云数据，大小为 (num_points, 3)
        surface_points (np.ndarray): 曲面上的点，大小为 (num_surface_points, 3)
        threshold (float): 最近邻点的距离阈值，用于确定匹配

        返回:
        np.ndarray: 更新后的点云，大小为 (num_points, 3)
        """
        # 构建原始点云的 KDTree
        tree = cKDTree(original_points)

        # 查找每个曲面点的最近邻
        distances, indices = tree.query(surface_points, k=1)

        # 仅保留距离小于阈值的点
        mask = distances < threshold

        # 替换原始点云中对应的点
        original_points[indices[mask]] = surface_points[mask]

        return original_points

    def generate_points_on_surface(self,coeffs, num_points=5000, x_range=(-10,10), y_range=(3,103)):
        """
        在拟合的二次曲面上生成均匀分布的点

        参数:
        coeffs (np.ndarray): 二次曲面的系数 [a, b, c, d, e, f]
        num_points (int): 要生成的点数
        x_range (tuple): x 坐标的范围 (x_min, x_max)
        y_range (tuple): y 坐标的范围 (y_min, y_max)

        返回:
        np.ndarray: 曲面上的点，大小为 (num_points, 3)
        """
        a, b, c, d, e, f = coeffs
        x_min, x_max = x_range
        y_min, y_max = y_range

        # 随机生成 x 和 y 坐标
        x = np.random.uniform(x_min, x_max, num_points)
        y = np.random.uniform(y_min, y_max, num_points)

        # 根据二次曲面方程计算 z 坐标
        z = a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

        return np.column_stack((x, y, z))
    def compute_distances(self,points, coeffs):
        """
        计算点到拟合二次曲面的距离

        参数:
        points (np.ndarray): 点云数据，大小为 (num_points, 3)
        coeffs (np.ndarray): 二次曲面的参数 [a, b, c, d, e, f]

        返回:
        np.ndarray: 每个点到拟合曲面的距离
        """
        A = np.c_[
            points[:, 0] ** 2, points[:, 1] ** 2, points[:, 0] * points[:, 1], points[:, 0], points[:, 1], np.ones(
                points.shape[0])]
        z_fit = np.dot(A, coeffs)
        distances = np.abs(points[:, 2] - z_fit)
        return distances
    def filter_surface_points(self,point_cloud, coeffs, threshold=0.3):
        """
        过滤点云中不属于拟合曲面的点

        参数:
        point_cloud (np.ndarray): 点云数据，大小为 (point_num, 3)
        coeffs (np.ndarray): 二次曲面的参数 [a, b, c, d, e, f]
        threshold (float): 距离阈值，判断点是否属于曲面

        返回:
        np.ndarray: 过滤后的点云，大小为 (num_filtered_points, 3)
        """
        # 计算点到拟合曲面的距离
        distances = self.compute_distances(point_cloud, coeffs)

        # 找到内点
        mask = distances < threshold
        # 将不符合条件的点填充为0
        filtered_point_cloud = np.where(mask[:, np.newaxis], point_cloud, 0)

        return filtered_point_cloud
    def fit_quadratic_surface(self,points):
        """
        拟合二次曲面 z = ax^2 + by^2 + cxy + dx + ey + f

        参数:
        points (np.ndarray): 点云数据，大小为 (num_points, 3)

        返回:
        np.ndarray: 拟合曲面的参数 [a, b, c, d, e, f]
        """
        A = np.c_[
            points[:, 0] ** 2, points[:, 1] ** 2, points[:, 0] * points[:, 1], points[:, 0], points[:, 1], np.ones(
                points.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
        return C
    def cart2hom(self, pts_3d):
        """
        将笛卡尔坐标转换为齐次坐标。

        参数:
        pts_3d -- nx3 笛卡尔坐标

        返回:
        pts_3d_hom -- nx4 齐次坐标
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    def project_image_to_rect(self, uv_depth):
        '''
        将图像坐标 (u, v) 和深度值 (depth) 投影到三维矩形坐标系。

        参数:
        uv_depth -- 输入为nx3的数组，前两列是uv坐标，第三列是深度值

        返回:
        pts_3d_rect -- nx3数组，表示三维矩形坐标系中的点
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect
    def project_rect_to_ref(self, pts_3d_rect):
        '''
        将三维矩形坐标系中的点投影到参考坐标系。

        参数:
        pts_3d_rect -- 输入和输出均为nx3的数组

        返回:
        pts_3d_ref -- 参考坐标系中的点
        '''
        return np.transpose(np.dot(np.linalg.inv(self.R_pitch), np.transpose(pts_3d_rect)))
    def project_ref_to_velo(self, pts_3d_ref):
        '''
        将参考坐标系中的点投影到激光雷达坐标系。

        参数:
        pts_3d_ref -- 参考坐标系中的点

        返回:
        pts_3d_velo -- 激光雷达坐标系中的点
        '''
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -self.cam_height],
            [0, 0, 0, 1]
        ])
        return np.dot(pts_3d_ref, np.transpose(T))[:, :3]
    def project_rect_to_velo(self, pts_3d_rect):
        '''
        将三维矩形坐标系中的点投影到激光雷达坐标系。

        参数:
        pts_3d_rect -- 三维矩形坐标系中的点

        返回:
        pts_3d_velo -- 激光雷达坐标系中的点
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)
    def project_image_to_velo(self, uv_depth):
        '''
        将图像坐标投影到激光雷达坐标系。

        参数:
        uv_depth -- 图像坐标和深度值

        返回:
        pts_3d_velo -- 激光雷达坐标系中的点
        '''
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)
if __name__ == '__main__':


    # 示例初始化
    K = np.array([[2015., 0., 960.],
                  [0., 2015., 540.],
                  [0., 0., 1.]])

    cam_height = 1.55
    pitch = 3

    calib = Calibration(K, cam_height, pitch)
    print(calib)
