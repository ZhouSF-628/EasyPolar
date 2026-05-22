import cv2
import numpy as np
import torch
import torch.nn.functional as F
from os.path import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
 
 
class AoPLoss(torch.nn.Module):
    def __init__(self):
        super(AoPLoss, self).__init__()
 
    def forward(self, pred_aop, gt_aop):
        abs_err = torch.abs(pred_aop - gt_aop)
        angular_err = torch.min(abs_err, torch.abs(abs_err - torch.pi))
        
        return torch.mean(angular_err)
 
 
def read_file(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        file = cv2.imread(file_name, -1)
        if file.dtype == np.uint16:
            file = file / 65535.0
        elif file.dtype == np.uint8:
            file = file / 255.0
        
        return file.astype(np.float32)
    
    elif ext == '.bin' or ext == '.raw' or ext == '.npy':
        return np.load(file_name).astype(np.float32)
 
    else:
        raise ValueError(f'Unsupported file format: {ext}')
 
def read_pfm(pfm_path):
    with open(pfm_path, 'rb') as file:
        header = file.readline().decode('utf-8').strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError("Invalid PFM header: " + header)
 
        dim_match = file.readline().decode('utf-8').strip()
        width, height = map(int, dim_match.split(' '))
        scale = float(file.readline().decode('utf-8').strip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'
 
        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # 翻转数据
        return data.copy()  # 确保返回的数组是连续的
    
 
def load_camera_info(camera_info_path, image_width, image_height):
    camera_info = {}
    with open(camera_info_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("focal_len:"):
                camera_info['focal_len'] = float(line.split(":")[1].strip())  # 单位：mm
            elif line.startswith("baseline:"):
                camera_info['baseline'] = float(line.split(":")[1].strip())  # 单位：m
            elif line.startswith("sensor_size:"):
                camera_info['sensor_size'] = float(line.split(":")[1].strip())  # 单位：mm（这里我们假设是 sensor width）
 
    # 计算内参 fx, fy
    sensor_width = camera_info['sensor_size']
    sensor_height = sensor_width * image_height / image_width  # 假设 sensor 高宽比与图像一致
 
    focal_len = camera_info['focal_len']
    fx = focal_len / sensor_width * image_width
    fy = focal_len / sensor_height * image_height
    cx = image_width / 2
    cy = image_height / 2
 
    # 附加这些参数
    camera_info['fx'] = fx
    camera_info['fy'] = fy
    camera_info['cx'] = cx
    camera_info['cy'] = cy
 
    return camera_info
 
 
def ssim_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.mean(torch.clamp((1 - SSIM) / 2, 0, 1))
 
def calculate_aop_mae(pred_aop, gt_aop):
    """
    Calculate Mean Angular Error (MAE) for Angle of Polarization (AoP) in degrees.
    Handles both torch.Tensor and numpy.ndarray inputs.
    
    Args:
        pred_aop: Predicted AoP (torch.Tensor or np.ndarray)
        gt_aop: Ground truth AoP (torch.Tensor or np.ndarray)
        
    Returns:
        float: Mean angular error in degrees
    """
    # 检查输入是否为None
    if pred_aop is None or gt_aop is None:
        raise ValueError("pred_aop and gt_aop cannot be None")
    
    # 检查数据类型是否一致
    if type(pred_aop) != type(gt_aop):
        raise TypeError(f"pred_aop and gt_aop must be same type. Got {type(pred_aop)} and {type(gt_aop)}")
    
    datatype = gt_aop.dtype
    
    if isinstance(gt_aop, torch.Tensor):
        # PyTorch tensor path
        if datatype == torch.float32 or datatype == torch.float64:
            # 确保两个tensor在同一个设备上
            if pred_aop.device != gt_aop.device:
                pred_aop = pred_aop.to(gt_aop.device)
                
            error = torch.abs(pred_aop - gt_aop)
            angular_error = torch.minimum(error, torch.abs(error - torch.pi))
            mean_angular_error = torch.mean(angular_error) / torch.pi * 180
            
            # 返回Python float，方便累加
            return mean_angular_error.item()
        else:
            raise ValueError(f"Unsupported tensor dtype: {datatype}. Expected torch.float32 or torch.float64")
            
    elif isinstance(gt_aop, np.ndarray):
        # NumPy array path
        if datatype == np.float32 or datatype == np.float64:
            error = np.abs(pred_aop - gt_aop)
            angular_error = np.minimum(error, np.abs(error - np.pi))
            mean_angular_error = np.mean(angular_error) / np.pi * 180
            
            # 返回Python float
            return float(mean_angular_error)
        else:
            raise ValueError(f"Unsupported numpy dtype: {datatype}. Expected np.float32 or np.float64")
            
    else:
        raise TypeError(f"Unsupported input type: {type(gt_aop)}. Expected torch.Tensor or numpy.ndarray")
# def calculate_aop_mae(pred_aop, gt_aop):
#     datatype = gt_aop.dtype
#     if datatype == torch.float32 or datatype == torch.float64:
#         error = torch.abs(pred_aop - gt_aop)
#         angular_error = torch.minimum(error, torch.abs(error - torch.pi))
#         mean_angular_error = torch.mean(angular_error) / torch.pi * 180
#         return mean_angular_error
#     elif datatype == np.float32 or datatype == np.float64:
#         error = np.abs(pred_aop - gt_aop)
#         angular_error = np.minimum(error, np.abs(error - np.pi))
#         mean_angular_error = np.mean(angular_error) / np.pi * 180
#         return mean_angular_error
        
#     else:
#         print("Erorr Datatype When Calculate AoP MAE !")
 
 
def compute_aop_dop_from_Ii(I0, I45, I90, I135):
 
    if isinstance(I0, np.ndarray):
 
        s0 = (I0 + I45 + I90 + I135) / 2
        s1 = I0 - I90
        s2 = I45 - I135
 
        aop = np.arctan2(s2, s1) / 2
        dop = np.clip(np.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8), 0, 1)
 
    elif isinstance(I0, torch.Tensor):
 
        s0 = (I0 + I45 + I90 + I135) / 2
        s1 = I0 - I90
        s2 = I45 - I135
 
        aop = torch.atan2(s2, s1) / 2
        dop = torch.clamp(torch.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8), 0, 1)
 
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
 
    return s0, s1, s2, aop, dop
 
 
def compute_Ii_from_aop_dop_unpol(aop, dop, I_un):
 
    if isinstance(aop, np.ndarray):
        s0 = I_un * 2
        s1 = dop * s0 * np.cos(2 * aop)
        s2 = dop * s0 * np.sin(2 * aop)
 
        I0 = (s0 + s1) / 2
        I45 = (s0 + s2) / 2
        I90 = (s0 - s1) / 2
        I135 = (s0 - s2) / 2
 
    elif isinstance(aop, torch.Tensor):
        s0 = I_un * 2
        s1 = dop * s0 * torch.cos(2 * aop)
        s2 = dop * s0 * torch.sin(2 * aop)
 
        I0 = (s0 + s1) / 2
        I45 = (s0 + s2) / 2
        I90 = (s0 - s1) / 2
        I135 = (s0 - s2) / 2
 
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
 
    return I0, I45, I90, I135
 
 
def compute_aop_dop_from_0_45_unopl(I0, I45, I_un):
 
    if isinstance(I0, np.ndarray):
 
        s0 = I_un * 2
        s1 = 2 * (I0 - I_un)
        s2 = 2 * (I45 - I_un)
 
        aop = np.arctan2(s2, s1) / 2
        dop = np.clip(np.sqrt(s1 ** 2 + s2 ** 2 + 1e-8) / (s0 + 1e-8), 0, 1)
 
    elif isinstance(I0, torch.Tensor):
 
        s0 = I_un * 2
        s1 = 2 * (I0 - I_un)
        s2 = 2 * (I45 - I_un)
 
        aop = torch.atan2(s2, s1) / 2
        dop = torch.clamp(torch.sqrt(s1 ** 2 + s2 ** 2 + 1e-8) / (s0 + 1e-8), 0, 1)
 
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
 
    return s0, s1, s2, aop, dop
 
def compute_aop_dop_from_s1_s2_unpol(s1, s2, I_un):
    
    if isinstance(s1, np.ndarray):
 
        s0 = I_un * 2
 
        aop = np.arctan2(s2, s1) / 2
        dop = np.clip(np.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8), 0, 1)
 
    elif isinstance(s1, torch.Tensor):
 
        s0 = I_un * 2
 
        aop = torch.atan2(s2, s1) / 2
        dop = torch.clamp(torch.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8), 0, 1)
 
    else:
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")
 
    return aop, dop
 
# def warp_with_disp(img, disparity, direction='L2R', mask=None):
#     if disparity.ndim == 3:
#         disparity = disparity[:, :, 0]
#     h, w = img.shape[:2]
#     x = np.arange(w).reshape(1, -1).repeat(h, axis=0)
#     y = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
#     x = x.astype(np.float32)
#     y = y.astype(np.float32)
#     if direction == 'L2R':
#         x += disparity
#     elif direction == 'R2L':
#         x -= disparity
#     warpped_img = cv2.remap(img, x, y, cv2.INTER_LINEAR)
    
#     if mask is not None:
#         mask = mask.astype(np.bool8)
#         warpped_img[~mask] = 0
#     return warpped_img
 
def warp_with_disp(img, disparity, direction='L2R', target_img=None, mask=None):
    # 统一类型检查逻辑
    is_numpy = isinstance(img, np.ndarray) and isinstance(disparity, np.ndarray)
    is_torch = isinstance(img, torch.Tensor) and isinstance(disparity, torch.Tensor)
    
    if not (is_numpy or is_torch):
        raise TypeError("Inputs must be both numpy arrays or both torch tensors")
 
    # Numpy处理分支 (保持原有逻辑)
    if isinstance(img, np.ndarray) and isinstance(disparity, np.ndarray):
        # 维度校验
        if img.ndim != 3:
            raise ValueError(f"Numpy image expects HWC format, got {img.shape}")
        if disparity.ndim == 3:
            disparity = disparity[:, :, 0]
        if disparity.shape != img.shape[:2]:
            raise ValueError(f"Disparity shape {disparity.shape} mismatch with image {img.shape[:2]}")
 
        disparity = disparity.astype(np.float32)
        
 
        h, w = img.shape[:2]
        x = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
        y = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)
 
        # 坐标变换
        if direction == 'L2R':
            x = x + disparity
        elif direction == 'R2L':
            x = x - disparity
        else:
            raise ValueError(f"Invalid direction: {direction}")
 
        # 确保 map 为连续的 np.float32 数组，以匹配 OpenCV 的 remap 要求
        map_x = np.ascontiguousarray(x, dtype=np.float32)
        map_y = np.ascontiguousarray(y, dtype=np.float32)
 
        # 执行 warp（明确指定参数名以通过类型检查）
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
 
        # 处理mask
        if mask is not None:
            # 将 mask 安全地转换为布尔类型，兼容 numpy 的标准布尔类型
            if mask.dtype != np.bool_ and mask.dtype is not bool:
                mask = mask.astype(bool)
            warped_img[~mask] = 0
 
        # 替换误差大区域
        if target_img is not None:
            error_threshold = 0.00  # 误差阈值
            if target_img.shape != img.shape:
                raise ValueError(f"target_img shape {target_img.shape} does not match img {img.shape}")
 
            # 计算误差并找出高误差区域（L1误差）
            error = np.abs(warped_img.astype(np.float32) - target_img.astype(np.float32))
            error_map = np.mean(error, axis=-1)  # H x W
            mask_high_error = error_map < error_threshold
 
            # 统计超过阈值的像素数量
            high_error_pixels = np.sum(error_map >= error_threshold)  # 注意这里用的是 >=
            total_pixels = error_map.size
            high_error_percentage = (high_error_pixels / total_pixels) * 100
            
            # print(f"超过误差阈值的像素统计:")
            # print(f"总像素数: {total_pixels}")
            # print(f"超过阈值({error_threshold})的像素数: {high_error_pixels}")
            # print(f"占比: {high_error_percentage:.2f}%")
 
            # 替换高误差区域
            warped_img[mask_high_error] = target_img[mask_high_error]
 
        return warped_img
 
    # Torch处理分支 (新增逻辑)
    elif is_torch:
        # 维度校验
        if img.ndim != 4:
            raise ValueError(f"Torch image expects BCHW format, got {img.shape}")
        if disparity.ndim == 4:  # 处理B1HW格式
            disparity = disparity.squeeze(1)
        if disparity.ndim != 3:
            raise ValueError(f"Torch disparity expects B1HW or BHW format, got {disparity.shape}")
        if disparity.shape[-2:] != img.shape[-2:]:
            raise ValueError(f"Disparity shape {disparity.shape} mismatch with image {img.shape}")
 
        # 生成坐标网格
        B, C, H, W = img.shape
        device = img.device
        
        # 创建基础坐标 (batch维度通过expand处理)
        xx = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W).float()  # (B,H,W)
        yy = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W).float()
        
        # 应用视差
        if direction == 'L2R':
            xx = xx + disparity
        elif direction == 'R2L':
            xx = xx - disparity
        else:
            raise ValueError(f"Invalid direction: {direction}")
 
        # 构建采样网格 (需要归一化到[-1,1])
        xx_normalized = (xx / (W-1)) * 2 - 1  # 从[0,W-1]映射到[-1,1]
        yy_normalized = (yy / (H-1)) * 2 - 1
        grid = torch.stack([xx_normalized, yy_normalized], dim=-1)  # (B,H,W,2)
 
        # 执行可微分采样
        warped_img = F.grid_sample(
            input=img,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
 
        # 处理mask
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.ndim == 3:  # 扩展mask到和图像相同通道数
                mask = mask.unsqueeze(1).expand_as(img)
            warped_img[~mask] = 0
 
        # # 替换误差大区域
        # if target_img is not None:
        #     if target_img.shape != img.shape:
        #         raise ValueError(f"target_img shape {target_img.shape} does not match img {img.shape}")
            
        #     # 计算绝对误差
        #     error = torch.abs(warped_img - target_img)  # [B,C,H,W]
            
        #     # 计算动态阈值：每个像素阈值为目标值的10%
        #     dynamic_threshold = 0.2 * target_img  # [B,C,H,W]
            
        #     # 创建高误差掩码（逐像素比较）
        #     mask_high_error = error > dynamic_threshold  # [B,C,H,W]
            
        #     # 替换高误差区域
        #     warped_img[mask_high_error] = target_img[mask_high_error]
 
        # return warped_img
 
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
 
    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]
 
    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
 
def compute_image_metrics(pred, gt, max_value=1):
    # print(f"pred shape: {pred.shape}, gt shape: {gt.shape}")
    psnr_value = psnr(pred, gt, data_range=max_value)
    ssim_value = ssim(pred, gt, data_range=max_value, multichannel=True, channel_axis=-1)
 
    return psnr_value, ssim_value
 
def normal_from_depth_perspective(depth, focal_len, sensor_size):
    """
    从深度图计算法线图（透视投影）。
 
    参数:
    depth: torch.Tensor or numpy.ndarray
        深度图，形状为 (B, 1, H, W)、(H, W, 1) 或 (H, W)
    focal_len: float, numpy.ndarray, or torch.Tensor
        焦距（单位：米），如果 depth 是 batch，则为形状 (B, 1, 1, 1)
    sensor_size: float, numpy.ndarray, or torch.Tensor
        传感器宽/高（单位：米），如果 depth 是 batch，则为形状 (B, 1, 1, 1)
 
    返回:
    法线图 (同类型为输入)：(B, H, W, 3) 或 (H, W, 3)
    """
    is_torch = isinstance(depth, torch.Tensor)
    is_numpy = isinstance(depth, np.ndarray)
 
    if is_numpy:
        depth = torch.from_numpy(depth).unsqueeze(0) if depth.ndim == 2 else torch.from_numpy(np.transpose(depth, (2, 0, 1)))
        focal_len = torch.tensor(focal_len).float().view(1, 1, 1, 1)
        sensor_size = torch.tensor(sensor_size).float().view(1, 1, 1, 1)
        device = torch.device('cpu')
    elif is_torch:
        device = depth.device
        focal_len = focal_len.to(device) if isinstance(focal_len, torch.Tensor) else torch.tensor(focal_len, device=device).view(1, 1, 1, 1)
        sensor_size = sensor_size.to(device) if isinstance(sensor_size, torch.Tensor) else torch.tensor(sensor_size, device=device).view(1, 1, 1, 1)
    else:
        raise ValueError("depth must be a torch.Tensor or numpy.ndarray")
 
    depth = depth.float()
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(0)  # (1, 1, H, W)
 
    B, _, H, W = depth.shape
 
    # 创建像素坐标网格
    u = torch.linspace(-W / 2, W / 2, W, device=device)
    v = torch.linspace(-H / 2, H / 2, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    uu = uu.view(1, 1, H, W).expand(B, -1, -1, -1)
    vv = vv.view(1, 1, H, W).expand(B, -1, -1, -1)
 
    m2pix_x = W / sensor_size  # (B, 1, 1, 1)
    m2pix_y = H / sensor_size
    sensor_xx = uu / m2pix_x
    sensor_yy = vv / m2pix_y
 
    # 计算深度梯度
    du = depth[:, 0, :, :].unsqueeze(1)
    dv = depth[:, 0, :, :].unsqueeze(1)
    du = torch.gradient(du, dim=-1)[0]
    dv = torch.gradient(dv, dim=-2)[0]
 
    dZ_sensor_x = -du * 2 * m2pix_x / W
    dZ_sensor_y = -dv * 2 * m2pix_y / H
 
    nx = -dZ_sensor_x * focal_len
    ny = dZ_sensor_y * focal_len
    nz = (depth + sensor_xx * dZ_sensor_x + sensor_yy * dZ_sensor_y) / 700
 
    normal = torch.cat([nx, ny, nz], dim=1)  # (B, 3, H, W)
    normal = normal.permute(0, 2, 3, 1)  # (B, H, W, 3)
 
    norm = torch.linalg.norm(normal, dim=3, keepdim=True)
    normal = normal / (norm + 1e-8)
 
    if is_numpy:
        return normal.squeeze(0).cpu().numpy()
    else:
        return normal.permute(0, 3, 1, 2)  # (B, H, W, 3)
 
 
def compute_azimuth_from_normal(normal):
    """
    计算法线的方位角（Azimuth）。
 
    参数:
    normal (torch.Tensor or numpy.ndarray): 法线图，torch为 (B, 3, H, W)，numpy为 (H, W, 3)
 
    返回:
    torch.Tensor or numpy.ndarray: 方位角图，torch为 (B, 1, H, W)，numpy为 (H, W, 1)
    """
    is_torch = isinstance(normal, torch.Tensor)
    is_numpy = isinstance(normal, np.ndarray)
 
    if is_torch:
        # 输入为 (B, 3, H, W)
        x = normal[:, 0, :, :]
        y = normal[:, 1, :, :]
        azimuth = torch.atan2(y, x).unsqueeze(1)  # (B, 1, H, W)
        return azimuth
 
    elif is_numpy:
        # 输入为 (H, W, 3)
        x = normal[..., 0]
        y = normal[..., 1]
        azimuth = np.arctan2(y, x)[..., np.newaxis]  # (H, W, 1)
        return azimuth
 
    else:
        raise ValueError("输入必须是 torch.Tensor 或 numpy.ndarray")