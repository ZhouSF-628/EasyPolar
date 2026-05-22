import argparse
import random
import time
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
import os
import os.path as osp
import cv2
import logging
import torch.nn.functional as F
from tqdm import tqdm
 
# 配置 logger
logging.Formatter.converter = time.localtime 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model.EasyPolar2 import EasyPolarNet
 
from utils.utils import *
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser()
 
parser.add_argument('--train_data', type=str, default='/home/bupt803/Zsf/Dataset/EasyPolar/PolarTripleView/all')
parser.add_argument('--test_data', type=str, default='/home/bupt803/Zsf/Dataset/EasyPolar/PolarTripleView/test')
parser.add_argument('--output', type=str, default='./result/20260214_syn')
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument('--log_dir', type=str, default='../tf_dir')
 
args = parser.parse_args()
 
 
class WeightedFocusLoss(nn.Module):
    """
    加权关注 Loss:
    - mode='high': 值越大权重越大 (用于 DoP)
    - mode='low':  值越小权重越大 (用于 Conf)
    """
    def __init__(self, mode='high', weight_factor=5.0):
        super().__init__()
        assert mode in ['high', 'low']
        self.mode = mode
        self.factor = weight_factor
        self.l1 = nn.L1Loss(reduction='none') # 必须是 none
 
    def forward(self, pred, gt):
        pixel_loss = self.l1(pred, gt)
        target = gt.detach()
        
        if self.mode == 'high':
            # GT 越大(如0.8)，权重越大 -> 关注高反光区域
            weights = 1.0 + self.factor * target
        else:
            # GT 越小(如0.1)，权重越大 -> 关注低置信度(错误)区域
            weights = 1.0 + self.factor * (1.0 - target)
            
        return (pixel_loss * weights).mean()
 
class GradientLoss(nn.Module):
    """ 计算图像梯度的 L1 Loss，强迫输出平滑连续 """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
 
    def forward(self, pred, gt):
        # x方向梯度: 右 - 左
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        gt_dx = torch.abs(gt[:, :, :, :-1] - gt[:, :, :, 1:])
        # y方向梯度: 下 - 上
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        gt_dy = torch.abs(gt[:, :, :-1, :] - gt[:, :, 1:, :])
        
        return self.l1(pred_dx, gt_dx) + self.l1(pred_dy, gt_dy)
    
 
class data_io(data.Dataset):
    def __init__(self, args, train=False, train_real=False):
        self.train_real = train_real
        self.train = train
        self.args = args
        self.image_list = []        # list of image paths
        self.normal_list = []
        self.view_list = []         # list of view directions (Plucker coordinates)
        # self.disparity_list = []    # list of disparity paths
        self.load_file()            # load the file paths
 
    def load_file(self):
        if self.train:
            root = self.args.train_data
        else:
            root = self.args.test_data
        print(f"Loading data from {root} ...")
        assert os.path.exists(root)
 
        file_list = os.listdir(root)
        
        L_pol0_list = sorted([osp.join(root, f'{idx}/preprocess/L_pol0.png') for idx in file_list])
        M_unpol_list = sorted([osp.join(root, f'{idx}/middle/img/S0.png') for idx in file_list])
        M_pol0_list = sorted([osp.join(root, f'{idx}/middle/img/I0.png') for idx in file_list])
        M_pol45_list = sorted([osp.join(root, f'{idx}/middle/img/I45.png') for idx in file_list])
        R_pol45_list = sorted([osp.join(root, f'{idx}/preprocess/R_pol45.png') for idx in file_list])
        # depth_list = sorted([osp.join(root, f'{name}/depth.npy') for name in file_list])
        normal_list = sorted([osp.join(root, f'{idx}/middle/normal/normal.npy') for idx in file_list])
        view_L_list = sorted([osp.join(root, f'{idx}/preprocess/view_L.npy') for idx in file_list])
        view_M_list = sorted([osp.join(root, f'{idx}/preprocess/view_M.npy') for idx in file_list])
        view_R_list = sorted([osp.join(root, f'{idx}/preprocess/view_R.npy') for idx in file_list])
 
        assert len(L_pol0_list) == len(M_unpol_list) == len(M_pol0_list) == len(M_pol45_list) == len(R_pol45_list) == len(normal_list) == len(view_L_list) == len(view_M_list) == len(view_R_list) > 0, [L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, normal_list, view_L_list, view_M_list, view_R_list]
 
        for L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, normal, view_L, view_M, view_R in tqdm(zip(L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list,  normal_list, view_L_list, view_M_list, view_R_list)):
            L_pol0_image = read_file(L_pol0)[:, 100:-100, :]    # 裁剪
            M_unpol_image = read_file(M_unpol)[:, 100:-100, :]
            M_pol0_image = read_file(M_pol0)[:, 100:-100, :]
            M_pol45_image = read_file(M_pol45)[:, 100:-100, :]
            R_pol45_image = read_file(R_pol45)[:, 100:-100, :]
            normal = np.load(normal)[:, 100:-100, :]
            view_L = np.load(view_L)[:, 100:-100, :]  # [H, W, 6]
            view_M = np.load(view_M)[:, 100:-100, :]
            view_R = np.load(view_R)[:, 100:-100, :]
 
            self.image_list += [ [L_pol0_image, M_unpol_image, M_pol0_image, M_pol45_image, R_pol45_image] ]
            self.normal_list += [normal]
            self.view_list += [ [view_L, view_M, view_R] ]
 
    def data_augmentation(self, data_pack, patch_size=128):
        H, W = data_pack.shape[:2]
 
        y = random.randint(0, H - patch_size)
        x = random.randint(0, W - patch_size)
 
        patch = data_pack[y:y + patch_size, x:x + patch_size]
        
        if random.random() < 0.5:
            patch = np.ascontiguousarray(np.flipud(patch))
 
        return patch
 
 
    def __len__(self):
        return len(self.image_list)
 
    def __getitem__(self, index):
        L_pol0 = self.image_list[index][0]
        M_unpol = self.image_list[index][1]
        M_pol0 = self.image_list[index][2]
        M_pol45 = self.image_list[index][3]
        R_pol45 = self.image_list[index][4]
        normal = self.normal_list[index]
        view_L = self.view_list[index][0]
        view_M = self.view_list[index][1]
        view_R = self.view_list[index][2]
 
        data_pack = np.concatenate([L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, normal, view_L, view_M, view_R], axis=-1)
        # data_pack = np.concatenate([warp_pol0, M_unpol, M_pol0, M_pol45, warp_pol45], axis=-1)
 
        if self.train:
            data_pack = self.data_augmentation(data_pack, patch_size=self.args.patch_size)
        
        L_pol0 = torch.from_numpy(data_pack[..., :3]).permute(2, 0, 1).float()
        M_unpol = torch.from_numpy(data_pack[..., 3:6]).permute(2, 0, 1).float()
        M_pol0 = torch.from_numpy(data_pack[..., 6:9]).permute(2, 0, 1).float()
        M_pol45 = torch.from_numpy(data_pack[..., 9:12]).permute(2, 0, 1).float()
        R_pol45 = torch.from_numpy(data_pack[..., 12:15]).permute(2, 0, 1).float()
        normal = torch.from_numpy(data_pack[..., 15:18]).permute(2, 0, 1).float()  # [3, H, W]
        views = torch.from_numpy(data_pack[..., 18:]).permute(2, 0, 1).float()  # [H, W, 18]
 
        return L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, normal, views
 
def test_model(model, test_data_loader, epoch, device, best_mae, save_path):
    model.eval()
    avg_mae = 0
    avg_psnr_dop = 0
    avg_psnr_0 = 0
    avg_psnr_45 = 0
 
    index = 0
    patch_size = 640
    padding = 64
 
    with torch.no_grad():
        for batch in test_data_loader:
            # 1. 搬运数据
            L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, normal, lights = [x.to(device) for x in batch]
            
            B, C, H, W = M_unpol.shape
            
            # 2. 预先对【整张大图】进行 Padding
            # 这样我们在切片时，边缘的 Patch 也能取到(反射的)边界，
            # 而中间的 Patch 能取到(真实的)邻居。
            # 这里的 padding 对应上面的 'pad=8'，但我们要 pad 整个大图
            L_pol0_pad = F.pad(L_pol0, (padding, padding, padding, padding), mode='reflect')
            M_unpol_pad = F.pad(M_unpol, (padding, padding, padding, padding), mode='reflect')
            R_pol45_pad = F.pad(R_pol45, (padding, padding, padding, padding), mode='reflect')
            normal_pad = F.pad(normal, (padding, padding, padding, padding), mode='reflect')
            lights_pad = F.pad(lights, (padding, padding, padding, padding), mode='reflect')
            
            # 初始化输出画布 (原始尺寸 H, W)
            out_sin = torch.zeros((B, 3, H, W), device=device) # 假设 AoP 是 6 通道
            out_cos = torch.zeros((B, 3, H, W), device=device)
            out_dop = torch.zeros((B, 3, H, W), device=device)
            out_conf = torch.zeros((B, 1, H, W), device=device)
            
            # 3. 滑动窗口推理
            # 步长(stride) 等于 patch_size，保证输出无缝拼接
            for y in range(0, H, patch_size):
                for x in range(0, W, patch_size):
                    
                    # --- A. 计算当前 Patch 在【原图】中的有效范围 ---
                    h_start, w_start = y, x
                    h_end = min(y + patch_size, H)
                    w_end = min(x + patch_size, W)
                    
                    # 当前有效区域的实际大小 (边缘可能小于 1024)
                    h_eff, w_eff = h_end - h_start, w_end - w_start
                    
                    # --- B. 计算在【Padded 大图】中的切片坐标 ---
                    # 输入需要包含周围的 padding 上下文
                    # 坐标映射: Padded图坐标 = 原图坐标 + padding
                    # 我们要切: [y : y+h_eff+2*padding, x : x+w_eff+2*padding]
                    input_slice_y = h_start # Padded图上的起始点 = h_start + padding - padding
                    input_slice_x = w_start 
                    
                    # 这里的切片逻辑：
                    # 中心有效区在 Padded图的 [y+p : y+p+h_eff, x+p : x+p+w_eff]
                    # 我们需要它的上下文，所以切 [y : y+h_eff+2p, x : x+w_eff+2p]
                    
                    patch_L = L_pol0_pad[:, :, y : y + h_eff + 2*padding, x : x + w_eff + 2*padding]
                    patch_M = M_unpol_pad[:, :, y : y + h_eff + 2*padding, x : x + w_eff + 2*padding]
                    patch_R = R_pol45_pad[:, :, y : y + h_eff + 2*padding, x : x + w_eff + 2*padding]
                    patch_norm = normal_pad[:, :, y : y + h_eff + 2*padding, x : x + w_eff + 2*padding]
                    patch_light = lights_pad[:, :, y : y + h_eff + 2*padding, x : x + w_eff + 2*padding]
                    
                    # --- C. 推理 ---
                    # 此时输入包含：左边padding + 有效区 + 右边padding
                    # 上下文是真实的邻居像素！
                    pred_sin_cos, pred_dop, pred_conf = model(patch_M, patch_L, patch_R, patch_norm, patch_light) # 注意你的forward参数顺序
                    
                    # --- D. 裁剪 (Crop) ---
                    # 预测出来的结果也是带 padding 的，我们只取中间的有效部分
                    # 裁剪掉四周的 padding
                    valid_sin_cos = pred_sin_cos[:, :, padding : padding + h_eff, padding : padding + w_eff]
                    valid_dop = pred_dop[:, :, padding : padding + h_eff, padding : padding + w_eff]
                    valid_conf = pred_conf[:, :, padding : padding + h_eff, padding : padding + w_eff]
                    
                    # --- E. 填回画布 ---
                    out_sin[:, :, h_start:h_end, w_start:w_end] = valid_sin_cos[:, 0:3]
                    out_cos[:, :, h_start:h_end, w_start:w_end] = valid_sin_cos[:, 3:6]
                    out_dop[:, :, h_start:h_end, w_start:w_end] = valid_dop
                    out_conf[:, :, h_start:h_end, w_start:w_end] = valid_conf
 
            # 4. 后处理 (AoP sin/cos -> angle)
            out_aop = 0.5 * torch.atan2(out_sin, out_cos)
            # sin_aop = out_aop[:, 0:3, :, :]
            # cos_aop = out_aop[:, 3:6, :, :]
            # out_aop = 0.5 * torch.atan2(sin_aop, cos_aop)
 
            out_aop = out_aop.clamp(-np.pi/2, np.pi/2)
            out_dop = out_dop.clamp(0, 1)
 
            M_unpol = M_unpol.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            M_pol0 = M_pol0.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            M_pol45 = M_pol45.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            L_pol0 = L_pol0.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            # M_out_0 = out_0.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            R_pol45 = R_pol45.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            # M_out_45 = out_45.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            aop_out = out_aop.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            dop_out = out_dop.squeeze().permute(1, 2, 0).detach().cpu().numpy()
 
            # _, _, _, aop_out, dop_out = compute_aop_dop_from_0_45_unopl(M_out_0, M_out_45, M_unpol)
            _, _, _, aop_gt, dop_gt = compute_aop_dop_from_0_45_unopl(M_pol0, M_pol45, M_unpol)
            M_out_0, M_out_45, _, _ = compute_Ii_from_aop_dop_unpol(aop_out, dop_out, M_unpol)
            _, _, _, init_aop, init_dop = compute_aop_dop_from_0_45_unopl(L_pol0, R_pol45, M_unpol)
            # print(aop_out.min(), aop_out.max(), dop_gt.min(), dop_gt.max())
 
            psnr_0 = psnr(M_pol0, M_out_0, data_range=1)
            psnr_45 = psnr(M_pol45, M_out_45, data_range=1)
            avg_psnr_0 += psnr_0
            avg_psnr_45 += psnr_45
 
            mae = calculate_aop_mae(aop_out, aop_gt)
            avg_mae += mae
 
            # input_mae = calculate_aop_mae(aop_in, aop_gt)
            # print(f"Index: {index}, Input AoP MAE: {input_mae:.4f}, Output AoP MAE: {mae:.4f}")
 
            psnr_dop = psnr(dop_gt, dop_out, data_range=1)
            avg_psnr_dop += psnr_dop
        
 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
 
            plt.figure(figsize=(20, 10))
            plt.subplot(2, 4, 1)
            plt.imshow(aop_gt.mean(-1), cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
            plt.title('GT AoP')
            plt.colorbar()
            plt.axis('off')
 
            plt.subplot(2, 4, 2)
            plt.imshow(aop_out.mean(-1), cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
            plt.title(f'Output AoP (MAE: {mae:.2f})')
            plt.axis('off')
 
            error_map = np.abs(aop_out - aop_gt).mean(-1)
            plt.subplot(2, 4, 3)
            plt.imshow(error_map, cmap='jet')
            plt.title('Error Map (AoP)')
            plt.colorbar()
            plt.axis('off')
 
            plt.subplot(2, 4, 5)
            plt.imshow(dop_gt.mean(-1), cmap='jet', vmin=0, vmax=1)
            plt.title('GT DoP')
            plt.colorbar()
            plt.axis('off')
 
            plt.subplot(2, 4, 6)
            plt.imshow(dop_out.mean(-1), cmap='jet', vmin=0, vmax=1)
            plt.title(f'Output DoP (PSNR: {psnr_dop:.2f})')
            plt.axis('off')
 
            error_map = np.abs(dop_out - dop_gt).mean(-1)
            plt.subplot(2, 4, 7)
            plt.imshow(error_map, cmap='jet')
            plt.title('Error Map (DoP)')
            plt.colorbar()
            plt.axis('off')
 
            error_map = np.abs(init_aop - aop_gt).mean(-1)
            plt.subplot(2, 4, 4)
            plt.imshow(error_map, cmap='jet', vmin=0, vmax=np.pi)
            plt.colorbar()
            plt.title('Input AoP Error Map')
            plt.axis('off')
 
            conf_map = out_conf.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().mean(-1)
            plt.subplot(2, 4, 8)
            plt.imshow(conf_map, cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title('Confidence Map')
            plt.axis('off')
 
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{index}.png'))
            plt.close()

            if not os.path.exists(os.path.join(save_path, f'{index:03d}')):
                os.makedirs(os.path.join(save_path, f'{index:03d}'))
            cv2.imwrite(os.path.join(save_path, f'{index:03d}/pol0.png'), (M_out_0 * 65535).astype(np.uint16))
            cv2.imwrite(os.path.join(save_path, f'{index:03d}/pol45.png'), (M_out_45 * 65535).astype(np.uint16))
            aop_map = cv2.applyColorMap(((aop_out + np.pi/2) / np.pi * 255).astype(np.uint8), cv2.COLORMAP_HSV)
            dop_map = cv2.applyColorMap((dop_out * 255).astype(np.uint8), cv2.COLORMAP_JET)
 
            cv2.imwrite(os.path.join(save_path, f'{index:03d}/aop.png'), aop_map)
            cv2.imwrite(os.path.join(save_path, f'{index:03d}/dop.png'), dop_map)
 
            index += 1
 
    avg_mae = avg_mae / len(test_data_loader)
    avg_psnr_dop = avg_psnr_dop / len(test_data_loader)
    avg_psnr_0 /= len(test_data_loader)
    avg_psnr_45 /= len(test_data_loader)

    if avg_mae < best_mae:
        best_mae = avg_mae
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
 
    logger.info(f"===> Test Epoch: {epoch}, MAE: {avg_mae:.5f}, PSNR DoP: {avg_psnr_dop:.5f}")
 
    return best_mae


def check_data_health(data, name="Input"):
    """
    递归检查数据中是否包含 NaN 或 Inf。
    支持结构：Tensor, List[Tensor], Tuple[Tensor], Dict[key: Tensor]
    """
    has_error = False

    # 1. 如果是 Tensor，直接检查
    if isinstance(data, torch.Tensor):
        if torch.isnan(data).any():
            print(f"❌ [严重错误] {name} 中发现 NaN!")
            has_error = True
        if torch.isinf(data).any():
            print(f"❌ [严重错误] {name} 中发现 Inf (无穷大)!")
            has_error = True
        
        # 顺便打印一下统计信息，帮你看是否有数值异常（比如特别大或负数）
        if has_error:
            print(f"   -> 统计信息: Min={data.min().item():.4f}, Max={data.max().item():.4f}, Mean={data.mean().item():.4f}")
            return True # 发现错误直接返回
        return False

    # 2. 如果是 列表 或 元组 (例如 [I0, I45, I90])
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            # 递归检查每一个元素
            if check_data_health(item, name=f"{name}[{i}]"):
                has_error = True

    # 3. 如果是 字典 (例如 {'image': ..., 'label': ...})
    elif isinstance(data, dict):
        for key, item in data.items():
            if check_data_health(item, name=f"{name}['{key}']"):
                has_error = True
    
    return has_error


def main():
    logger.info(args)
    torch.backends.cudnn.benchmark = True
 
    # 设置随机种子
    torch.manual_seed(args.seed)
    seed = args.seed
 
    logger.info(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
 
    # 设置 GPU
    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    
    # 加载数据
    logger.info("===> Loading datasets")
 
    train_dataset = data_io(args, train=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info(f"Training data size: {len(train_dataset)}")
 
    test_dataset = data_io(args, train=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    logger.info(f"Test data size: {len(test_dataset)}")
 
    # 加载模型
    logger.info("===> Building models")
    # model = MainReconstructionNet(in_ch=3, feat_ch=64, out_channels=3, patch=3, refine_blocks=3).to(device)
    # model = PolarRecNet(p_channels=3, i_channels=3, out_channels=3).to(device)
    model = EasyPolarNet().to(device)
    # model.load_state_dict(torch.load('./result/20260204_syn/best.pth'))
 
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
 
    l1loss = nn.L1Loss().to(device)
    criterion_base = nn.L1Loss().to(device) # 普通 L1
    criterion_dop  = WeightedFocusLoss(mode='high', weight_factor=5.0).to(device) # DoP 专用
    criterion_conf = WeightedFocusLoss(mode='low',  weight_factor=5.0).to(device) # Conf 专用
    criterion_grad = GradientLoss().to(device) # AoP 梯度专用
    # mseloss = nn.MSELoss().to(device)
    best_mae = 90
 
    # 开始训练
    logger.info("===> Start Training")
    for epoch in range(args.epochs):
        model.train()
        avg_image_loss = 0
        avg_aop_loss = 0
        avg_dop_loss = 0
 
        epoch_stats = {'total': 0, 'aop': 0, 'dop': 0, 'conf': 0, 'int': 0}
        num_batches = len(train_data_loader)
        var_names = ["L_pol0", "M_unpol", "M_pol0", "M_pol45", "R_pol45", "normal", "lights"]
        for train_data in train_data_loader:
            if check_data_health(train_data, name="Batch_Data"):
                
                # 2. 如果总检查挂了，开始逐个变量排查
                # 使用 zip 把数据和名字配对
                found_culprit = False
                for tensor_val, name_str in zip(train_data, var_names):
                    # 对单个变量进行检查，传入具体的名字
                    if check_data_health(tensor_val, name=name_str):
                        print(f"   👉 找到元凶了！变量名: 【{name_str}】")
                        # 如果是 tensor，顺便打印一下具体数值情况
                        if hasattr(tensor_val, 'min'):
                            print(f"      数值范围: Min={tensor_val.min()}, Max={tensor_val.max()}")
                        found_culprit = True

                if not found_culprit:
                    print("   ⚠️ 奇怪，总检查报错但分项检查未发现，可能是列表结构嵌套层级问题。")
                    
                import sys; sys.exit()
    
            L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, normal, lights = [x.to(device) for x in train_data]
            # print(L_pol0.shape, M_unpol.shape, M_pol0.shape, M_pol45.shape, R_pol45.shape, lights.shape)

            if torch.isnan(L_pol0).any() or torch.isinf(L_pol0).any():
                print("🚨 破案了！是输入数据本身含有 NaN 或 Inf！请检查 DataLoader。")
                # 可以选择打印出是第几个 batch 或者保存下这个坏数据看看
                import sys; sys.exit()
 
            optimizer.zero_grad()
 
            aop_out, dop_out, conf = model(M_unpol, L_pol0, R_pol45, normal, lights)
            sin_out = aop_out[:, 0:3, :, :]
            cos_out = aop_out[:, 3:6, :, :]
            aop_out = 0.5 * torch.atan2(sin_out, cos_out)

            if torch.isnan(aop_out).any():
                print("❌ [前向传播] 网络输出直接就是 NaN！(可能是权重已经坏了，或者输入数据有 NaN)")
                import sys; sys.exit()
 
            # aop_out = 0.5 * torch.atan2(sin_out, cos_out)
            # print(f"GT Range: sin({torch.sin(2 * aop_out).min().item()}~{torch.sin(2 * aop_out).max().item()}); cos({torch.cos(2 * aop_out).min().item()}~{torch.cos(2 * aop_out).max().item()}); aop({aop_out.min().item()}~{aop_out.max().item()})")
            # print(f"Output Range: sin({sin_out.min().item()}~{sin_out.max().item()}); cos({cos_out.min().item()}~{cos_out.max().item()}); aop({aop_out.min().item()}~{aop_out.max().item()}")
            # _, _, _, aop_gt, dop_gt = compute_aop_dop_from_0_45_unopl(M_pol0, M_pol45, M_unpol)
            # sin_gt = torch.sin(2 * aop_gt)
            # cos_gt = torch.cos(2 * aop_gt)
            with torch.no_grad():
                # A. 计算中间视角的真实偏振参数 (作为标签)
                # 输出: I_tot, rho, theta, ...
                # 假设 compute_aop_dop 返回的 aop_gt 是弧度, dop_gt 是 [0,1]
                _, _, _, aop_gt, dop_gt = compute_aop_dop_from_0_45_unopl(M_pol0, M_pol45, M_unpol)
                
                # 转换为 Sin/Cos GT (用于 AoP Loss)
                sin_gt = torch.sin(2.0 * aop_gt)
                cos_gt = torch.cos(2.0 * aop_gt)
                
                # B. 动态生成 Conf GT (指数衰减)
                # 计算 Pseudo 参数 (基于左右视角的输入)
                # 注意: 这里最好加上 Warping，如果没有 Warping，直接算会有几何误差，正好用来训练 Conf
                _, _, _, pseudo_aop, pseudo_dop = compute_aop_dop_from_0_45_unopl(L_pol0, R_pol45, M_unpol)
                
                # 计算 Pseudo 与 GT 的物理误差
                # AoP 误差: 考虑周期性 (最小角度差)
                aop_diff = torch.abs(pseudo_aop - aop_gt)
                aop_err = torch.min(aop_diff, torch.pi - aop_diff) # (B, 3, H, W)
                # DoP 误差
                dop_err = torch.abs(pseudo_dop - dop_gt)           # (B, 3, H, W)
                
                # 综合误差 (取平均)
                total_err = (aop_err + dop_err) / 2.0
                total_err_mean = total_err.mean(dim=1, keepdim=True) # 变成 (B, 1, H, W)
                
                # 生成 Conf Target: 误差越大，Conf 越接近 0
                # beta=5.0 控制敏感度
                conf_gt = torch.exp(-5.0 * total_err_mean)
 
            pol0_out, pol45_out, _, _ = compute_Ii_from_aop_dop_unpol(aop_out, dop_out, M_unpol)
 
            pol0_loss = l1loss(pol0_out, M_pol0)
            pol45_loss = l1loss(pol45_out, M_pol45)
            loss_int = (pol0_loss + pol45_loss) / 2.0
            
            # loss_aop = l1loss(sin_gt, sin_out) + l1loss(cos_gt, cos_out)
            loss_aop_val = criterion_base(sin_out, sin_gt) + criterion_base(cos_out, cos_gt)
            loss_aop_grad = criterion_grad(sin_out, sin_gt) + criterion_grad(cos_out, cos_gt)
            loss_aop = loss_aop_val + 0.2 * loss_aop_grad # 梯度权重 0.2
            # aop_loss = l1loss(aop_gt, aop_out)
            # weight_factor = 5.0 
            # weight_map = 1.0 + weight_factor * dop_gt.detach()
            # loss_dop = l1loss(dop_gt, dop_out)
 
            loss_dop = criterion_dop(dop_out, dop_gt)
            # conf_loss = l1loss(conf, 1.0 - error.mean(dim=1, keepdim=True))
            loss_conf = criterion_conf(conf, conf_gt)
            # print(f'AoP Loss: {aop_loss.item():.5f}, DoP Loss: {dop_loss.item():.5f}, I0 Loss: {pol0_loss.item():.5f}, I45 Loss: {pol45_loss.item():.5f}')
 
            total_loss = 5.0 * loss_aop + 2.0 * loss_dop + 1.0 * loss_conf + 1.0 * loss_int
            # total_loss = aop_loss
 
            loss = total_loss

            if torch.isnan(loss):
                print("❌ [前向传播] Loss 计算结果是 NaN！(可能是 Log(0), Acos(>1), Sqrt(<0) 等导致)")
                import sys; sys.exit()
 
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     parameters=model.parameters(),
            #     max_norm=1.0,  # 推荐初始值，可根据实际情况调整
            #     norm_type=2,   # L2范数
            #     error_if_nonfinite=True
            # )

            # 【检查点 C】：检查梯度 (最关键的一步)
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"❌ [反向传播] 梯度爆炸！在层 {name} 发现 NaN 梯度")
                        has_nan_grad = True
                        break
                    if torch.isinf(param.grad).any():
                        print(f"⚠️ [反向传播] 梯度无穷大 (Inf)！在层 {name}")
                        has_nan_grad = True # Inf 通常紧接着会导致 NaN
                        break

            if has_nan_grad:
                print("🛑 停止更新权重，避免污染模型")
                # 这里可以选择跳过这一步 optimizer.step()，或者直接退出分析
                import sys; sys.exit()
            else:
                optimizer.step()

            optimizer.step()
            # avg_image_loss += pol0_loss.item() + pol45_loss.item()
            avg_aop_loss += loss_aop.item()
            avg_dop_loss += loss_dop.item()
            avg_image_loss += pol0_loss.item() + pol45_loss.item()
 
        # scheduler.step()
 
        avg_image_loss = avg_image_loss / len(train_data_loader)
        avg_aop_loss = avg_aop_loss / len(train_data_loader)
        avg_dop_loss = avg_dop_loss / len(train_data_loader)
        logger.info(f"Training Image Loss: {avg_image_loss:.5f}, AoP Loss: {avg_aop_loss:.5f}, DoP Loss: {avg_dop_loss:.5f}")
 
        if epoch % 10 == 0:
            best_mae = test_model(model, test_data_loader, epoch, device, best_mae, save_path=args.output)
 
 
if __name__ == '__main__':
    main()