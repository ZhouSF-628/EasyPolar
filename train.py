import argparse
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import logging  # 导入 logging 模块

# 配置 logger
logging.Formatter.converter = time.localtime 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.TriplePolarNet import TriplePolarNet

from utils.polar_dataset import TripleViewDataset
from utils.utils import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
 
parser.add_argument('--train_data', type=str, default='../../input1/PolarTripleView/train')
parser.add_argument('--test_data', type=str, default='../../input1/PolarTripleView/test')
parser.add_argument('--output', type=str, default='./result')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument('--log_dir', type=str, default='../tf_dir')
 
args = parser.parse_args()

def test_model(model, test_data_loader, epoch, device, best_mae, save_path, writer):
    model.eval()
    avg_mae = 0
    avg_psnr_dolp = 0
    avg_psnr_0 = 0
    avg_psnr_45 = 0
    avg_psnr_init = 0
    avg_mae_init = 0

    index = 0

    with torch.no_grad():
        for data in test_data_loader:
            L_pol0, M_unpol, M_pol0, M_pol45, R_pol45 = [x.to(device) for x in data]

            _, _, _, aop_gt, dolp_gt = compute_aop_dop_from_0_45_unopl(M_pol0, M_pol45, M_unpol)

            init_0, init_45, aop_out, dolp_out, _, _ = model(L_pol0, M_unpol, R_pol45)

            aop_out = aop_out.clamp(0, 1)
            dolp_out = dolp_out.clamp(0, 1)

            aop_out = aop_out.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            aop_out = (aop_out * np.pi - np.pi / 2)
            aop_gt = aop_gt.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            dolp_out = dolp_out.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            dolp_gt = dolp_gt.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            M_unpol = M_unpol.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            M_pol0 = M_pol0.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            M_pol45 = M_pol45.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            init_0 = init_0.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            init_45 = init_45.squeeze().permute(1, 2, 0).detach().cpu().numpy()

            # 计算初始化图像得到的 aop 和 dolp
            _, _, _, aop_init, dolp_init = compute_aop_dop_from_0_45_unopl(init_0, init_45, M_unpol)

            # 计算初始化的图像 PSNR 和 AoP, DoLP
            psnr_init_0 = psnr(M_pol0, init_0, data_range=1)
            psnr_init_45 = psnr(M_pol45, init_45, data_range=1)
            avg_psnr_init += (psnr_init_0 + psnr_init_45) / 2

            mae_init_aop = calculate_aop_mae(aop_init, aop_gt)
            avg_mae_init += mae_init_aop

            # 计算最终输出的图像 PSNR 和 AoP, DoLP
            I0, I45, I90, I135 = compute_Ii_from_aop_dop_unpol(aop_out, dolp_out, M_unpol)
            I0 = I0.clip(0, 1)
            I45 = I45.clip(0, 1)
            I90 = I90.clip(0, 1)
            I135 = I135.clip(0, 1)

            psnr_I0 = psnr(M_pol0, I0, data_range=1)
            psnr_I45 = psnr(M_pol45, I45, data_range=1)
            avg_psnr_0 += psnr_I0
            avg_psnr_45 += psnr_I45
            
            mae_aop = calculate_aop_mae(aop_out, aop_gt)
            avg_mae += mae_aop

            psnr_dolp = psnr(dolp_gt, dolp_out, data_range=1)
            avg_psnr_dolp += psnr_dolp

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(aop_gt.mean(-1), cmap='jet', vmin=0, vmax=1)
            plt.title("GT AoP")
            plt.colorbar()
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(aop_out.mean(-1), cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
            plt.title(f'Output AoP (MAE: {mae_aop:.2f})')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(dolp_gt.mean(-1), cmap='jet', vmin=0, vmax=1)
            plt.title('GT DoP')
            plt.colorbar()
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(dolp_out.mean(-1), cmap='jet', vmin=0, vmax=1)
            plt.title(f'Output DoP (PSNR: {psnr_dolp:.2f})')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{index}.png'))
            plt.close()

            index += 1

    avg_mae /= len(test_data_loader)
    avg_psnr_dolp /= len(test_data_loader)
    avg_psnr_0 /= len(test_data_loader)
    avg_psnr_45 /= len(test_data_loader)

    # 记录测试指标
    writer.add_scalar('Test/MAE_AoP', avg_mae, epoch)
    writer.add_scalar('Test/PSNR_DoP', avg_psnr_dolp, epoch)
    writer.add_scalar('Test/PSNR_I0', avg_psnr_0, epoch)
    writer.add_scalar('Test/PSNR_I45', avg_psnr_45, epoch)

    if avg_mae < best_mae:
        best_mae = avg_mae
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))

    logger.info(f"===> Epoch: {epoch}, MAE Init: {avg_mae_init:.4f}, Final: {avg_mae:.4f}, PSNR DoLP: {avg_psnr_dolp:.4f}, PSNR Init: {avg_psnr_init:.4f}")

    return best_mae
 
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

    train_dataset = TripleViewDataset(args, train=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info(f"Training data size: {len(train_dataset)}")

    test_dataset = TripleViewDataset(args, train=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    logger.info(f"Test data size: {len(test_dataset)}")
 
    # 加载模型
    logger.info("===> Building models")
    model = TriplePolarNet().to(device)
    model.load_state_dict(torch.load('./checkpoint/best.pth'))
 
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    l1loss = nn.L1Loss().to(device)
    best_mae = 90

    writer = SummaryWriter(log_dir=args.log_dir)

    # 开始训练
    logger.info("===> Start Training")
    for epoch in range(args.epochs):
        model.train()
        avg_pol0_loss = 0
        avg_pol45_loss = 0
        avg_aop_loss = 0
        avg_dolp_loss = 0
 
        for data in train_data_loader:
            L_pol0, M_unpol, M_pol0, M_pol45, R_pol45 = [x.to(device) for x in data]

            optimizer.zero_grad()

            _, _, _, aop_gt, dolp_gt = compute_aop_dop_from_0_45_unopl(M_pol0, M_pol45, M_unpol)

            out_0, out_45, aop_out, dolp_out, aop_conf, dolp_conf, aop_in, dolp_in = model(L_pol0, M_unpol, R_pol45)

            aop_out = aop_out * torch.pi - torch.pi / 2

            aop_error = torch.mean(torch.abs(aop_in - aop_gt), axis=1, keepdim=True)
            aop_error = aop_error / (torch.max(aop_error) + 1e-8)
            dolp_error = torch.mean(torch.abs(dolp_in - dolp_gt), axis=1, keepdim=True)
            dolp_error = dolp_error / (torch.max(dolp_error) + 1e-8)

            confidence_loss = l1loss(aop_conf, 1 - aop_error) + l1loss(dolp_conf, 1 - dolp_error)

            pol0_loss = l1loss(out_0, M_pol0)
            pol45_loss = l1loss(out_45, M_pol45)
            aop_loss = l1loss(aop_out, aop_gt) + l1loss(aop_in, aop_gt) * 10
            dolp_loss = l1loss(dolp_out, dolp_gt) + l1loss(dolp_in, dolp_gt) * 10

            loss = pol0_loss + pol45_loss + aop_loss + dolp_loss + confidence_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=1.0,
                norm_type=2,
                error_if_nonfinite=True
            )
            optimizer.step()

            avg_pol0_loss += pol0_loss.item()
            avg_pol45_loss += pol45_loss.item()
            avg_aop_loss += aop_loss.item()
            avg_dolp_loss += dolp_loss.item()

        scheduler.step()
 
        avg_pol0_loss /= len(train_data_loader)
        avg_pol45_loss /= len(train_data_loader)
        avg_aop_loss /= len(train_data_loader)
        avg_dolp_loss /= len(train_data_loader)

        logger.info(f"Pol0 Loss: {avg_pol0_loss:.5f}, Pol45 Loss: {avg_pol45_loss:.5f}, AoP Loss: {avg_aop_loss:.5f}, DoLP Loss: {avg_dolp_loss:.5f}")

        writer.add_scalar('Train/AoP_Loss', avg_aop_loss, epoch)
        writer.add_scalar('Train/DoP_Loss', avg_dolp_loss, epoch)

        if epoch % 1 == 0:
            best_mae = test_model(model, test_data_loader, epoch, device, best_mae, save_path=args.output, writer=writer)

 
if __name__ == '__main__':
    main()