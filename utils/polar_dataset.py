import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import os
import os.path as osp
from utils.utils import read_file, warp_with_disp, load_camera_info


# class PolarStereoDataset(data.Dataset):
#     def __init__(self, args, train=False, train_real=False):
#         self.train_real = train_real
#         self.train = train
#         self.args = args
#         self.image_list = []        # list of image paths
#         self.disparity_list = []    # list of disparity paths
#         self.valid_list = []        # list of valid disparity paths
#         self.camera_info_list = []  # list of camera info paths
#         self.load_file()            # load the file paths
#         self.file_nums = len(self.image_list)

#     def load_file(self):
#         if self.train:
#             root = self.args.train_data
#         else:
#             root = self.args.test_data
        
#         assert os.path.exists(root)

#         file_list = os.listdir(root)
        
#         # L_pol0_list = sorted([osp.join(root, f'{name}/left/img/I0.png') for name in file_list])
#         L_pol0_list = sorted([osp.join(root, f'{name}/preprocess/L_pol0.png') for name in file_list])
#         M_unpol_list = sorted([osp.join(root, f'{name}/middle/img/S0.png') for name in file_list])
#         M_pol0_list = sorted([osp.join(root, f'{name}/middle/img/I0.png') for name in file_list])
#         M_pol45_list = sorted([osp.join(root, f'{name}/middle/img/I45.png') for name in file_list])
#         # R_pol45_list = sorted([osp.join(root, f'{name}/right/img/I45.png') for name in file_list])
#         R_pol45_list = sorted([osp.join(root, f'{name}/preprocess/R_pol45.png') for name in file_list])
#         # disp_ML_list = sorted([osp.join(root, f'{name}/disparity/dispM.npy') for name in file_list])
#         # disp_MR_list = sorted([osp.join(root, f'{name}/disparity/dispM.npy') for name in file_list])
#         disp_ML_list = sorted([osp.join(root, f'{name}/disparity/IGEV_MR.npy') for name in file_list])
#         disp_MR_list = sorted([osp.join(root, f'{name}/disparity/IGEV_MR.npy') for name in file_list])
#         valid_ML_list = sorted([osp.join(root, f'{name}/disparity/maskL.png') for name in file_list])
#         valid_MR_list = sorted([osp.join(root, f'{name}/disparity/maskR.png') for name in file_list])
#         camera_info_list = sorted([osp.join(root, f'{name}/camera_info.txt') for name in file_list])

#         assert len(L_pol0_list) == len(M_unpol_list) == len(M_pol0_list) == len(M_pol45_list) == len(R_pol45_list) == len(valid_ML_list) == len(valid_MR_list) == len(disp_ML_list) == len(disp_MR_list) == len(camera_info_list)> 0, [L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, valid_ML_list, valid_MR_list, disp_ML_list, disp_MR_list, camera_info_list]
        
#         for L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispMR, valid_ML, valid_MR, camera_info in zip(L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_ML_list, disp_MR_list, valid_ML_list, valid_MR_list, camera_info_list):
#             self.image_list += [ [L_pol0, M_unpol, M_pol0, M_pol45, R_pol45] ]
#             self.disparity_list += [ [dispMR] ]
#             # self.valid_list += [ [valid_ML, valid_MR] ]
#             # self.camera_info_list += [camera_info]

#         if self.train_real:
#             # 加载 real 数据集
#             real_root = self.args.real_data
#             assert os.path.exists(real_root)
#             real_file_list = os.listdir(real_root)

#             L_pol0_list = sorted([osp.join(real_root, f'{idx}/rgb_L0.png') for idx in real_file_list])
#             M_unpol_list = sorted([osp.join(real_root, f'{idx}/rgb_M.png') for idx in real_file_list])
#             M_pol0_list = sorted([osp.join(real_root, f'{idx}/rgb_M0.png') for idx in real_file_list])
#             M_pol45_list = sorted([osp.join(real_root, f'{idx}/rgb_M45.png') for idx in real_file_list])
#             R_pol45_list = sorted([osp.join(real_root, f'{idx}/rgb_R45.png') for idx in real_file_list])
#             disp_MR_list = sorted([osp.join(real_root, f'{idx}/IGEV_MR.npy') for idx in real_file_list])

#             assert len(L_pol0_list) == len(M_unpol_list) == len(M_pol0_list) == len(M_pol45_list) == len(R_pol45_list) == len(disp_MR_list) > 0, [L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_MR_list]

#             for L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispMR in zip(L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_MR_list):
#                 self.image_list += [ [L_pol0, M_unpol, M_pol0, M_pol45, R_pol45] ]
#                 self.disparity_list += [ [dispMR] ]
#                 # self.valid_list += [ [None, None] ]
#                 # self.camera_info_list += [None]


#     def data_augmentation(self, data_pack, patch_size=128):
#         H, W = data_pack.shape[:2]

#         y = random.randint(0, H - patch_size)
#         x = random.randint(0, W - patch_size)

#         patch = data_pack[y:y + patch_size, x:x + patch_size]
        
#         if random.random() < 0.5:
#             patch = np.ascontiguousarray(np.flipud(patch))

#         return patch


#     def __len__(self):
#         return self.file_nums

#     def __getitem__(self, index):
#         L_pol0 = read_file(self.image_list[index][0])
#         M_unpol = read_file(self.image_list[index][1])
#         M_pol0 = read_file(self.image_list[index][2])
#         M_pol45 = read_file(self.image_list[index][3])
#         R_pol45 = read_file(self.image_list[index][4])
#         dispML = read_file(self.disparity_list[index][0])           # (H, W, 1)
#         dispMR = read_file(self.disparity_list[index][1])
#         valid_ML = read_file(self.valid_list[index][0])             # (H, W)
#         valid_MR = read_file(self.valid_list[index][1])
#         # print(dispML.shape, valid_ML.shape)

#         h, w = valid_ML.shape

#         camera_info = load_camera_info(self.camera_info_list[index], h, w)

#         # L_pol0 = self.preprocess(L_pol0, M_unpol, dispML, 'L2R', valid_ML)
#         # R_pol45 = self.preprocess(R_pol45, M_unpol, dispMR, 'R2L', valid_MR)

#         L_pol0 = L_pol0[:, 100:-100, :]
#         M_unpol = M_unpol[:, 100:-100, :]
#         M_pol0 = M_pol0[:, 100:-100, :]
#         M_pol45 = M_pol45[:, 100:-100, :]
#         R_pol45 = R_pol45[:, 100:-100, :]
#         dispML = dispML[:, 100:-100, :]
#         dispMR = dispMR[:, 100:-100, :]
#         valid_ML = valid_ML[:, 100:-100]
#         valid_MR = valid_MR[:, 100:-100]
        
#         valid_ML = valid_ML[..., np.newaxis]
#         valid_MR = valid_MR[..., np.newaxis]

#         data_pack = np.concatenate([L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispML, dispMR, valid_ML, valid_MR], axis=-1)

#         if self.train:
#             data_pack = self.data_augmentation(data_pack, patch_size=self.args.patch_size)
#         else:
#             h, w, _ = data_pack.shape
#             if h > 1024 or w > 1024:
#                 data_pack = cv2.resize(data_pack, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        
#         L_pol0 = torch.from_numpy(data_pack[..., :3]).permute(2, 0, 1).float()
#         M_unpol = torch.from_numpy(data_pack[..., 3:6]).permute(2, 0, 1).float()
#         M_pol0 = torch.from_numpy(data_pack[..., 6:9]).permute(2, 0, 1).float()
#         M_pol45 = torch.from_numpy(data_pack[..., 9:12]).permute(2, 0, 1).float()
#         R_pol45 = torch.from_numpy(data_pack[..., 12:15]).permute(2, 0, 1).float()
#         dispML = torch.from_numpy(data_pack[..., 15:16]).permute(2, 0, 1).float()
#         dispMR = torch.from_numpy(data_pack[..., 16:17]).permute(2, 0, 1).float()
#         valid_ML = torch.from_numpy(data_pack[..., 17:18]).permute(2, 0, 1).float()
#         valid_MR = torch.from_numpy(data_pack[..., 18:19]).permute(2, 0, 1).float()


#         return L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispML, dispMR, valid_ML, valid_MR, camera_info


# # 用于测试网络的数据集, 网络输入 pol00, pol45, pol90, 输出 aop, dop
# class PolarDataset(data.Dataset):
#     def __init__(self, args, train=False):
#         self.train = train
#         self.args = args
#         self.image_list = []        # list of image paths
#         self.load_file()            # load the file paths
#         self.file_nums = len(self.image_list)

#     def load_file(self):
#         if self.train:
#             root = self.args.train_data
#         else:
#             root = self.args.test_data
#         print(root)
#         assert os.path.exists(root)

#         file_list = os.listdir(root)
        
#         pol00_list = sorted([osp.join(root, f'{idx}/RGB_0.png') for idx in file_list])
#         pol45_list = sorted([osp.join(root, f'{idx}/RGB_45.png') for idx in file_list])
#         pol90_list = sorted([osp.join(root, f'{idx}/RGB_90.png') for idx in file_list])

#         assert len(pol00_list) == len(pol45_list) == len(pol90_list) > 0, [pol00_list, pol45_list, pol90_list]

#         for pol00, pol45, pol90 in zip(pol00_list, pol45_list, pol90_list):
#             self.image_list += [ [pol00, pol45, pol90] ]

#     def data_augmentation(self, data_pack, patch_size=128):
#         H, W = data_pack.shape[:2]

#         # 随机裁剪
#         y = random.randint(0, H - patch_size)
#         x = random.randint(0, W - patch_size)
#         cropped_data = data_pack[y:y + patch_size, x:x + patch_size].copy()  # 确保裁剪后的数据步长为正

#         # 随机水平翻转
#         if random.random() > 0.5:
#             cropped_data = np.flip(cropped_data, axis=1).copy()  # 确保翻转后的数据步长为正

#         # 随机垂直翻转
#         if random.random() > 0.5:
#             cropped_data = np.flip(cropped_data, axis=0).copy()  # 确保翻转后的数据步长为正

#         return cropped_data


#     def __len__(self):
#         return self.file_nums

#     def __getitem__(self, index):
#         pol00 = read_file(self.image_list[index][0])
#         pol45 = read_file(self.image_list[index][1])
#         pol90 = read_file(self.image_list[index][2])

#         data_pack = np.concatenate([pol00, pol45, pol90], axis=-1)

#         if self.train:
#             data_pack = self.data_augmentation(data_pack, patch_size=self.args.patch_size)
#         # else:
#         #     data_pack = self.data_augmentation(data_pack, patch_size=384)

#         pol00 = torch.from_numpy(data_pack[..., :3]).permute(2, 0, 1).float()
#         pol45 = torch.from_numpy(data_pack[..., 3:6]).permute(2, 0, 1).float()
#         pol90 = torch.from_numpy(data_pack[..., 6:9]).permute(2, 0, 1).float()

#         s0 = pol00 + pol90
#         s1 = pol00 - pol90
#         s2 = 2 * pol45 - s0

#         aop = torch.atan2(s2, s1) / 2
#         aop = (aop + torch.pi / 2) / torch.pi
#         dop = torch.clamp(torch.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-8), 0, 1)

#         return pol00, pol45, pol90, aop, dop

# class TripleviewDataset(data.Dataset):
#     def __init__(self, args, train=False):
#         self.train = train
#         self.args = args
#         self.image_list = []        # list of image paths
#         self.disparity_list = []    # list of disparity paths
#         self.camera_info_list = []  # list of camera info paths
#         self.load_file()            # load the file paths
#         self.file_nums = len(self.image_list)

#     def load_file(self):
#         if self.train:
#             root = self.args.train_data
#         else:
#             root = self.args.test_data
        
#         assert os.path.exists(root)

#         file_list = os.listdir(root)
        
#         L_pol0_list = sorted([osp.join(root, f'{name}/left/img/I0.png') for name in file_list])
#         M_unpol_list = sorted([osp.join(root, f'{name}/middle/img/S0.png') for name in file_list])
#         M_pol0_list = sorted([osp.join(root, f'{name}/middle/img/I0.png') for name in file_list])
#         M_pol45_list = sorted([osp.join(root, f'{name}/middle/img/I45.png') for name in file_list])
#         R_pol45_list = sorted([osp.join(root, f'{name}/right/img/I45.png') for name in file_list])
#         disp_ML_list = sorted([osp.join(root, f'{name}/disparity/dispM.npy') for name in file_list])
#         camera_info_list = sorted([osp.join(root, f'{name}/camera_info.txt') for name in file_list])

#         assert len(L_pol0_list) == len(M_unpol_list) == len(M_pol0_list) == len(M_pol45_list) == len(R_pol45_list) ==  len(disp_ML_list) == len(camera_info_list)> 0, [L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_ML_list, camera_info_list]
        
#         for L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispM, camera_info in zip(L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_ML_list, camera_info_list):
#             self.image_list += [ [L_pol0, M_unpol, M_pol0, M_pol45, R_pol45] ]
#             self.disparity_list += [ [dispM] ]
#             self.camera_info_list += [camera_info]

#     def data_augmentation(self, data_pack, patch_size=128):
#         H, W = data_pack.shape[:2]

#         y = random.randint(0, H - patch_size)
#         x = random.randint(0, W - patch_size)

#         patch = data_pack[y:y + patch_size, x:x + patch_size]
        
#         if random.random() < 0.5:
#             patch = np.ascontiguousarray(np.flipud(patch))

#         return patch

#     def __len__(self):
#         return self.file_nums

#     def __getitem__(self, index):
#         L_pol0 = read_file(self.image_list[index][0])
#         M_unpol = read_file(self.image_list[index][1])
#         M_pol0 = read_file(self.image_list[index][2])
#         M_pol45 = read_file(self.image_list[index][3])
#         R_pol45 = read_file(self.image_list[index][4])
#         dispM = read_file(self.disparity_list[index][0])  # (H, W, 1)

#         camera_info = load_camera_info(self.camera_info_list[index], dispM.shape[0], dispM.shape[1])
#         fx = camera_info['fx']
#         fy = camera_info['fy']
#         cx = camera_info['cx']
#         cy = camera_info['cy']
#         baseline = camera_info['baseline']  # 相机间距

#         # 视差转深度
#         depth = fx * baseline / (dispM[..., 0] + 1e-6)  # 避免除零

#         # 计算中心视角下的 3D 坐标
#         H, W = depth.shape
#         xx, yy = np.meshgrid(np.arange(W), np.arange(H))
#         X = (xx - cx) * depth / fx
#         Y = (yy - cy) * depth / fy
#         Z = depth
#         pts_3D = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

#         # 相机中心：中间为原点
#         C_left = np.array([-baseline, 0, 0])
#         C_middle = np.array([0, 0, 0])
#         C_right = np.array([baseline, 0, 0])

#         # 计算每个像素对应的 view direction（归一化）
#         def get_view_direction(C):
#             vec = pts_3D - C.reshape(1, 1, 3)
#             norm = np.linalg.norm(vec, axis=-1, keepdims=True)
#             return vec / (norm + 1e-6)

#         view_L = get_view_direction(C_left)   # (H, W, 3)
#         view_M = get_view_direction(C_middle)
#         view_R = get_view_direction(C_right)

#         # warp图像
#         L_pol0 = warp_with_disp(L_pol0, dispM, 'L2R')
#         R_pol45 = warp_with_disp(R_pol45, dispM, 'R2L')

#         # 裁剪图像和view direction
#         # L_pol0 = L_pol0[:, 100:-100, :]
#         # M_unpol = M_unpol[:, 100:-100, :]
#         # M_pol0 = M_pol0[:, 100:-100, :]
#         # M_pol45 = M_pol45[:, 100:-100, :]
#         # R_pol45 = R_pol45[:, 100:-100, :]
#         # dispM = dispM[:, 100:-100, :]
#         # view_L = view_L[:, 100:-100, :]
#         # view_M = view_M[:, 100:-100, :]
#         # view_R = view_R[:, 100:-100, :]

#         data_pack = np.concatenate([L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispM, view_L, view_M, view_R], axis=-1)
#         if self.train:
#             data_pack = self.data_augmentation(data_pack, patch_size=self.args.patch_size)
#         else:
#             H, W, _ = data_pack.shape
#             target_size = (W // 2, H // 2)  # OpenCV 使用 (width, height) 格式
#             data_pack = cv2.resize(data_pack, target_size, interpolation=cv2.INTER_CUBIC)

#         # 分离图像和视角方向
#         L_pol0 = torch.from_numpy(data_pack[..., :3]).permute(2, 0, 1).float()
#         M_unpol = torch.from_numpy(data_pack[..., 3:6]).permute(2, 0, 1).float()
#         M_pol0 = torch.from_numpy(data_pack[..., 6:9]).permute(2, 0, 1).float()
#         M_pol45 = torch.from_numpy(data_pack[..., 9:12]).permute(2, 0, 1).float()
#         R_pol45 = torch.from_numpy(data_pack[..., 12:15]).permute(2, 0, 1).float()
#         dispM = torch.from_numpy(data_pack[..., 15:16]).permute(2, 0, 1).float()
#         view_L = torch.from_numpy(data_pack[..., 16:19]).permute(2, 0, 1).float()
#         view_M = torch.from_numpy(data_pack[..., 19:22]).permute(2, 0, 1).float()
#         view_R = torch.from_numpy(data_pack[..., 22:25]).permute(2, 0, 1).float()

#         return L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, dispM, view_L, view_M, view_R

class TripleViewDataset(data.Dataset):
    def __init__(self, args, train=False):
        self.train = train
        self.args = args
        self.image_list = []        # list of image paths
        self.disparity_list = []    # list of disparity paths
        self.load_file()            # load the file paths
        self.file_nums = len(self.image_list)

    def load_file(self):
        if self.train:
            root = self.args.train_data
        else:
            root = self.args.test_data
        
        assert os.path.exists(root)

        file_list = os.listdir(root)
        
        L_pol0_list = sorted([osp.join(root, f'{name}/left/img/I0.png') for name in file_list])
        M_unpol_list = sorted([osp.join(root, f'{name}/middle/img/S0.png') for name in file_list])
        M_pol0_list = sorted([osp.join(root, f'{name}/middle/img/I0.png') for name in file_list])
        M_pol45_list = sorted([osp.join(root, f'{name}/middle/img/I45.png') for name in file_list])
        R_pol45_list = sorted([osp.join(root, f'{name}/right/img/I45.png') for name in file_list])
        disp_list = sorted([osp.join(root, f'{name}/disparity/IGEV_MR.npy') for name in file_list])

        assert len(L_pol0_list) == len(M_unpol_list) == len(M_pol0_list) == len(M_pol45_list) == len(R_pol45_list) == len(disp_list) > 0, [L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_list]
        
        for L_pol0, M_unpol, M_pol0, M_pol45, R_pol45, disp in zip(L_pol0_list, M_unpol_list, M_pol0_list, M_pol45_list, R_pol45_list, disp_list):
            self.image_list += [ [L_pol0, M_unpol, M_pol0, M_pol45, R_pol45] ]
            self.disparity_list += [ [disp] ]

    def data_augmentation(self, data_pack, patch_size=128):
        H, W = data_pack.shape[:2]

        y = random.randint(0, H - patch_size)
        x = random.randint(0, W - patch_size)

        patch = data_pack[y:y + patch_size, x:x + patch_size]
        
        if random.random() < 0.5:
            patch = np.ascontiguousarray(np.flipud(patch))

        return patch

    def __len__(self):
        return self.file_nums

    def __getitem__(self, index):
        L_pol0 = read_file(self.image_list[index][0])
        M_unpol = read_file(self.image_list[index][1])
        M_pol0 = read_file(self.image_list[index][2])
        M_pol45 = read_file(self.image_list[index][3])
        R_pol45 = read_file(self.image_list[index][4])
        disp = read_file(self.disparity_list[index][0])
        # print(dispML.shape, valid_ML.shape)

        L_pol0 = warp_with_disp(L_pol0, disp, 'L2R', target_img=M_unpol)
        R_pol45 = warp_with_disp(R_pol45, disp, 'R2L', target_img=M_unpol)

        # L_pol0 = L_pol0[:, 100:-100, :]
        # M_unpol = M_unpol[:, 100:-100, :]
        # M_pol0 = M_pol0[:, 100:-100, :]
        # M_pol45 = M_pol45[:, 100:-100, :]
        # R_pol45 = R_pol45[:, 100:-100, :]

        data_pack = np.concatenate([L_pol0, M_unpol, M_pol0, M_pol45, R_pol45], axis=-1)

        if self.train:
            data_pack = self.data_augmentation(data_pack, patch_size=self.args.patch_size)
        else:
            h, w, _ = data_pack.shape
            if h > 2048 or w > 2448:
                data_pack = cv2.resize(data_pack, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
            # elif h > 1024 or w > 1024:
            #     # print(f"Data range before resize: {data_pack[..., :3].min()} to {data_pack[..., :3].max()}")
            #     data_pack = cv2.resize(data_pack, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
            data_pack = np.clip(data_pack, 0, 1)
                # print(f"Data range after resize: {data_pack[..., :3].min()} to {data_pack[..., :3].max()}")

        L_pol0 = torch.from_numpy(data_pack[..., :3]).permute(2, 0, 1).float()
        M_unpol = torch.from_numpy(data_pack[..., 3:6]).permute(2, 0, 1).float()
        M_pol0 = torch.from_numpy(data_pack[..., 6:9]).permute(2, 0, 1).float()
        M_pol45 = torch.from_numpy(data_pack[..., 9:12]).permute(2, 0, 1).float()
        R_pol45 = torch.from_numpy(data_pack[..., 12:15]).permute(2, 0, 1).float()

        return L_pol0, M_unpol, M_pol0, M_pol45, R_pol45