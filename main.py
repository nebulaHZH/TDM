"""
2D Medical Denoising Diffusion Probabilistic Model (TDM)
基于Transformer的去噪扩散概率模型，用于2D医学图像合成
"""

from typing import Any
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io
import os
import sys
import json
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from random import randint
import random
import time
import re
from timm.models.layers import DropPath
from einops import rearrange
from scipy import ndimage
from skimage import io
from skimage import transform
from natsort import natsorted
from skimage.transform import rotate, AffineTransform
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,    
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    ResizeWithPadOrCropd
)

# The diffusion module adapted from https://github.com/openai/guided-diffusion
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from diffusion.normal_diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil


def print_system_info():
    """打印系统信息和GPU状态"""
    print("=== 系统信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"CPU核心数: {psutil.cpu_count()}")
    print(f"系统内存: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print("=" * 50)



class CustomDataset(Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path
        # 只搜索常见图像格式
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        file_list = []
        
        for ext in extensions:
            file_list.extend(glob.glob(os.path.join(self.imgs_path, ext)))
        
        file_list = natsorted(file_list, key=lambda y: y.lower())
        self.data = []
        
        for img_path in file_list:
            if os.path.exists(img_path):
                class_name = os.path.basename(img_path)
                self.data.append([img_path, class_name])
        
        print(f"找到 {len(self.data)} 个图像文件")
        
        # 简化的变换链，移除可能导致问题的变换
        self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),  # 使用默认读取器
                    EnsureChannelFirstd(keys=["image"]),
                    ScaleIntensityd(keys=["image"], minv=-1, maxv=1.0),
                    ResizeWithPadOrCropd(
                        keys=["image"],
                        spatial_size=(256, 256),
                        constant_values=-1,
                    ),
                    ToTensord(keys=["image"]),
                ]
            )
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        cao = {"image": img_path}
        affined_data_dict = self.train_transforms(cao)                    
        img_tensor = affined_data_dict['image'].to(torch.float)
        
        return img_tensor


def create_diffusion_model(diffusion_steps=1000, learn_sigma=True, timestep_respacing=[50]):
    """创建扩散模型"""
    # 固定参数
    sigma_small = False
    class_cond = False
    noise_schedule = 'linear'
    use_kl = False
    predict_xstart = False
    rescale_timesteps = True
    rescale_learned_sigmas = True
    use_checkpoint = False

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    
    return diffusion


def create_network_model(device, image_size=256):
    """创建网络模型"""
    from network.Diffusion_model_transformer import SwinVITModel
    
    # 网络参数
    num_channels = 128
    channel_mult = (1, 1, 2, 2, 4, 4)
    attention_resolutions = "64,32,16,8"
    num_heads = [4, 4, 4, 8, 16, 16]
    window_size = [[4, 4], [4, 4], [4, 4], [8, 8], [8, 8], [4, 4]]
    num_res_blocks = [2, 2, 1, 1, 1, 1]
    sample_kernel = ([2, 2], [2, 2], [2, 2], [2, 2], [2, 2]),

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))
    
    class_cond = False
    use_scale_shift_norm = True
    resblock_updown = False

    model = SwinVITModel(
        image_size=(image_size, image_size),
        in_channels=1,
        model_channels=num_channels,
        out_channels=2,
        sample_kernel=sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size=window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=False,
    ).to(device)
    
    return model


def train_epoch(model, optimizer, data_loader, diffusion, schedule_sampler, device, epoch, loss_history):
    """训练一个epoch"""
    model.train()
    total_samples = len(data_loader.dataset)
    loss_sum = []
    total_time = 0
    
    # 使用tqdm显示batch级别进度条
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), 
                desc=f'Epoch {epoch}', ncols=100, miniters=1)
    
    for i, x1 in pbar:
        traindata = x1.to(device)
        
        # 提取随机时间步进行训练
        t, weights = schedule_sampler.sample(traindata.shape[0], device)

        aa = time.time()
        
        # 优化TDM网络
        optimizer.zero_grad()
        all_loss = diffusion.training_losses(model, traindata, t=t)
        loss = (all_loss["loss"] * weights).mean()
        loss.backward()
        loss_sum.append(loss.detach().cpu().numpy())
        optimizer.step()
        
        batch_time = time.time() - aa
        total_time += batch_time
        
        # 更新进度条
        current_loss = np.nanmean(loss_sum[-10:])  # 显示最近10个batch的平均损失
        pbar.set_postfix({
            'loss': f'{current_loss:.6f}',
            'time': f'{batch_time:.3f}s'
        })
        
        # 强制刷新输出
        sys.stdout.flush()

    # 计算平均损失
    average_loss = np.nanmean(loss_sum) 
    loss_history.append(average_loss)
    
    print(f"Epoch {epoch} - 平均损失: {average_loss:.6f}, 总时间: {total_time:.2f}s")
    return average_loss


def evaluate_model(model, diffusion, epoch, save_path, device, image_size=256, num_sample=4):
    """评估模型并生成样本"""
    model.eval()
    aa = time.time()
    
    with torch.no_grad():
        print(f"正在生成 {num_sample} 个样本...")
        # 使用tqdm显示采样进度
        x_clean = diffusion.p_sample_loop(
            model, 
            (num_sample, 1, image_size, image_size), 
            clip_denoised=True,
            progress=True
        )
    
    generation_time = time.time() - aa
    print(f'Epoch {epoch} 生成完成，耗时: {generation_time:.2f}s')
    
    # 显示生成的图像 - 修复旋转问题并拼接显示
    plt.rcParams['figure.figsize'] = [20, 5]
    
    # 创建拼接图像
    images_to_concat: list[Any] = []
    for ind in range(num_sample):
        # 修复图像旋转问题：转置图像矩阵
        img = x_clean[ind, 0, :, :].cpu().numpy().T  # 添加.T来修正旋转
        images_to_concat.append(img)
    
    # 水平拼接所有图像
    concatenated_image = np.concatenate(images_to_concat, axis=1)
    
    # 显示拼接后的图像，无边框无标题
    plt.figure()
    plt.imshow(concatenated_image, cmap='gray')
    plt.axis('off')  # 去掉坐标轴和边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉所有边距
    
    plt.tight_layout()
    if epoch % 10 == 0:
        # 保存生成样本图像为PNG格式
        sample_save_path = os.path.join(save_path, f'generated_samples_epoch{epoch}.png')
        plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
        print(f"生成样本已保存到: {sample_save_path}")
    
    # plt.show()

def save_checkpoint(model, optimizer, epoch, loss, save_path, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(save_path, 'best_model.pt'))
        print(f"保存最佳模型 (Epoch {epoch}, Loss: {loss:.6f})")
    
    # 定期保存检查点
    if epoch % 50 == 0:
        torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch}.pt'))


def run_diagnostic():
    """运行训练前诊断"""
    print("=== 训练前诊断 ===")
    
    # 打印系统信息
    print_system_info()
    
    # 检查数据路径
    data_path = 'D:\\0-nebula\\dataset\\ixi_paried\\t2_resized'
    if os.path.exists(data_path):
        print(f"✓ 数据路径存在: {data_path}")
        # 检查数据集大小
        dataset = CustomDataset(data_path)
        print(f"✓ 数据集大小: {len(dataset)} 个样本")
    else:
        print(f"✗ 数据路径不存在: {data_path}")
        return False
    
    # 检查batch size配置
    batch_size = 1  # 优化后的batch size
    print(f"✓ Batch size: {batch_size}")
    
    # 测试单个batch处理时间
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_network_model(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数数量: {pytorch_total_params:,}")
    
    
    print("=" * 50)
    return True


def main():
    """主函数"""
    # 运行诊断
    if not run_diagnostic():
        print("诊断失败，请检查配置")
        return
    
    # 超参数配置
    BATCH_SIZE_TRAIN = 1  # 优化后的batch size
    image_size = 256

    
    # 扩散模型参数
    diffusion_steps = 1000
    learn_sigma = True
    timestep_respacing = [50]
    
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建扩散模型
    diffusion = create_diffusion_model(diffusion_steps, learn_sigma, timestep_respacing)
    schedule_sampler = UniformSampler(diffusion)
    
    # 创建网络模型
    model = create_network_model(device, image_size)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {pytorch_total_params:,}')
    
    # 优化器配置
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    
    # 数据加载器配置
    training_set = CustomDataset('D:\\0-nebula\\dataset\\ixi_paried\\t2_resized')
    params = {
        'batch_size': BATCH_SIZE_TRAIN,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': False,
        'num_workers': 4,  # 优化数据加载
        'persistent_workers': True,
        'prefetch_factor': 2
    }
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    
    # 训练配置
    N_EPOCHS = 500
    save_path = "checkpoints/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    train_loss_history = []
    best_loss = float('inf')
    
    print("开始训练...")
    print("=" * 50)
    
    # 训练循环 - 使用epoch级别进度条
    epoch_pbar = tqdm(range(N_EPOCHS), desc='训练进度', ncols=100)
    
    for epoch in epoch_pbar:
        start_time = time.time()
        
        # 训练一个epoch
        average_loss = train_epoch(model, optimizer, train_loader, diffusion, 
                                 schedule_sampler, device, epoch, train_loss_history)
        
        epoch_time = time.time() - start_time
        
        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'loss': f'{average_loss:.6f}',
            'time': f'{epoch_time:.1f}s'
        })
        
        # 保存最佳模型
        if average_loss < best_loss:
            best_loss = average_loss
            save_checkpoint(model, optimizer, epoch, average_loss, save_path, is_best=True)
        
        # 定期保存检查点
        save_checkpoint(model, optimizer, epoch, average_loss, save_path)
        
        # 每5个epoch进行评估
        if epoch % 10 == 0:
            evaluate_model(model, diffusion, epoch, save_path, device)
        
        
        # 强制刷新输出
        sys.stdout.flush()
    
    print("\n训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    
    # 保存损失历史 - 改为PNG图片和JSON日志格式
    # 1. 保存为PNG图片
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', linewidth=2)
    plt.title('Training Loss History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0)
    loss_plot_path = os.path.join(save_path, 'training_loss_history.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线图已保存到: {loss_plot_path}")
    
    # 2. 保存为JSON日志格式
    loss_data = {
        "train_loss_history": train_loss_history,
        "total_epochs": len(train_loss_history),
        "final_loss": train_loss_history[-1] if train_loss_history else None,
        "best_loss": min(train_loss_history) if train_loss_history else None,
        "training_completed": True
    }
    json_path = os.path.join(save_path, 'training_history.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(loss_data, f, indent=2, ensure_ascii=False)
    print(f"训练历史日志已保存到: {json_path}")


if __name__ == "__main__":
    main()