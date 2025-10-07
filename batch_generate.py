"""
批量图像生成脚本
基于已训练的扩散模型权重生成一排4张医学图像
无标题，无边框，干净的图像输出
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch import nn
import glob
from natsort import natsorted

# 导入必要的模块
import sys
sys.path.append('./diffusion')
sys.path.append('./network')

from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler
from network.Diffusion_model_transformer import SwinVITModel

# 配置参数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 4  # 生成4张图像

# 模型配置参数
NUM_CHANNELS = 128
CHANNEL_MULT = (1, 1, 2, 2, 4, 4)
ATTENTION_RESOLUTIONS = "64,32,16,8"
NUM_HEADS = [4, 4, 4, 8, 16, 16]
WINDOW_SIZE = [[4, 4], [4, 4], [4, 4], [8, 8], [8, 8], [4, 4]]
NUM_RES_BLOCKS = [2, 2, 1, 1, 1, 1]
SAMPLE_KERNEL = ([2, 2], [2, 2], [2, 2], [2, 2], [2, 2])

# 扩散模型配置
DIFFUSION_STEPS = 1000
NOISE_SCHEDULE = 'linear'
TIMESTEP_RESPACING = [50]  # 使用50步加速采样
LEARN_SIGMA = True
USE_SCALE_SHIFT_NORM = True

def initialize_model():
    """初始化扩散模型和网络"""
    print("正在初始化模型...")
    
    # 创建扩散过程
    diffusion = create_gaussian_diffusion(
        steps=DIFFUSION_STEPS,
        learn_sigma=LEARN_SIGMA,
        sigma_small=False,
        noise_schedule=NOISE_SCHEDULE,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing=TIMESTEP_RESPACING,
    )
    
    # 处理注意力分辨率
    attention_ds = []
    for res in ATTENTION_RESOLUTIONS.split(","):
        attention_ds.append(IMAGE_SIZE // int(res))
    
    # 创建网络模型
    model = SwinVITModel(
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        in_channels=1,
        model_channels=NUM_CHANNELS,
        out_channels=2,
        sample_kernel=SAMPLE_KERNEL,
        num_res_blocks=NUM_RES_BLOCKS,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=CHANNEL_MULT,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=USE_SCALE_SHIFT_NORM,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(DEVICE)
    
    # 输出模型参数数量
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {pytorch_total_params:,}')
    
    return model, diffusion

def load_model_weights(model, weight_path):
    """加载预训练权重"""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    print(f"正在加载权重: {weight_path}")
    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=True)
        print("权重加载成功!")
    except Exception as e:
        print(f"权重加载失败: {e}")
        raise
    
    return model

def generate_images(model, diffusion, num_images=4):
    """生成指定数量的图像"""
    print(f"正在生成 {num_images} 张图像...")
    
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        # 生成图像
        generated_images = diffusion.p_sample_loop(
            model, 
            (num_images, 1, IMAGE_SIZE, IMAGE_SIZE),
            clip_denoised=True,
            progress=True  # 显示生成进度
        )
    
    generation_time = time.time() - start_time
    print(f"图像生成完成! 耗时: {generation_time:.2f}秒")
    
    return generated_images

def save_batch_images(images, output_path, filename="generated_batch"):
    """保存一排4张图像，无标题，无边框"""
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 转换为numpy数组并预处理
    images_np = images.cpu().numpy()
    
    # 创建一排4张图像的布局
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i in range(4):
        if i < len(images_np):
            # 取第一个通道
            img = images_np[i, 0]
            # 从[-1,1]转换到[0,1]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            
            # 显示图像
            axes[i].imshow(img, cmap='gray')
        else:
            # 如果图像不足4张，显示空白
            axes[i].imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray')
        
        # 移除坐标轴、标题和边框
        axes[i].axis('off')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # 调整子图间距，移除边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    
    # 保存图像
    output_file = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    
    print(f"批量图像已保存到: {output_file}")
    return output_file

def find_latest_checkpoint(checkpoints_dir):
    """查找最新的权重文件"""
    if not os.path.exists(checkpoints_dir):
        return None
    
    # 查找.pt文件
    pt_files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    if not pt_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(pt_files, key=os.path.getmtime)
    return latest_file

def main():
    """主函数"""
    print("="*50)
    print("医学图像扩散模型 - 批量生成脚本")
    print("="*50)
    
    # 初始化模型
    try:
        model, diffusion = initialize_model()
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    # 查找权重文件
    # 这里您需要指定权重文件的路径
    weight_paths = [
        "C:/Pan research/Diffusion model/result/ACDC/ViTRes1.pt",  # 原始路径
        "./checkpoints/ViTRes1.pt",  # 相对路径
        "./ViTRes1.pt",  # 当前目录
        "./result/ViTRes1.pt",  # result目录
    ]
    
    weight_path = None
    for path in weight_paths:
        if os.path.exists(path):
            weight_path = path
            break
    
    # 如果找不到指定路径，尝试查找最新的检查点
    if weight_path is None:
        weight_path = find_latest_checkpoint("./checkpoints")
        if weight_path is None:
            weight_path = find_latest_checkpoint(".")
    
    if weight_path is None:
        print("\n错误: 找不到权重文件!")
        print("请确保以下路径之一存在权重文件:")
        for path in weight_paths:
            print(f"  - {path}")
        print("\n提示:")
        print("1. 如果您有训练好的模型，请将.pt文件放在上述路径之一")
        print("2. 或者修改脚本中的weight_paths变量指向您的权重文件")
        print("3. 您也可以先运行训练脚本来获得权重文件")
        return
    
    # 加载权重
    try:
        model = load_model_weights(model, weight_path)
    except Exception as e:
        print(f"加载权重失败: {e}")
        return
    
    # 生成图像
    try:
        generated_images = generate_images(model, diffusion, BATCH_SIZE)
    except Exception as e:
        print(f"图像生成失败: {e}")
        return
    
    # 保存图像
    output_dir = "./generated_images"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"batch_generated_{timestamp}"
    
    try:
        output_file = save_batch_images(generated_images, output_dir, filename)
        print(f"\n成功生成并保存图像!")
        print(f"输出文件: {output_file}")
    except Exception as e:
        print(f"保存图像失败: {e}")
        return
    
    print("\n生成完成!")

if __name__ == "__main__":
    main()