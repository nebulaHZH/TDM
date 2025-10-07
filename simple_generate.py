"""
简化版图像生成脚本
专注于批量生成4张图像，无标题无边框
使用说明：确保您有训练好的权重文件(.pt格式)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

# 添加路径
sys.path.append('./diffusion')
sys.path.append('./network')

def check_environment():
    """检查环境和依赖"""
    print("正在检查环境...")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name()}")
        try:
            print(f"✓ CUDA版本: {torch.version.cuda}")
        except:
            print("✓ CUDA可用")
    else:
        print("⚠ 未检测到GPU，将使用CPU（速度较慢）")
    
    # 检查依赖文件
    required_files = [
        './diffusion/Create_diffusion.py',
        './diffusion/resampler.py', 
        './network/Diffusion_model_transformer.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("✗ 缺少必要文件:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✓ 所有必要文件存在")
    
    return True

def find_weight_file():
    """查找权重文件"""
    possible_paths = [
        "./ViTRes1.pt",
        "./checkpoints/ViTRes1.pt", 
        "./result/ViTRes1.pt",
        "C:/Pan research/Diffusion model/result/ACDC/ViTRes1.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ 找到权重文件: {path}")
            return path
    
    print("✗ 未找到权重文件")
    print("请将训练好的.pt文件放在以下位置之一:")
    for path in possible_paths:
        print(f"  - {path}")
    return None

def generate_batch():
    """生成批量图像的主函数"""
    print("="*60)
    print("医学图像扩散模型 - 批量生成工具")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        return False
    
    # 查找权重文件
    weight_file = find_weight_file()
    if not weight_file:
        return False
    
    try:
        # 导入模块
        from Create_diffusion import create_gaussian_diffusion
        from resampler import UniformSampler
        from Diffusion_model_transformer import SwinVITModel
        
        print("✓ 成功导入所需模块")
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        return False
    
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型参数
    IMAGE_SIZE = 256
    NUM_CHANNELS = 128
    
    try:
        print("正在初始化扩散模型...")
        
        # 创建扩散过程
        diffusion = create_gaussian_diffusion(
            steps=1000,
            learn_sigma=True,
            noise_schedule='linear',
            timestep_respacing=[50],  # 50步快速采样
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
        )
        
        print("正在初始化网络模型...")
        
        # 模型配置
        channel_mult = (1, 1, 2, 2, 4, 4)
        attention_ds = [IMAGE_SIZE // int(res) for res in "64,32,16,8".split(",")]
        num_heads = [4, 4, 4, 8, 16, 16]
        window_size = [[4,4], [4,4], [4,4], [8,8], [8,8], [4,4]]
        num_res_blocks = [2, 2, 1, 1, 1, 1]
        sample_kernel = ([2,2], [2,2], [2,2], [2,2], [2,2])
        
        # 创建模型
        model = SwinVITModel(
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            in_channels=1,
            model_channels=NUM_CHANNELS,
            out_channels=2,
            sample_kernel=sample_kernel,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=0,
            channel_mult=channel_mult,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=num_heads,
            window_size=window_size,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
        ).to(device)
        
        # 输出模型信息
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型创建成功，参数数量: {param_count:,}")
        
        # 加载权重
        print(f"正在加载权重: {weight_file}")
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.eval()
        print("✓ 权重加载成功")
        
        # 生成图像
        print("开始生成图像...")
        start_time = time.time()
        
        with torch.no_grad():
            generated_images = diffusion.p_sample_loop(
                model,
                (4, 1, IMAGE_SIZE, IMAGE_SIZE),  # 生成4张图像
                clip_denoised=True,
                progress=True
            )
        
        generation_time = time.time() - start_time
        print(f"✓ 图像生成完成，耗时: {generation_time:.1f}秒")
        
        # 保存图像
        print("正在保存图像...")
        
        # 创建输出目录
        output_dir = "./generated_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换图像数据
        images_np = generated_images.cpu().numpy()
        
        # 创建图像布局 - 一排4张
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for i in range(4):
            img = images_np[i, 0]  # 取第一个通道
            img = (img + 1) / 2    # 从[-1,1]转换到[0,1]
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')  # 无坐标轴
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # 调整布局，移除边框和间距
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
        
        # 保存文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"batch_generated_{timestamp}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()
        
        print(f"✓ 图像已保存到: {output_file}")
        
        # 额外保存单张图像
        for i, img_data in enumerate(images_np):
            img = img_data[0]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            
            single_file = os.path.join(output_dir, f"single_{timestamp}_{i+1}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(single_file, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
            plt.close()
        
        print(f"✓ 同时保存了4张单独图像到: {output_dir}")
        print("="*60)
        print("生成完成!")
        
        return True
        
    except Exception as e:
        print(f"✗ 生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_batch()
    if not success:
        print("\n生成失败，请检查上述错误信息")
        input("按Enter键退出...")
    else:
        print("\n生成成功!")