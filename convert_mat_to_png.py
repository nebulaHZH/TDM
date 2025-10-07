"""
MAT文件转PNG图片工具
将现有的.mat格式文件转换为PNG图片格式
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import json
from pathlib import Path


def convert_mat_to_png(mat_file_path, output_dir=None):
    """
    将.mat文件转换为PNG图片
    
    Args:
        mat_file_path: .mat文件路径
        output_dir: 输出目录，默认为mat文件同目录
    """
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(mat_file_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取mat文件
    try:
        data = scipy.io.loadmat(mat_file_path)
        print(f"成功读取文件: {mat_file_path}")
        print(f"文件内容键值: {list(data.keys())}")
    except Exception as e:
        print(f"读取mat文件失败: {e}")
        return
    
    # 获取文件基本名称
    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    
    # 查找图像数据
    img_data = None
    data_key = None
    
    # 常见的图像数据键名
    possible_keys = ['img', 'image', 'data', 'images', 'train_loss_history']
    
    for key in possible_keys:
        if key in data and not key.startswith('__'):
            img_data = data[key]
            data_key = key
            break
    
    # 如果没找到，尝试第一个非系统键
    if img_data is None:
        for key, value in data.items():
            if not key.startswith('__'):
                img_data = value
                data_key = key
                break
    
    if img_data is None:
        print("未找到有效的图像数据")
        return
    
    print(f"找到数据键: {data_key}, 数据形状: {img_data.shape}")
    
    # 根据数据类型进行不同处理
    if data_key == 'train_loss_history' or (len(img_data.shape) == 1):
        # 损失历史数据，绘制曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(img_data.flatten(), linewidth=2)
        plt.title(f'Loss History - {base_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'{base_name}_loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"损失曲线图已保存: {plot_path}")
        
        # 同时保存为JSON
        loss_data = {
            "loss_history": img_data.flatten().tolist(),
            "total_epochs": len(img_data.flatten()),
            "final_loss": float(img_data.flatten()[-1]) if len(img_data.flatten()) > 0 else None,
            "min_loss": float(np.min(img_data.flatten())) if len(img_data.flatten()) > 0 else None,
            "source_file": mat_file_path
        }
        
        json_path = os.path.join(output_dir, f'{base_name}_loss_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=2, ensure_ascii=False)
        print(f"损失数据JSON已保存: {json_path}")
        
    else:
        # 图像数据处理
        if len(img_data.shape) >= 3:
            # 多维图像数据
            num_images = img_data.shape[0] if len(img_data.shape) >= 4 else 1
            
            for i in range(min(num_images, 10)):  # 最多保存10张图片
                if len(img_data.shape) == 4:
                    # (batch, channels, height, width) 或 (batch, height, width, channels)
                    if img_data.shape[1] == 1 or img_data.shape[-1] == 1:
                        # 单通道图像
                        if img_data.shape[1] == 1:
                            img = img_data[i, 0]  # (height, width)
                        else:
                            img = img_data[i, :, :, 0]  # (height, width)
                    else:
                        img = img_data[i, 0] if img_data.shape[1] < img_data.shape[-1] else img_data[i, :, :, 0]
                elif len(img_data.shape) == 3:
                    img = img_data[i] if img_data.shape[0] <= 10 else img_data[0]
                else:
                    img = img_data
                
                # 数据范围处理
                if img.max() <= 1.0 and img.min() >= -1.0:
                    # 可能是[-1,1]或[0,1]范围
                    if img.min() < 0:
                        img = (img + 1) / 2  # [-1,1] -> [0,1]
                elif img.max() > 1.0:
                    # 归一化到[0,1]
                    img = (img - img.min()) / (img.max() - img.min())
                
                # 确保数据在有效范围内
                img = np.clip(img, 0, 1)
                
                # 保存图片
                plt.figure(figsize=(8, 8))
                plt.imshow(img, cmap='gray')
                plt.title(f'{base_name} - Image {i}')
                plt.axis('off')
                plt.tight_layout()
                
                img_path = os.path.join(output_dir, f'{base_name}_image_{i}.png')
                plt.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"图片已保存: {img_path}")
        
        else:
            # 2D图像
            img = img_data
            if img.max() <= 1.0 and img.min() >= -1.0:
                if img.min() < 0:
                    img = (img + 1) / 2
            elif img.max() > 1.0:
                img = (img - img.min()) / (img.max() - img.min())
            
            img = np.clip(img, 0, 1)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            plt.title(f'{base_name}')
            plt.axis('off')
            plt.tight_layout()
            
            img_path = os.path.join(output_dir, f'{base_name}.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"图片已保存: {img_path}")


def convert_all_mat_files(directory):
    """
    转换目录中所有的.mat文件
    
    Args:
        directory: 包含.mat文件的目录路径
    """
    mat_files = list(Path(directory).glob("*.mat"))
    
    if not mat_files:
        print(f"在目录 {directory} 中未找到.mat文件")
        return
    
    print(f"找到 {len(mat_files)} 个.mat文件")
    
    for mat_file in mat_files:
        print(f"\n处理文件: {mat_file}")
        convert_mat_to_png(str(mat_file))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("1. 转换单个文件: python convert_mat_to_png.py <mat_file_path>")
        print("2. 转换目录中所有文件: python convert_mat_to_png.py <directory_path>")
        print("\n例如:")
        print("python convert_mat_to_png.py result/ACDC/test_example_epoch0.mat")
        print("python convert_mat_to_png.py result/ACDC/")
    else:
        path = sys.argv[1]
        
        if os.path.isfile(path) and path.endswith('.mat'):
            # 单个文件
            convert_mat_to_png(path)
        elif os.path.isdir(path):
            # 目录
            convert_all_mat_files(path)
        else:
            print(f"错误: {path} 不是有效的.mat文件或目录")