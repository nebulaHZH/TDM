#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境设置辅助脚本
用于检查和安装2D医学图像去噪扩散概率模型所需的Python包
"""

import subprocess
import sys  
import os
import platform

def check_python_version():
    """检查Python版本是否满足要求"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本不满足要求。当前版本: {version.major}.{version.minor}.{version.micro}")
        print("请安装Python 3.8或更高版本")
        return False

def check_gpu_available():
    """检查是否有可用的GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ 检测到GPU: {gpu_name} (共{gpu_count}个GPU)")
            return True
        else:
            print("⚠️  未检测到可用的GPU，将使用CPU进行计算")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检测GPU")
        return False

def install_requirements():
    """安装requirements.txt中的依赖包"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ 未找到{requirements_file}文件")
        return False
    
    print("开始安装依赖包...")
    try:
        # 升级pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装依赖包
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装依赖包时出错: {e}")
        return False

def create_virtual_environment():
    """创建虚拟环境"""
    venv_name = "tdm_env"
    
    if os.path.exists(venv_name):
        print(f"⚠️  虚拟环境 '{venv_name}' 已存在")
        return True
    
    try:
        print(f"正在创建虚拟环境: {venv_name}")
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print(f"✅ 虚拟环境 '{venv_name}' 创建成功")
        
        # 提示如何激活虚拟环境
        if platform.system() == "Windows":
            activate_cmd = f"{venv_name}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_name}/bin/activate"
        
        print(f"请运行以下命令激活虚拟环境:")
        print(f"  {activate_cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 创建虚拟环境时出错: {e}")
        return False

def verify_installation():
    """验证关键包是否安装成功"""
    key_packages = [
        "torch", "torchvision", "monai", "transformers", 
        "diffusers", "numpy", "PIL", "matplotlib"
    ]
    
    print("\n验证关键包安装状态...")
    all_installed = True
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装或安装失败")
            all_installed = False
    
    return all_installed

def main():
    """主函数"""
    print("=" * 60)
    print("2D医学图像去噪扩散概率模型 - 环境设置脚本")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 询问用户是否要创建虚拟环境
    while True:
        choice = input("\n是否要创建新的虚拟环境? (y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            if not create_virtual_environment():
                return
            print("\n请先激活虚拟环境，然后重新运行此脚本进行包安装。")
            return
        elif choice in ['n', 'no', '否']:
            break
        else:
            print("请输入 y 或 n")
    
    # 安装依赖包
    if not install_requirements():
        return
    
    # 验证安装
    if verify_installation():
        print("\n🎉 环境设置完成！所有关键包均已正确安装。")
        
        # 检查GPU
        check_gpu_available()
        
        print("\n现在您可以运行以下命令开始使用:")
        print("  jupyter notebook \"TDM main.ipynb\"")
    else:
        print("\n⚠️  部分包安装失败，请检查错误信息并手动安装缺失的包。")

if __name__ == "__main__":
    main()