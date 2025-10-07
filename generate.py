



# 超参数配置
from diffusion.respace import SpacedDiffusion
import os
import torch
import numpy as np
from torchvision.utils import save_image
from diffusion import GaussianDiffusion
from diffusion.resampler import UniformSampler
from main import CustomDataset, create_diffusion_model, create_network_model, evaluate_model
from matplotlib import pyplot as plt
import torch as th

def sample_and_save_images(diffusion:SpacedDiffusion, model, num_images=4, image_size=256, save_path="outputs/samples.png", device="cuda"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    # 生成起始噪声
    shape = (num_images, 1, image_size, image_size)  # 灰度图像只有 1 个通道
    start_noise = th.randn(shape, device=device)

    # 使用 DDIM 采样方法进行确定性生成
    model.eval()
    
    with th.no_grad():
        x_clean = diffusion.p_sample_loop(
            model,
            shape,
            noise=start_noise,
            device=device,
            clip_denoised=True,
            progress=True
        )

   
    plt.rcParams['figure.figsize'] = [20, 5]

    # 获取原来的图像并拼接
    images:list[torch.Tensor] = []
    for ind in range(num_images):
        # 图像转置
        img = x_clean[ind,0,:,:].cpu().numpy().T
        images.append(img)
        
    # 水平拼接所有的图像
    all_in_one_images = np.concatenate(images, axis=1)

    for i in range(all_in_one_images.shape[0]):
        save_image(all_in_one_images[i], save_path=f"{save_path[:-4]}_{i+1:03d}.png")

    print(f"生成样本已经保存到：{save_path}")


BATCH_SIZE_TRAIN = 1  # 优化后的batch size
image_size = 256
img_size = (image_size, image_size)
spacing = (1, 1)
channels = 1
# 扩散模型参数
diffusion_steps = 1000
learn_sigma = True
timestep_respacing = 'ddim50'

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建扩散模型
diffusion: SpacedDiffusion = create_diffusion_model(diffusion_steps, learn_sigma, timestep_respacing)
schedule_sampler = UniformSampler(diffusion)

# 创建网络模型
model = create_network_model(device, image_size)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'模型参数数量: {pytorch_total_params:,}')

# 优化器配置
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

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

# 采样配置
nums = 100
save_path = "outputs/"
if os.path.exists(save_path) == False:
    os.mkdir(save_path)
# 加载检查点
checkpoint = torch.load('checkpoints/checkpoint_epoch_450.pt',weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
#批量生成图像
for num in range(nums):
    sample_and_save_images(
        diffusion=diffusion,
        num_images=1,
        model=model,
        save_path=save_path+f'final_denoised_pred_xstart_{num+1:03d}.png',
    )

print(f"所有 {nums} 张完全去噪图像生成完成！")

# 额外诊断信息
print(f"实际扩散步数: {diffusion.num_timesteps}")
print(f"使用的时间步重采样: {timestep_respacing}")