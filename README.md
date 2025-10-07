
# 2D医学图像去噪扩散概率模型
**这是论文"[基于Transformer的去噪扩散概率模型用于2D医学图像合成](https://iopscience.iop.org/article/10.1088/1361-6560/acca5c/meta)"的代码仓库。**

代码基于 [image-guided diffusion](https://github.com/openai/guided-diffusion)、[SwinUnet](https://github.com/HuCaoFighting/Swin-Unet) 和 [Monai](https://monai.io/) 创建

版本更新 1.1：
通过修改变分界损失代码（遵循image-guided diffusion），我们能够使用1000个训练时间步和50个推理时间步（而不是论文中的4000个训练时间步和500个推理时间步），并稳定训练过程以生成精美的图像！**也许这对2D合成不是很重要，但对3D合成至关重要！！**
详细信息请参见我们的另一篇论文"[使用基于3D Transformer的去噪扩散模型从MRI生成合成CT](https://arxiv.org/abs/2305.19467)"

# 依赖包

项目依赖包已列在`requirements.txt`文件中。

## Python环境要求
- Python 3.8 或更高版本
- CUDA 11.1 或更高版本（用于GPU加速）

## 安装方法

### 方法1：使用自动安装脚本（最简单）
我们提供了一个自动化的环境设置脚本：
```bash
python setup_env.py
```
该脚本将：
- 检查Python版本是否满足要求
- 可选创建虚拟环境
- 自动安装所有依赖包
- 验证安装结果
- 检测GPU可用性

### 方法2：手动使用pip安装
```bash
# 创建虚拟环境（推荐）
python -m venv tdm_env
# 激活虚拟环境
# Windows:
tdm_env\Scripts\activate
# Linux/Mac:
source tdm_env/bin/activate

# 安装依赖包
pip install -r requirements.txt
```

### 方法3：使用conda安装（保留选项）
如果你更喜欢使用conda，也可以使用原有的环境配置：
```bash
conda env create -f test_env.yaml
conda activate DL
```

## 快速开始

环境安装完成后，您可以按照以下步骤快速开始：

1. **启动Jupyter Notebook**
   ```bash
   jupyter notebook "TDM main.ipynb"
   ```

2. **或者直接在Python中使用**
   ```python
   # 导入必要的模块
   from diffusion.Create_diffusion import *
   from network.Diffusion_model_transformer import *
   
   # 按照下面的使用方法章节进行操作
   ```

3. **验证GPU可用性**
   ```python
   import torch
   print(f"CUDA可用: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU数量: {torch.cuda.device_count()}")
       print(f"当前GPU: {torch.cuda.get_device_name(0)}")
   ```


# 使用方法

使用方法在jupyter notebook文件TDM main.ipynb中。包括如何构建扩散过程、如何构建网络、如何调用扩散过程进行训练以及采样新的合成图像。不过，我们在下面给出简单示例：

**创建扩散过程**
```
from diffusion.Create_diffusion import *
from diffusion.resampler import *

diffusion = create_gaussian_diffusion(
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule='linear',
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=True,
    rescale_learned_sigmas=True,
    timestep_respacing=[250],
)
schedule_sampler = UniformSampler(diffusion)
```

**创建网络**
```
attention_resolutions="64,32,16,8"
attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))

image_size = 256
from network.Diffusion_model_transformer import *
model = SwinVITModel(
        image_size=(image_size,image_size),
        in_channels=1,
        model_channels=128,
        out_channels=2,
        sample_kernel=([2,2],[2,2],[2,2],[2,2],[2,2]),
        num_res_blocks=[2,2,1,1,1,1],
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_classes=None,
        num_heads=[4,4,4,8,16,16],
        window_size = [[4,4],[4,4],[4,4],[8,8],[8,8],[4,4]],
        use_scale_shift_norm=True,
        resblock_updown=False,
    )
```

**训练扩散模型**
```
batch_size = 10
t, weights = schedule_sampler.sample(batch_size, device)
all_loss = diffusion.training_losses(model,traindata,t=t)
loss = (all_loss["loss"] * weights).mean()
```

**生成新的合成图像**
```
num_sample = 10
image_size = 256
x = diffusion.p_sample_loop(model,(num_sample, 1, image_size, image_size),clip_denoised=True)
```


# 可视化示例

![image_1](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/3a814bd3-1107-4d23-b295-9088530754d8)
![image_2](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/cfb2d2c8-f611-497c-93ff-99b7f1ad27a7)
![image_3](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e183a0fd-dcd0-4b1a-8c5f-b861c05b4b9f)
![image_27](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/6c43ef4a-6903-4a72-9363-421fd5c264b4)

![image_4](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/877cfa01-d1b9-4728-ad14-58ac41a3ef9d)
![image_402](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/8c44d75c-7a9b-4de6-ba01-bae18b5dfe2c)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/955b5c65-e4a6-4e08-a870-bd59ad0682bd)
![image_69](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/48f9413e-e630-41e3-9edf-57ad3887822c)

![image_1](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e19f614d-3441-407c-bbbb-e76d2cda6fa3)
![image_5](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/959e8a26-4925-4799-a2b7-a4f8f2e15e43)
![image_7](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/1b4dffb9-a324-4e4b-b76a-1f18648bdb37)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e1300ad7-2a5a-42ea-8980-8f37427ca7b1)

![image_8](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/0ac4a0f3-ce65-4280-8442-ac8f2e000c4d)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/32a0d462-ebbe-465e-9ac2-e8c5d8f75e07)
![image_4](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/f64e4cc0-155d-4b17-b6aa-68d2362be7ec)
![image_46](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/43a3b4ce-7469-4f18-8dd7-87689df410b7)


