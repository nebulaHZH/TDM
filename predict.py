import os
import time
import scipy
import torch
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from main import CustomDataset
from matplotlib import pyplot as plt


BATCH_SIZE_TRAIN = 1
N_EPOCHS = 400
path ="./checkpoints"
output_path = "./output"
PATH = path+'/TDM.pt'

def evaluate(model,epoch,path):
    num_sample = 1
    model.eval()
    aa = time.time()
    img = []
    with torch.no_grad():
        x_clean = diffusion.p_sample_loop(model,(num_sample, 1, image_size, image_size),clip_denoised=True)
        img.append(x_clean.cpu().numpy())
    print('Generate for the epoch #'+str(epoch)+' result:')
    plt.rcParams['figure.figsize'] = [20, 20]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for ind in range(num_sample):
        img_data = x_clean[ind,0,:,:].cpu().numpy()
        img_data_rotated = np.rot90(img_data, k=-1)  # k=-1 表示顺时针旋转90度
        ax.imshow(img_data_rotated, cmap='gray')
    plt.savefig(path+'/'+str(epoch)+'_sample.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


params = {
    "batch_size": BATCH_SIZE_TRAIN,
    'pin_memory': True,
    "shuffle": True,
    'drop_last': False
}
data_path = 'E:\\nebula\\t2_resized'
dataset = CustomDataset(data_path)
train_loader = torch.utils.data.DataLoader(dataset, **params)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load(PATH), strict=False)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('parameter number is '+str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,weight_decay = 1e-4)

num_sample = 200
image_size = 256

for epoch in range(0, num_sample):
    evaluate(model,epoch,output_path)
