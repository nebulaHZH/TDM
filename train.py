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
PATH = path+'/TDM.pt'

def train(model, optimizer,data_loader1, loss_history):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #1: set the model to training mode
    model.train()
    total_samples = len(data_loader1.dataset)
    loss_sum = []
    total_time = 0
    
    #2: Loop the whole dataset, x1 (traindata) is the image batch
    for i, x1 in enumerate(data_loader1):

        traindata = x1.to(device)
        
        #3: extract random timestep for training
        t, weights = schedule_sampler.sample(traindata.shape[0], device)

        aa = time.time()
        
        #4: Optimize the TDM network
        optimizer.zero_grad()
        all_loss = diffusion.training_losses(model,traindata,t=t)
        loss = (all_loss["loss"] * weights).mean()
        loss.backward()
        loss_sum.append(loss.detach().cpu().numpy())
        optimizer.step()
        
        #5:print out the intermediate loss for every 100 batches
        total_time += time.time()-aa
        if i % 100 == 0:
            print('optimization time: '+ str(time.time()-aa))
            print('[' +  '{:5}'.format(i * BATCH_SIZE_TRAIN) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader1)) + '%)]  Loss: ' +
                  '{:6.7f}'.format(np.nanmean(loss_sum)))

    #6: print out the average loss for this epoch
    average_loss = np.nanmean(loss_sum) 
    loss_history.append(average_loss)
    print("Total time per sample is: "+str(total_time))
    print('Averaged loss is: '+ str(average_loss))
    return average_loss


def evaluate(model,epoch,path):
    num_sample = 1
    model.eval()
    aa = time.time()
    prediction = []
    true = []
    img = []
    loss_all = []
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
        ax.imshow(x_clean[ind,0,:,:].cpu().numpy(), cmap='gray')
    plt.savefig(path+'/epoch_'+str(epoch)+'_sample.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    # plt.show()
    data = {"img":img}
    print(str(time.time()-aa))
    scipy.io.savemat(path+ 'test_example_epoch'+str(epoch)+'.mat',data)    



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
# batch_size = 10
# t, weights = schedule_sampler.sample(batch_size, torch.device("cuda"))
# all_loss = diffusion.training_losses(model,train_loader,t=t)
# loss = (all_loss["loss"] * weights).mean()

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('parameter number is '+str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,weight_decay = 1e-4)
# path ="C:/Pan research/Diffusion model/result/ACDC/"
# PATH = path+'ViTRes1.pt' # Use your own path
# best_loss = 1
# if not os.path.exists(path):
#   os.makedirs(path) 
train_loss_history, test_loss_history = [], []

for epoch in range(0, N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time() 
    average_loss = train(model, optimizer, train_loader, train_loss_history)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

    if epoch % 20 == 0:
        evaluate(model,epoch,path)
        torch.save(model.state_dict(), PATH)


num_sample = 10
image_size = 256
x = diffusion.p_sample_loop(model,(num_sample, 1, image_size, image_size),clip_denoised=True)


for i in range(num_sample):
    plt.imshow(x[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
