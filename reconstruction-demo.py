import torch
print(torch.cuda.is_available())  # 检查CUDA是否可用
print(torch.cuda.current_device())  # 当前使用的CUDA设备
print(torch.cuda.device_count())  # 可用的GPU数量

from torchsummary import summary
from src.models.video_cav_mae import VideoCAVMAE
from src.dataloader import *
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from scipy.ndimage import gaussian_filter

from dataset.dataset_ft import *




def draw_spectrum(spectrum_np):
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrum_np.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency bins')
    plt.xlabel('Time frames')
    plt.title('Spectrogram')
    plt.show()


def save_spectrum(spectrum_np, save_path=None):
    """
    绘制频谱图，并选择性保存图片。

    参数：
    - spectrum_np: numpy 数组，表示频谱数据。
    - save_path: str，可选，图片保存的完整路径（包括文件名和扩展名）。

    返回：
    - None
    """
    blurred_spectrum = gaussian_filter(spectrum_np, sigma=4) #
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrum_np.T, aspect='auto', origin='lower', cmap='viridis')
    #plt.imshow(blurred_spectrum.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency bins')
    plt.xlabel('Time frames')
    plt.title('Spectrogram')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Spectrogram saved to {save_path}")

    plt.show()


def draw_picture(images):
    images = images - images.min()
    images = images / images.max()

    # 使用 torchvision.utils.make_grid 创建一个网格
    grid = torchvision.utils.make_grid(images)

    # 将网格张量转换为numpy数组
    grid_np = grid.numpy().transpose((1, 2, 0))

    # 展示图片
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def save_picture(images, save_path=None):
    """
    绘制图像网格，并选择性保存图片。

    参数：
    - images: tensor，图像数据。
    - save_path: str，可选，图片保存的完整路径（包括文件名和扩展名）。

    返回：
    - None
    """
    print(images.min(),images.max())
    images = images - images.min()  # 归一化
    print(images.min(),images.max())
    images = images / images.max()

    #images=images+7.74
    #images=images/11.08


    # 使用 torchvision.utils.make_grid 创建一个网格
    grid = torchvision.utils.make_grid(images)

    # 将网格张量转换为numpy数组
    grid_np = grid.numpy().transpose((1, 2, 0))

    # 展示图片
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')  # 不显示坐标轴

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Image grid saved to {save_path}")

    plt.show()



dataset_mean=-5.081
dataset_std=4.4849
im_res = 224
data_val = "./demo/demo.csv"

#val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0,'mode':'eval','mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': im_res}


#val_loader = DataLoader(VideoAudioDataset(data_val, val_audio_conf, stage=1), batch_size=1, shuffle=True, pin_memory=True, drop_last=False)


video_config = {'image_size': 224}
audio_config = {'num_mel_bins': 128, 'target_length': 1024}
val_loader = DataLoader(ft_dataset_validate(data_val, video_config, audio_config), batch_size=1, shuffle=True,
                         pin_memory=True, drop_last=False)

for i, (audio, frames, labels) in enumerate(val_loader):
    print(audio.shape)
    print(frames.shape)
    print(labels)
    break

#================================================================
# Plt original spectrum
spectrum_np = audio.squeeze(0).numpy()  # 变为 [1024, 128]
save_spectrum(spectrum_np, "./demo/original_spectrum.png")

# Plt original frames
frames_ = frames.squeeze(0).transpose(0,1)
save_picture(frames_, "./demo/original_frames.png")


#================================================================
# Load Reconstruction Model

from torch.nn.parallel import DataParallel

# 设置 device_ids，主设备为 cuda:4
# 设置空闲GPU为默认设备
torch.cuda.set_device('cuda:0')
device = torch.device('cuda:0')

print(f"Using GPU with ID: {torch.cuda.current_device()}")

video_cavmae = VideoCAVMAE()
video_cavmae = torch.nn.DataParallel(video_cavmae, device_ids=[0]).to(device)
model = torch.load('./weights/pre-trained/stage-2.pth', map_location="cpu")

missing, unexpected = video_cavmae.load_state_dict(model)

assert len(missing) == 0 and len(unexpected) == 0

print("Model loaded, the data flow is:")

# summary(video_cavmae, input_size=[(1024, 128), (3, 16, 224, 224)])  # 模型测试


audio = audio.to(device)
print(audio.shape)
frames = frames.to(device)
print(frames.shape)


total_loss, nce_loss, c_acc, rec_loss_a, rec_loss_v, audio_recon, video_recon = video_cavmae(audio, frames)
audio_recon, video_recon = audio_recon.cpu().detach(), video_recon.cpu().detach()
c_acc.cpu().detach()

#Plt reconstructed spectrum
spectrum_np = audio_recon.squeeze(0).squeeze(0).transpose(0,1).numpy()  # 变为 [1024, 128]
save_spectrum(spectrum_np, "./demo/reconstructed_spectrum.png")

# Plt reconstructed images
frames = video_recon.squeeze(0).transpose(0,1)

save_picture(frames, "./demo/reconstructed_frames.png")
print("OK")