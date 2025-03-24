import torch
import os
import torchaudio
torchaudio.set_audio_backend("librosa") # 若被弃用，用pip install soundfile代替
import ffmpeg
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import PIL
import csv
import random
from PIL import ImageEnhance
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录（即 preprocess_pipeline 所在的目录）
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import preprocess_pipeline.pipeline as pipeline

class ft_dataset_validate(Dataset):
    '''
    用于微调阶段的验证数据集
    '''    
    def __init__(self, csv_file, video_config, audio_config):
        '''
        构造方法: 
        csv_file: 视频数据集文件夹的路径

        video_config: 存储视频处理配置参数的字典,应包含:
        'image_size': 224
        'image_norm': False
        'image_augment': False
        'image_mean'
        'image_std'
         


        audio_config: 存储音频处理配置参数的字典,应包含:
        'num_mel_bins': 128,            # Mel频率倒谱系数(MFCC)中的Mel频带数量,用于音频特征提取
        'target_length': 1024,          # 音频目标长度
        'audio_norm': False
        'noise_augment': False
        'audio_mean'
        'audio_std'
        '''

        # 从csv文件中读取数据集信息
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) # 跳过 CSV 文件的第一行，即表头: video_name, label
            for row in reader:
                self.data.append(row)
        self.num_samples = len(self.data)
        print(f'Dataset {csv_file} contains {self.num_samples} samples.')
        
        # 配置视频处理参数
        self.video_config = video_config
        self.image_size = self.video_config.get('image_size', 224)
        self.image_norm = self.video_config.get('image_norm', False)
        self.image_augment = self.video_config.get('image_augment', False)
        
        if self.image_augment:
            self.frame_augment = T.Compose([
                T.ToPILImage(), # 将图片转换成PIL图像格式，T.Resize()只能作用于 PIL 图像
                T.Resize(size=(self.image_size, self.image_size)), 
                T.RandomHorizontalFlip(p=0.5), #数据增强，水平翻转
            ])
        else:
            self.frame_augment = T.Compose([
                T.ToPILImage(), # 将图片转换成PIL图像格式，T.Resize()只能作用于 PIL 图像
                T.Resize(size=(self.image_size, self.image_size)), 
            ])

        if self.image_norm:
            # 数据集的标准差和均值
            self.image_mean = self.video_config.get('image_mean', [0.4850, 0.4560, 0.4060])
            self.image_std = self.video_config.get('image_std', [0.2290, 0.2240, 0.2250])

            self.transform_frame = T.Compose([
                T.ToTensor(),   
                # ToTensor() 将输入图像从 PIL 或 NumPy 格式转换为 PyTorch 张量（Tensor）
                # 在转换过程中，所有像素值会自动除以 255，映射到 [0, 1] 范围
                # ToTensor() 还会改变图像通道的顺序: H,W,C -->  C,H,W
                T.Normalize(
                   mean = self.image_mean,
                    std = self.image_std
                )
            ])
        else: # 不使用标准化，用于计算视频的均值和标准差
            self.transform_frame = T.Compose([
                T.ToTensor()
                # 将输入图像从 PIL 或 NumPy 格式转换为 PyTorch 张量（Tensor）
                # 在转换过程中，所有像素值会自动除以 255，映射到 [0, 1] 范围
                # ToTensor() 还会改变图像通道的顺序: H,W,C -->  C,H,W
                #T.Normalize(
                #   mean=[0.4850, 0.4560, 0.4060],
                #    std=[0.2290, 0.2240, 0.2250]
                #)
            ])            

        # 配置音频处理参数
        self.audio_config = audio_config
        self.audio_norm = self.audio_config.get('audio_norm', True)
        self.noise_augment = self.audio_config.get('noise_augment', False)
        self.num_mel_bins = self.audio_config.get('num_mel_bins', 128)
        self.target_length = self.audio_config.get('target_length', 1024)
        self.audio_mean = self.audio_config.get('audio_mean', -7.6)
        self.audio_std = self.audio_config.get('audio_std', 3.5)

        self.face_detector = pipeline.load_face_detection_model()

    def get_frames(self, video_name):
        frames = pipeline.extract_frames(video_name)
        frames = pipeline.extract_face_regions(frames, self.face_detector)
        #frames = [pipeline.extract_mouth_region(frame) for frame in frames]
        return frames

    def get_banks(self, video_name):
        return pipeline.wav2fbank(video_name, melbins=self.num_mel_bins, target_length=self.target_length)
    

    def __getitem__(self, index):
        video_name, label = self.data[index] # eg. xx.mp4,0

        # 将二元分类标签转换为独热码
        #label = torch.tensor([int(label), 1-int(label)]).float()
        label = torch.tensor(int(label), dtype=torch.long)  # 确保label是整数
        # label = torch.nn.functional.one_hot(label, num_classes=4).float()

        #根据文件名得到视频帧(16张嘴部区域)
        frames = self.get_frames(video_name)

        # if self.image_augment:

        frames = [self.frame_augment(frame) for frame in frames]

        frames = [self.transform_frame(frame) for frame in frames]
        frames = torch.stack(frames) #将列表[f1,f2,...,fn]转换成torch张量，形状为[n, f.shape]
        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # frames: (T, C, H, W) -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)

        #根据文件名得到音频信息(梅尔谱图)
        fbank = self.get_banks(video_name)
        if self.audio_norm: # 标准化
            fbank = (fbank - self.audio_mean) / (self.audio_std)
        else: # 不使用正则化，用于计算音频的均值和标准差
            pass
        
        #音频数据增强
        if self.noise_augment == True and random.random() < 0.5:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10  # 噪声
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0) # 时间偏移

        return fbank, frames, label

    def __len__(self):
        return self.num_samples
    
class ft_dataset_train(Dataset):
    pass


def compute_dataset_mean_std(dataset):
    """
    计算数据集的均值和标准差，用于归一化。

    参数:
        dataset: PyTorch Dataset 实例

    返回:
        image_mean, image_std: 图像的均值和标准差 (list)
        audio_mean, audio_std: 音频的均值和标准差 (float)
    """
    image_sum = torch.zeros(3)  # 图像通道数
    image_sum_sq = torch.zeros(3)
    num_pixels = 0

    audio_sum = 0.0
    audio_sum_sq = 0.0
    num_audio_samples = 0

    for i in range(len(dataset)):
        fbank, frames, _ = dataset[i]  # 只需音频特征 fbank 和视频帧 frames

        # 计算图像均值和标准差
        num_pixels += frames.shape[1] * frames.shape[2] * frames.shape[3]  # C x T x H x W
        image_sum += frames.sum(dim=(1, 2, 3))
        image_sum_sq += (frames ** 2).sum(dim=(1, 2, 3))

        # 计算音频均值和标准差
        num_audio_samples += fbank.numel()
        audio_sum += fbank.sum().item()
        audio_sum_sq += (fbank ** 2).sum().item()

    # 计算均值
    image_mean = image_sum / num_pixels
    audio_mean = audio_sum / num_audio_samples

    # 计算标准差
    image_std = torch.sqrt(image_sum_sq / num_pixels - image_mean ** 2)
    audio_std = (audio_sum_sq / num_audio_samples - audio_mean ** 2) ** 0.5

    return image_mean.tolist(), image_std.tolist(), audio_mean, audio_std


# 测试数据集数据流
if __name__ == '__main__':
    video_config = {
    'image_size': 224,
    'image_norm': False,
    'image_augment': False,
    }
    audio_config = {
    'num_mel_bins': 128,
    'target_length': 1024,
    'audio_norm': False,
    'noise_augment': False,
    }
    # Initialize dataset
    csv_file = "E:/jielun/graduation-project/data/csv/recon_demo.csv"
    dataset = ft_dataset_validate(csv_file, video_config, audio_config)

    # Test dataset length
    print(f"Dataset contains {len(dataset)} samples.")

    # Retrieve a sample
    sample_index = 0  # Change as needed
    fbank, frames, label = dataset[sample_index]

    # Output details
    print(f"Sample {sample_index}:")
    print(f" - FBank shape: {fbank.shape}")  # Expected: (target_length, num_mel_bins)
    print(f" - Frames shape: {frames.shape}")  # Expected: (C, T, H, W) -> (3, 16, 224, 224)
    print(f" - Label: {label}, Label shape: {label.shape}")  # Expected: torch tensor of shape (2,)

    # Ensure data types
    assert isinstance(fbank, torch.Tensor), "FBank should be a torch.Tensor"
    assert isinstance(frames, torch.Tensor), "Frames should be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    print(f"{compute_dataset_mean_std(dataset)}")
    """
    ([0.4372663199901581, 0.3238029181957245, 0.2959437072277069],
     [0.2681775689125061, 0.22621804475784302, 0.2293681651353836], 
     -7.617182418172405, 3.487446738313055)

    """
    print("Dataset test passed successfully!")
