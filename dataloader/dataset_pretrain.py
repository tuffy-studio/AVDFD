import torch
import os
import torchaudio
torchaudio.set_audio_backend("librosa")
from pydub import AudioSegment
import ffmpeg
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from decord import VideoReader
from decord import cpu
import torchvision.transforms as T
import PIL
import csv
import random
from PIL import ImageEnhance

class pt_dataset_train(Dataset):
    def __init__(self, csv_file, video_config, audio_config):
        '''
        构造方法: 
        csv_file: 视频数据集文件夹的路径
        video_config: 存储视频处理配置参数的字典
        audio_config: 存储音频处理配置参数的字典
        '''
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) # 跳过 CSV 文件的第一行，即表头: video_name, label
            for row in reader:
                self.data.append(row)
        self.num_samples = len(self.data)
        print(f'Dataset {csv_file} contains {self.num_samples} samples.')
        
        self.video_config = video_config
        self.audio_config = audio_config
        



class pt_dataset_validate(Dataset):
    pass
