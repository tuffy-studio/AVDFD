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

# 用于随机裁剪和调整图像大小
class RandomCropAndResize:
    def __init__(self, im_res):
        self.im_res = im_res

    def __call__(self, x):
        crop = T.RandomCrop(self.im_res)
        resize = T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC)
        return resize(crop(x))

class RandomAdjustContrast:
    def __init__(self, factor: list):
        self.factor = random.uniform(factor[0], factor[1])

    def __call__(self, x):
        return ImageEnhance.Contrast(x).enhance(self.factor)

class RandomColor:
    def __init__(self, factor: list):
        self.factor = random.uniform(factor[0], factor[1])

    def __call__(self, x):
        return ImageEnhance.Color(x).enhance(self.factor)

class VideoAudioDataset(Dataset):
    def __init__(self, csv_file, audio_conf, stage, num_frames=16):
    # 构造方法: 
    # csv_file: csv文件路径
    # audio_conf:包含音频配置参数的字典
        #audio_conf = { 
        #'num_mel_bins': 128,            # Mel频率倒谱系数（MFCC）中的Mel频带数量，用于音频特征提取
        #'target_length': 1024,          # 音频目标长度
        #'freqm': 0,                     # 频率掩码的长度，用于SpecAugment（用于频率域数据增强），值为0表示不使用
        #'timem': 0,                     # 时间掩码的长度，用于SpecAugment（用于时间域数据增强），值为0表示不使用
        #'mode': 'train',                # 当前模式，'train' 表示训练模式，'eval' 表示评估模式，决定数据处理方式
        #'mean': args.dataset_mean,      # 所用数据集的均值，用于音频特征的标准化
        #'std': args.dataset_std,        # 所用数据集的标准差，用于音频特征的标准化
        #'noise': args.noise,            # 是否使用噪声增强，如果为True则添加噪声
        #'im_res': 224 }                 
    # stage: 预训练阶段
    # num_frames: 需要提取的视频帧数
        self.num_frames = num_frames
        self.stage = stage
        
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) # 跳过 CSV 文件的第一行(表头: video_name,target)
            for row in reader:
                self.data.append(row)
        print(f'According to the csv file, dataset has {len(self.data)} samples')

        self.num_samples = len(self.data)
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')  # dictionary.get(key, default=None)
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        # 将数据调整为均值为 0, 标准差为 1 的分布
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')

        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, 
        # if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(self.im_res, self.im_res)),
            T.RandomHorizontalFlip(),  # 随机水平翻转
            T.ToTensor(),   
            # 将输入图像从 PIL 或 NumPy 格式转换为 PyTorch 张量（Tensor）
            # 在转换过程中，所有像素值会自动除以 255，映射到 [0, 1] 范围
            # ToTensor() 还会改变图像通道的顺序: H,W,C -->  C,H,W
            T.Normalize(
                mean=[0.4372, 0.3238, 0.2959],
                std=[0.2681, 0.2262, 0.2293]
            )
        ])
        
        # Perform augment
        # 在 Stage1 中，可以通过拼接两个真实的视频、裁剪视频或翻转视频帧来进行数据增强
        self.augment_1 = ['None']
        self.augment_1_weight = [5]
        
        # 在 Stage2 中，除了可以拼接两个真实的视频外，
        # 还可以将一个真实视频和一个伪造视频拼接，并且有可能用随机音频替换视频中的音频部分
        self.augment_2 = ['None', 'concat', 'replace']
        self.augment_2_weight = [5, 1, 1]

    def extract_audio_from_video(self, video_file, output_audio_file):
        """从.mp4文件中提取音频并保存为.wav格式"""
        try:
            # 使用 ffmpeg 从视频中提取音频并保存为 wav 格式
            ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run(quiet=True)  # ac=1 表示单声道，ar='16k' 设置采样率
            print(f"Audio successfully extracted to {output_audio_file}.")
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e}")

    def _wav2fbank(self, filename):
        # 如果文件是 .mp4 格式，先提取音频
        if filename.endswith('.mp4'):
            temp_audio_file = filename.replace('.mp4', '.wav')
            if not os.path.exists(temp_audio_file):
                self.extract_audio_from_video(filename, temp_audio_file)
            else:
                # print(f"Audio file {temp_audio_file} already exists.")
                pass
            filename = temp_audio_file

        # 加载音频文件
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        try:
            # 尝试提取梅尔频率倒谱系数（MFCC）
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except Exception as e:
            # 捕获具体异常并输出错误信息
            print(f"Error in loading audio or computing fbank: {e}")
            # 返回默认的 fbank 值以避免崩溃
            fbank = torch.zeros([512, 128]) + 0.01
            print('There was an error loading the fbank. Returning default tensor.')
        
        #调整 fbank的大小到1024*128
        #(time_frames, 128)-->(1, time_frames, 128)-->(1, 128, time_frames)
        # -->(1, 128, 1024)-->(1, 1024, 128)-->(1024, 128)
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2), size=(self.target_length,),
            mode='linear', align_corners=False).transpose(1, 2).squeeze(0)

        return fbank


    def _concat_wav2fbank(self, filename1, filename2):
        # 如果文件是 .mp4 格式，先提取音频
        if filename1.endswith('.mp4'):
            temp_audio_file = filename1.replace('.mp4', '.wav')
            if not os.path.exists(temp_audio_file):
                self.extract_audio_from_video(filename1, temp_audio_file)
            else:
                pass
                #print(f"Audio file {temp_audio_file} already exists.")
            filename1 = temp_audio_file

        if filename2.endswith('.mp4'):
            temp_audio_file = filename2.replace('.mp4', '.wav')
            if not os.path.exists(temp_audio_file):
                self.extract_audio_from_video(filename2, temp_audio_file)
            else:
                pass
                #print(f"Audio file {temp_audio_file} already exists.")
            filename2 = temp_audio_file

        waveform1, sr1 = torchaudio.load(filename1)
        waveform2, sr2 = torchaudio.load(filename2)
        waveform1 = waveform1 - waveform1.mean()
        waveform2 = waveform2 - waveform2.mean()

        try:
            fbank1 = torchaudio.compliance.kaldi.fbank(waveform1, htk_compat=True, sample_frequency=sr1, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
            fbank2 = torchaudio.compliance.kaldi.fbank(waveform2, htk_compat=True, sample_frequency=sr2, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank1 = torch.zeros([512, 128]) + 0.01
            fbank2 = torch.zeros([512, 128]) + 0.01
            print("there is a loading error")

        fbank = torch.concat((fbank1, fbank2), dim=0)
        
        target_length = self.target_length

        # Perform Down/Up Sample
        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length,), mode='linear', align_corners=False).transpose(1,2).squeeze(0)

        return fbank

    # 从视频中提取指定数量的帧
    def _get_frames(self, video_name):
        try:
            vr = VideoReader(video_name)
            total_frames = len(vr)  # Total number of frames in the video
        
            # Calculate the indices to sample uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        
            # Read the frames using the calculated indices
            frames = [vr[i].asnumpy() for i in frame_indices]

            # todo:对所选的视频帧进行人脸提取
        except:
            frames = torch.zeros(self.num_frames, 3, 224, 224)
            
        return frames
    
    def _concat_get_frames(self, video_name1, video_name2):
        try:
            vr1 = VideoReader(video_name1)
            vr2 = VideoReader(video_name2)

            frames_1 = [vr1[i].asnumpy() for i in range(len(vr1))]
            frames_2 = [vr2[i].asnumpy() for i in range(len(vr2))]

            frames = frames_1 + frames_2

            total_frames = len(vr1) + len(vr2)

            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

            frames = [frames[i] for i in frame_indices]
        
        except:
            frames = torch.zeros(self.num_frames, 3, 224, 224)
            
        return frames
    
    def _augment_concat(self, index):
        video_name, label = self.data[index]
        index_1 = random.choice([i for i in range(len(self.data))])
        video_name_1, label_1 = self.data[index_1]

        fbank = self._concat_wav2fbank(video_name, video_name_1)
        frames = self._concat_get_frames(video_name, video_name_1)

        if self.stage == 1:
            label_ = 0
        else:
            if int(label) == 0 and int(label_1) == 0:
                label_ = 0
            else:
                label_ = 1
        
        return fbank, frames, label_

    def _augment_replace(self, index):
        video_name, label = self.data[index]
        # if int(label) == 0:
        #     frames = self._get_frames(video_name)
        #     fbank = self._wav2fbank(video_name)
        #     return fbank, frames, label
        # else:
        label = 1
        index_1 = random.choice([i for i in range(len(self.data))])
        video_name_1, label_1 = self.data[index_1]
            
        # Replace audio with other
        frames = self._get_frames(video_name)
        fbank = self._wav2fbank(video_name_1)
        return fbank, frames, label

    def __getitem__(self, index):
        video_name, label = self.data[index] # eg. xx.mp4,0

        # 在验证模型下不使用数据增强
        if self.mode == 'eval':
            try:
                fbank = self._wav2fbank(video_name)
            except Exception as e:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print("!!!There is an error in loading audio:")
                print(f"Error in loading audio or computing fbank: {e}")
            
            frames = self._get_frames(video_name)
            frames = [self.preprocess(frame) for frame in frames]
            frames = torch.stack(frames)
        
        else:
            # Data Augment
            if self.stage == 1:
                augment = random.choices(self.augment_1, weights=self.augment_1_weight)[0]
            elif self.stage == 2:
                augment = random.choices(self.augment_2, weights=self.augment_2_weight)[0]

            if augment == 'concat':
                fbank, frames, label = self._augment_concat(index)
            elif augment == 'replace':
                fbank, frames, label = self._augment_replace(index)
            else:
                try:
                    fbank = self._wav2fbank(video_name)
                except:
                    fbank = torch.zeros([self.target_length, 128]) + 0.01
                    print('there is an error in loading audio3')
                
                frames = self._get_frames(video_name)

            # for i, frame in enumerate(frames):
                # if random.uniform(0, 1) < 0.1:
                #     frames[i] = self.preprocess_aug(frame)
                # else:
                #     frames[i] = self.preprocess(frame)
            frames = [self.preprocess(frame) for frame in frames]
            frames = torch.stack(frames)

            # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True and random.random() < 0.5:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # frames: (T, C, H, W) -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        
        #label = torch.tensor([int(label), 1-int(label)]).float()
        label = torch.tensor(int(label), dtype=torch.long)  # 确保label是整数

        return fbank, frames, label

    def __len__(self):
        return self.num_samples


class VideoAudioEvalDataset(Dataset):
    def __init__(self, csv_file, audio_conf, num_frames=16):
        self.num_frames = num_frames
        
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append(row)

        print('Dataset has {:d} samples'.format(len(self.data)))
        self.num_samples = len(self.data)
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.ToPILImage(),
            # T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4372, 0.3238, 0.2959],
                std=[0.2681, 0.2262, 0.2293]
            )
           ])

    def extract_audio_from_video(self, video_file, output_audio_file):
        """从视频文件中提取音频并保存为.wav格式"""
        try:
            # 使用 ffmpeg 从视频中提取音频并保存为 wav 格式
            ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run(quiet=True)  # ac=1 表示单声道，ar='16k' 设置采样率
            #print(f"Audio successfully extracted to {output_audio_file}")
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e}")

    def _wav2fbank(self, filename):
        # 如果文件是 .mp4 格式，先提取音频
        if filename.endswith('.mp4'):
            temp_audio_file = filename.replace('.mp4', '.wav')
            if not os.path.exists(temp_audio_file):
                self.extract_audio_from_video(filename, temp_audio_file)
            else:
                # print(f"Audio file {temp_audio_file} already exists.")
                pass
            filename = temp_audio_file

        # 加载音频文件
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        try:
            # 尝试提取梅尔频率倒谱系数（MFCC）
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except Exception as e:
            # 捕获具体异常并输出错误信息
            print(f"Error in loading audio or computing fbank: {e}")
            # 返回默认的 fbank 值以避免崩溃
            fbank = torch.zeros([512, 128]) + 0.01
            print('There was an error loading the fbank. Returning default tensor.')

        # 调整 fbank 的大小到目标长度
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2), size=(self.target_length,),
            mode='linear', align_corners=False).transpose(1, 2).squeeze(0)

        return fbank

    def _get_frames(self, video_name):
        try:
            vr = VideoReader(video_name)
            total_frames = len(vr)  # Total number of frames in the video
        
            # Calculate the indices to sample uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        
            # Read the frames using the calculated indices
            frames = [vr[i].asnumpy() for i in frame_indices]
        except:
            frames = torch.zeros(self.num_frames, 3, 224, 224)
            
        return frames

    def __getitem__(self, index):
        video_name, label = self.data[index]
        #label = torch.tensor([int(label), 1-int(label)]).float()  # 是假的概率，是真的概率

        label = torch.tensor(int(label), dtype=torch.long)  # 确保label是整数
        
        try:
            fbank = self._wav2fbank(video_name)
        except:
            fbank = torch.zeros([self.target_length, 128]) + 0.01
            print('there is an error in loading audio')
            
        frames = self._get_frames(video_name)
        frames = [self.preprocess(frame) for frame in frames]
        frames = torch.stack(frames)
            
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # convert fbank to 8*128*128
        # fbank = fbank.unsqueeze(0)
        # fbank = fbank.reshape(8, -1, 128)
        
        # frames: (T, C, H, W) -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        
        return fbank, frames, label, video_name

    def __len__(self):
        return self.num_samples
