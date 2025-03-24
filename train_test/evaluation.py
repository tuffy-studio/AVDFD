import sys 
sys.path.append("/root/autodl-tmp/graduation-project/")

import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import csv
import argparse
from train_test.utils import *

# 读取模型参数路径与验证集路径
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="path to your stage-3 weights", default="/root/autodl-tmp/graduation-project/egs/exp/4-class-augment-3rd/models/audio_model.20.pth")
parser.add_argument("--csv_file", type=str, default="../data/csv/ft_test_4_segmented_modified.csv")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("==========")
print(f"running on {device}")
print("==========")

# 加载模型
model = VideoCAVMAEFT()
model = torch.nn.DataParallel(model)
ckpt = torch.load(args.checkpoint, map_location='cpu')
miss, unexp = model.load_state_dict(ckpt, strict=False)
assert len(miss) == 0 and len(unexp) == 0 


# 加载验证集
data_eval = args.csv_file

dataset_mean = -7.6171
dataset_std = 3.4874
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}
val_loader = torch.utils.data.DataLoader(
    dataloader.VideoAudioEvalDataset(csv_file=data_eval, audio_conf=val_audio_conf),
    batch_size=16, shuffle=False, num_workers=4, pin_memory=False)

data = []
with open(data_eval, 'r') as file:
    reader = csv.reader(file)    
    next(reader)
    for row in reader:
        data.append(row[0])

preds = {}
model.to(device)
A_predictions, A_targets = [], []


# 开始验证
with torch.no_grad():
    for i, (a_input, v_input, labels, video_names) in tqdm(enumerate(val_loader),
                                                            total=len(val_loader), 
                                                            desc="Processing data"):
        #if i ==100: 
           # break;
        a_input = a_input.to(device)
        v_input = v_input.to(device)

        with autocast():
            output = model(a_input, v_input) # output.shape=torch.tensor[4]

        # 二分类
        #probabilities = torch.sigmoid(output).cpu().numpy()

        # 将模型输出的四分类转换成概率分布，并以numpy数组的形式放到cpu上
        predictions = torch.softmax(output,  dim=-1, dtype=torch.float32).cpu()
        A_predictions.append(predictions)

        A_targets.append(labels)

        for j, item in enumerate(predictions):
            prediction = item
            preds[video_names[j].split("/")[-1]] = prediction
    
    A_predictions=torch.cat(A_predictions)
    A_targets=torch.cat(A_targets)
    A_targets = torch.nn.functional.one_hot(A_targets, num_classes=4).float()
    
    A_predictions=A_predictions.numpy() 

    A_targets=A_targets.numpy()

    stats = calculate_stats(A_predictions, A_targets)
    print(A_predictions[:, 0])
    print(A_targets[:, 0])
    
    #绘制曲线
    plot_confusion_matrix(A_predictions, A_targets)
    plot_confusion_matrix(A_predictions, A_targets, normalize=True)
    for i in range(4):
        plot_precision_recall_curve(A_targets[:, i], A_predictions[:, i], class_name=f"class {i+1}")
        plot_roc_curve(A_targets[:, i], A_predictions[:, i], class_name=f"class {i+1}")
    


# 保存验证数据
with open("prediction.csv", 'w') as f:
    f.write("video_name,y_pred\n")
    for k, v in preds.items():
        f.write(f"{k},{v}\n")




        