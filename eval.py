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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="path to your stage-3 weights", default="/root/autodl-tmp/graduation-project/egs/exp/4-class/models/audio_model.44.pth")
parser.add_argument("--csv_file", type=str, default="/root/autodl-tmp/test_real.csv")

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("==========")
print(f"running on {device}")
print("==========")
audio_model = VideoCAVMAEFT()
audio_model = torch.nn.DataParallel(audio_model)
ckpt = torch.load(args.checkpoint, map_location='cpu')
miss, unexp = audio_model.load_state_dict(ckpt, strict=False)
assert len(miss) == 0 and len(unexp) == 0

data_eval = args.csv_file

dataset_mean = -7.6171
dataset_std = 3.4874
target_length = 1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode': 'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}
val_loader = torch.utils.data.DataLoader(
    dataloader.VideoAudioEvalDataset(csv_file=data_eval, audio_conf=val_audio_conf),
    batch_size=4, shuffle=False, num_workers=4, pin_memory=False)

data = []
with open(data_eval, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(row[0])

preds = {}
audio_model.to(device)
with torch.no_grad():
    for i, (a_input, v_input, labels, video_names) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing data"):
        a_input = a_input.to(device)
        v_input = v_input.to(device)

        with autocast():
            audio_output = audio_model(a_input, v_input)
        probabilities = torch.softmax(audio_output, dim=-1, dtype=torch.float32).cpu().numpy()
        for j, item in enumerate(probabilities):
            probability = item
            max_index = np.argmax(probability)  
            #preds[video_names[j].split("/")[-1]] = probability
            preds[video_names[j].split("/")[-1]] = max_index
        
with open("prediction.csv", 'w') as f:
    f.write("video_name,y_pred\n")
    for k, v in preds.items():
        f.write(f"{k},{v}\n")


#print(calculate_stats(probs_False, labs_False))
#print(calculate_stats(probs_true, labs_true))



