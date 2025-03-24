import torch
from src.models.video_cav_mae import *
from collections import OrderedDict  # 有序字典，用于存储模型权重

input_weight_path = "./weights/pre-trained/stage-2.pth"  # 预训练模型权重
output_weight_path = "./weights/pre-trained/stage-2-adapted-4class.pth"  # 修改后的预训练模型权重
stage1_weight = torch.load(input_weight_path)
cavmae_ft = VideoCAVMAEFT()
cavmae_ft = torch.nn.DataParallel(cavmae_ft)
stage2_weight = OrderedDict()
for k in stage1_weight.keys():
    if ('mlp' in k and ('a2v' in k or 'v2a' in k)) or 'decoder' in k:
        continue
    stage2_weight[k] = stage1_weight[k]
missing, unexpected = cavmae_ft.load_state_dict(stage2_weight, strict=False)

'''
(['module.a2v.mlp.linear.weight',
  'module.a2v.mlp.linear.bias',
  'module.v2a.mlp.linear.weight',
  'module.v2a.mlp.linear.bias',
  'module.mlp_vision.weight',
  'module.mlp_vision.bias',
  'module.mlp_audio.weight',
  'module.mlp_audio.bias',
  'module.mlp_head.fc1.weight',
  'module.mlp_head.fc1.bias',
  'module.mlp_head.fc2.weight',
  'module.mlp_head.fc2.bias',
  'module.mlp_head.fc3.weight',
  'module.mlp_head.fc3.bias'],
 [])
'''

torch.save(cavmae_ft.state_dict(), output_weight_path)


