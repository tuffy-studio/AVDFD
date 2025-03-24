#!/bin/bash
#SBATCH -p a5
#SBATCH --qos regular
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2,3],sls-sm-[5,6,7,12]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="as-pretrain"
#SBATCH --output=../log/%j_as_pretrain.txt

# run cav-mae pretraining, fits larger GPUs (4*24GB GPUs)

contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True

# you can use any checkpoints with a decoder, but by default, we use vision-MAE checkpoint
# pretrain_path=../weights/pre-trained/stage-2-adapted.pth
#pretrain_path=/root/autodl-tmp/graduation-project/egs/exp/stage-3-3rd/models/audio_model.11.pth
# pretrain_path=../weights/pre-trained/stage-2-adapted-4class.pth
# pretrain_path=/root/autodl-tmp/graduation-project/egs/exp/4-class/models/audio_model.50.pth
#pretrain_path=/root/autodl-tmp/graduation-project/egs/exp/4-class/models/audio_model.50.pth
pretrain_path=/root/autodl-tmp/graduation-project/egs/exp/4-class-augment-3rd/models/audio_model.5.pth
# lr=1e-5
lr=5e-7

head_lr=50
epoch=50
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa_start=1
wa_end=20
dataset_mean=-7.6171
dataset_std=3.4874
target_length=1024
noise=True
batch_size=4
lr_adapt=False

n_print_steps=100

#tr_data=../data/csv/ft_train_4_augment_segmented_modified.csv
#tr_data=../data/csv/last_train.csv
#te_data=../data/csv/ft_test_4_segmented_modified.csv


# exp_dir=./exp/self-pretrain
save_dir=./exp/4-class-augment-3rd
mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python -W ignore ../src/run_ft.py \
--data-train ${tr_data} --data-val ${te_data} --save-dir $save_dir --n_classes 4 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--loss CE --metrics mAP --warmup True \
--wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--head_lr ${head_lr} \
--pretrain_path ${pretrain_path} --num_workers 4\