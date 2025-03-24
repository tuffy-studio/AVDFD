import sys
import os
import csv
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np

# 初始化CSV文件，写入表头
csv_file = './loss_total/loss_data.csv'
header = ['Epoch', 'Train_Audio_MAE_Loss', 'Val_Audio_MAE_Loss', 'Train_Visual_MAE_Loss',
          'Val_Visual_MAE_Loss', 'Train_Contrastive_Loss', 'Val_Contrastive_Loss', "Train_Total_Loss", "Val_Total_Loss"]

if not os.path.exists(csv_file):  # 如果文件不存在，创建文件并写入表头
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def update_loss_plot(epoch, train_loss, val_loss, name, save_path=None):
    """
    动态绘制训练损失和验证损失曲线，并更新图表。

    参数:
    - epoch: 当前的训练 epoch
    - train_loss: 当前的训练损失列表
    - val_loss: 当前的验证损失列表
    - save_path: 保存图像的路径 (可选)
    """
    # 创建一个新的图表
    plt.figure(figsize=(10, 6))

    # 绘制训练损失和验证损失曲线
    plt.plot(range(1, epoch + 1), train_loss, label='Train Loss', color='blue')
    plt.plot(range(1, epoch + 1), val_loss, label='Validation Loss', color='red')

    # 设置标题和标签
    plt.title(f'Train and Validation {name}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{name} Loss')

    # 添加图例
    plt.legend()

    # 强制横轴为整数
    xticks = range(1, epoch + 1, 10)  # 每10个 epoch 设置一个刻度
    plt.xticks(xticks)

    # 布局优化
    plt.tight_layout()

    # 如果提供了保存路径，保存图像
    if save_path:
        # 获取文件夹路径
        folder_path = os.path.dirname(save_path)

        # 如果文件夹不存在，创建文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 不存在，已创建。")

        # 保存图像
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")


def initialize_plot():
    # 设置图表和轴
    plt.figure(figsize=(10, 6))

    # 创建三个子图（分别对应MAE_A、MAE_V、Contrastive Loss）
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title('Reconstruct Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title('Cross Modality Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title('Contrastive Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')

    # 返回轴对象用于更新
    return ax1, ax2, ax3


def update_plot(ax1, ax2, ax3, epoch, train_loss_mae_a, val_loss_mae_a, 
                train_loss_mae_v, val_loss_mae_v, train_loss_c, val_loss_c, save_path=None):
    # 清除原有数据（避免重复绘制）
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # 重新设置标题、标签
    ax1.set_title('Audio MAE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.set_title('Visual MAE Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    
    ax3.set_title('Contrastive Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')

    # 绘制训练集和验证集损失曲线
    ax1.plot(range(1, epoch + 1), train_loss_mae_a, label='Train Audio MAE Loss', color='blue')
    ax1.plot(range(1, epoch + 1), val_loss_mae_a, label='Validation Audio MAE Loss', color='red')
    ax2.plot(range(1, epoch + 1), train_loss_mae_v, label='Train Visual MAE Loss', color='blue')
    ax2.plot(range(1, epoch + 1), val_loss_mae_v, label='Validation Visual MAE Loss', color='red')
    ax3.plot(range(1, epoch + 1), train_loss_c, label='Train Contrastive Loss', color='blue')
    ax3.plot(range(1, epoch + 1), val_loss_c, label='Validation Contrastive Loss', color='red')


    # 添加图例
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # 强制横轴为整数
    ax1.set_xticks(range(1, epoch + 1))
    ax2.set_xticks(range(1, epoch + 1))
    ax3.set_xticks(range(1, epoch + 1))

    # 更新图表
    plt.tight_layout()
    plt.pause(0.1)  # 暂停以更新图表

    if save_path:
        # 获取文件夹路径
        folder_path = os.path.dirname(save_path)
        
        # 如果文件夹不存在，创建文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 不存在，已创建。")

        # 保存图像
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")



def train(model, train_loader, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    print(f"Start pre-training model on {device}")
    
    # 初始化损失记录器和损失曲线
    train_loss_mae_a, train_loss_mae_v, train_loss_c = [], [], []
    val_loss_mae_a, val_loss_mae_v, val_loss_c = [], [], []
    train_loss = []
    val_loss = []
    # ax1, ax2, ax3 = initialize_plot()

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    best_epoch, best_loss = 0, np.inf

    # 全局训练步数，每个训练步（step）通常表示一次前向传播和反向传播的过程。
    # 一个步数相当于处理了一个批次（batch）数据，并且完成了模型参数的更新。
    global_step = 0 
    epoch = 1 
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.to(device)

    # 提取模型中所有需要梯度更新的参数
    trainables = [p for p in model.parameters() if p.requires_grad] 
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    # Adam优化器
    optimizer = torch.optim.Adam(trainables, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    
    # 分段学习率调整器MultiStepLR，调整optimizer的学习率
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, # 调整对象
                list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
                # 调整学习率的 epoch 列表 (10, 1000, 5)
                # 从 args.lrscheduler_start 开始到 1000，每隔 args.lrscheduler_step 个 epoch 降低一次学习率
                gamma=args.lrscheduler_decay # 学习率调整的衰减因子0.5
                )
    
    # GradScaler()的作用: 动态地调整梯度的缩放比例，防止梯度下溢或溢出，用于提高训练稳定性
    # 使用原因: 在使用 FP16 时，梯度值较小时可能下溢（接近零被截断为零）
    # 原理: 在反向传播时，将梯度值乘以一个动态缩放因子（scale），避免梯度下溢，在更新参数时，再将梯度值恢复到原始范围
    # 同时，动态调整缩放因子，成功更新: 如果梯度没有溢出或下溢，则逐步增大缩放因子，提高数值的动态范围。
    # 失败更新: 如果发生梯度溢出（数值太大），则减小缩放因子，降低数值范围，避免训练不稳定。
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    result = np.zeros([args.n_epochs, 10])  # 每轮训练记录10个指标
    model.train()
 
    # ====================================================================================
    # 训练主循环
    while epoch < args.n_epochs + 1:
        model.train()
        print('---------------')
        print(datetime.datetime.now()) # 打印当前的日期和时间
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        begin_time = time.time()
        end_time = time.time()
        
        # 每次迭代
        for i, (a_input, v_input, _) in enumerate(train_loader): # (a_input, v_input, _): fbank, frames, label
            # 确保音频和视频输入的 batch_size 相同
            assert a_input.shape[0] == v_input.shape[0] 
            B = a_input.shape[0]

            # 将数据传输到GPU，异步数据传输，不会阻塞程序的执行
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            
            # 更新数据加载时间
            data_time.update(time.time() - end_time) # batch加载时间
            per_sample_data_time.update((time.time() - end_time) / B) # 每个样本的平均加载时间
            dnn_start_time = time.time()
       
            with autocast(): # 自动混合精度
                # 前向传播：计算损失和其他指标
                loss, loss_c, c_acc, loss_mae_a, loss_mae_v, _, _ = model(a_input, v_input, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight)
                loss, loss_c, loss_mae_a, loss_mae_v, c_acc = loss.sum(), loss_c.sum(), loss_mae_a.sum(), loss_mae_v.sum(), c_acc.mean()
            
            optimizer.zero_grad() # 梯度清零
            scaler.scale(loss).backward() # 使用GradScaler来处理反向传播
            scaler.step(optimizer) # 使用GradScaler进行梯度更新
            scaler.update()  # 更新GradScaler的状态
            
            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Total Loss {loss_av_meter.val:.4f}\t'
                  'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
                  'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                  'Train Contrastive Acc {c_acc:.3f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        # 记录训练损失
        train_loss_mae_a.append(loss_a_meter.avg)
        train_loss_mae_v.append(loss_v_meter.avg)
        train_loss_c.append(loss_c_meter.avg)
        train_loss.append(loss_av_meter.avg)

        # ====================================================================================
        #模型验证阶段
        print('start validation')
        eval_loss_av, eval_loss_c, eval_c_acc, eval_loss_mae_a, eval_loss_mae_v = validate(model, test_loader, args)

        # 记录验证损失
        val_loss_mae_a.append(eval_loss_mae_a)
        val_loss_mae_v.append(eval_loss_mae_v)
        val_loss_c.append(eval_loss_c)
        val_loss.append(eval_loss_av)

        # 更新绘图
        # save_path=f"./loss2/epoch{epoch}.png"
        # update_plot(ax1, ax2, ax3, epoch, train_loss_mae_a, val_loss_mae_a, train_loss_mae_v, val_loss_mae_v, train_loss_c, val_loss_c, save_path=save_path)

        save_path = f"./loss_total/epoch{epoch}.png"
        update_loss_plot(epoch, train_loss, val_loss, name='Total Loss', save_path=save_path)
        #save_path = f"./loss_cl/epoch{epoch}.png"
        #update_loss_plot(epoch, train_loss_c, val_loss_c, name='Constractive learning loss', save_path=save_path)
        #save_path = f"./loss_cm/epoch{epoch}.png"
        #update_loss_plot(epoch, train_loss_mae_v, val_loss_mae_v, name='Cross Modality Loss', save_path=save_path)
        #save_path = f"./loss_re/epoch{epoch}.png"
        #update_loss_plot(epoch, train_loss_mae_a, val_loss_mae_a, name='Reconstruction Loss', save_path=save_path)



        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,
                             train_loss_mae_a[-1], val_loss_mae_a[-1],
                             train_loss_mae_v[-1], val_loss_mae_v[-1],
                             train_loss_c[-1], val_loss_c[-1]],
                             train_loss[-1], val_loss[-1])


        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))
        
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        
        # 如果验证集上的损失变得更小，保存最佳模型
        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch
            
        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        
        # 是否每轮都保存模型
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        # 每轮epoch更新各个变量
        epoch += 1
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()


def validate(model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()

    A_loss, A_loss_c, A_c_acc, A_loss_mae_a, A_loss_mae_v = [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                loss, loss_c, c_acc, loss_mae_a, loss_mae_v, _, _ = model(a_input, v_input,
                                                                          mae_loss_weight=args.mae_loss_weight,
                                                                          contrast_loss_weight=args.contrast_loss_weight)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_c, loss_mae_a, loss_mae_v, c_acc = loss.sum(), loss_c.sum(), loss_mae_a.sum(), loss_mae_v.sum(), c_acc.mean()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_c, c_acc, loss_mae_a, loss_mae_v