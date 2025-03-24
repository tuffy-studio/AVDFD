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
import torch.nn.functional as F

def save_loss(csv_file, epoch, loss):
    # 如果 CSV 文件不存在，就创建并写入表头
    try:
        # 打开 CSV 文件，追加模式（'a'）避免覆盖原数据
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # 如果文件为空，写入表头
            if file.tell() == 0:  # 检查文件是否为空
                writer.writerow(['epoch', 'loss'])  # 表头

            # 写入当前 epoch 和损失值
            writer.writerow([epoch, loss])
    except Exception as e:
        print(f"Error while saving loss: {e}")

def train(model, train_loader, test_loader, args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start training model on {device}")
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time, loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    

    # 
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # 在微调阶段需要重新初始化的层（即未经过预训练的层），因此它们在微调时会使用更大的学习率
    mlp_list = [
        'a2v.mlp.linear.weight',
        'a2v.mlp.linear.bias',

        'v2a.mlp.linear.weight',
        'v2a.mlp.linear.bias',

        'mlp_vision.weight',
        'mlp_vision.bias',

        'mlp_audio.weight',
        'mlp_audio.bias',

        'mlp_head.fc1.weight',
        'mlp_head.fc1.bias',
        
        'mlp_head.fc2.weight',
        'mlp_head.fc2.bias'
    ]

    # 知识点1:
    # DataParallel（数据并行）或 DistributedDataParallel（分布式数据并行）等并行训练的情况下，
    # 模型通常会被封装在一个 DataParallel 或 DistributedDataParallel 的包装类中。
    # model.module 可以访问到实际的模型本身，即去除了并行包装的模型。
    # 如果没有使用并行训练，model.module 就等同于 model。

    # 知识点2:
    # named_parameters是 PyTorch 的一个方法，用于获取模型中所有的参数并且返回一个生成器（或迭代器），
    # 每个元素是一个元组 (parameter_name, parameter_value)，
    # 其中 parameter_name 是参数的名称（通常是层的名称，比如 conv1.weight、fc.bias 等），
    # parameter_value 是该参数的实际值（即权重或偏置）。
    
    # 知识点3:
    # filter() 是一个内置的 Python 函数，用于从一个可迭代对象中筛选出满足特定条件的元素
    # 语法为: filter(function, iterable) 
    # function: 用于筛选的函数，iterable: 要筛选的可迭代对象。
    # filter() 会将 iterable 中的每个元素传入 function 中，如果 function 返回 True，则保留该元素，否则过滤掉。

    # 知识点4:
    # 匿名函数（lambda 函数，没有名称的函数）是 Python 中的一种简化版的函数定义方式，
    # 它不需要使用 def 关键字来定义函数，而是使用 lambda 关键字。
    # lambda 参数(一个或多个): 表达式(函数的返回值)
    # 计算两个数的和: add = lambda x, y: x + y // print(add(2, 3))  # 输出 5

    #mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
    # 包含所有在 mlp_list 中列出的参数名称及其对应值的列表
    mlp_params = list(
        filter(  # 1. filter() 用于筛选元素
            lambda kv: kv[0] in mlp_list,  # 2. lambda 函数检查参数名称是否在 mlp_list 中
            model.module.named_parameters()  # 3. 获取模型中所有参数的 (name, value) 元组
        )
    )

    #base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
    # 包含所有不在 mlp_list 中列出的参数名称及其对应值的列表
    base_params = list(
        filter(
            lambda kv: kv[0] not in mlp_list, 
            model.module.named_parameters()
            )
    )

    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    
    trainables = [p for p in model.parameters() if p.requires_grad]

    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    
    # torch.optim.Adam 是 PyTorch 中常用的优化器，它的主要参数有:
    # 1. params: iterable，通常是一个包含模型参数的列表，通常传入模型的 parameters()
    # 2. lr: 学习率
    # 3. betas: 
    # 4. weight_decay: L2 正则化（权重衰减）的系数
    # 5. eps: 一个非常小的常数，防止除以零错误, 默认值通常为 1e-8。
    # torch.optim.Adam 支持传入一个 包含字典 的列表，
    # 允许为模型的不同参数组(optimizer.param_groups)设置不同的学习率、权重衰减等优化器设置
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, 
                                  {'params': mlp_params, 'lr': args.lr * args.head_lr}], 
                                  weight_decay=5e-7, betas=(0.95, 0.999))
    
    checkpoint = torch.load("%s/models/best_optim_state.pth" % (exp_dir))  # 加载优化器状态
    #for param_group in optimizer.param_groups:
    #    param_group['weight_decay'] = 1e-3
    optimizer.load_state_dict(checkpoint)  # 恢复优化器状态

    # optimizer.param_groups是一个包含一个或多个字典的列表，每个字典描述了一个参数组的配置
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)
    
    # 知识点1:
    # model.parameters() 返回的是封装后的 DataParallel 对象的参数列表，它包含每个 GPU 上的副本
    # model.module 是访问 DataParallel 内部原始模型的接口,
    # model.module.parameters() 返回的是原始模型的参数

    # 知识点2：
    # numel() 是 PyTorch 中一个张量（tensor）的方法，
    # 用于返回张量中元素的总数（即该张量的形状中的所有维度相乘得到的值）
    
    # 知识点3:
    # '{:.3f}' 是格式化字符串，表示将结果保留小数点后 3 位。
    # format() 方法将计算结果插入到字符串中，形成最终的输出。 
    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    

    # torch.optim.lr_scheduler 是 PyTorch 中的一个模块，它提供了一些用于动态调整学习率的调度器，包括:
    # 1.StepLR: 在固定的步数后根据常数因子调整lr, 例如torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    # 2.MultiStepLR: 可以在训练过程中根据给定的多次步长（milestones）调整学习率
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # milestones：一个整数列表，指定在哪些 epoch 进行学习率调整
    # 3.ReduceLROnPlateau：基于性能指标（如验证集的损失）来调整学习率的策略。
    # 当性能指标在一段时间内不再改善时，学习率会被减少。
    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # mode: 如果设置为 'min'，当监控的指标不再下降时调整学习率；如果设置为 'max'，当指标不再上升时调整学习率
    # factor：每次调整时的衰减因子
    # patience：在调整学习率之前，允许多少个 epoch 内的性能指标没有改善。
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics

   # 定义损失函数
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1

    # 自动缩放梯度，解决在使用较低精度（如 float16）训练时可能出现的数值不稳定问题
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])  # for each epoch, 10 metrics to record
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        for i, (a_input, v_input, labels) in enumerate(train_loader):
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            # 使用 autocast 来进行混合精度前向计算
            with autocast():
                output = model(a_input, v_input)
                loss = loss_fn(output, labels)
            
            print(f"output: {output.to('cpu').detach()}")

            optimizer.zero_grad()

            # 使用 GradScaler 缩放损失并进行反向传播
            scaler.scale(loss).backward()

            # 使用 GradScaler 更新参数
            scaler.step(optimizer)

            # 更新 GradScaler 的状态
            scaler.update()
            
            print("OK")

            # loss_av is the main loss
            loss_meter.update(loss.item(), B)
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
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                with open("train_log.txt", "a") as log_file:
                    log_file.write('{0}, {1}, {2}, {loss_meter.val:.4f}\n'.format(epoch, i, len(train_loader),
                                loss_meter=loss_meter))
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            save_loss("train_loss_rtvc.csv", epoch=epoch, loss=loss_meter.avg)
        
            end_time = time.time()
            global_step += 1
        
            if args.save_model == True:
                torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        #========================================模型验证=================================
        print('start validation')
        stats, valid_loss = validate(model, test_loader, args, num_class=2)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        # print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        save_loss("test_loss_rtvc.csv", epoch=epoch, loss=valid_loss)

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        # 保存模型参数
        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
    
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()
            
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))
        epoch += 1

        # 每个epoch重置计数类
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()
        


def validate(model, val_loader, args, output_pred=False, num_class=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            print(f"validate number: {i}")
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            print(f"labels: {labels}")
            A_targets.append(labels)
            labels = labels.to(device)
 

            with autocast():
                audio_output = model(a_input, v_input)
                loss = args.loss_fn(audio_output, labels)

            predictions = audio_output.to('cpu').detach()
            print(f"output: {predictions}")

            A_predictions.append(predictions)

            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        target = F.one_hot(target, num_classes=num_class).float()
        stats = calculate_stats(audio_output.cpu(), target.cpu())

    if output_pred == False:
        return stats, loss
    else:
        return stats, audio_output, target