import csv
import time
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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

def draw_loss_curve(csv_file):
    epochs = []
    losses = []

    # 读取 CSV 文件
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                epoch = int(row[0])  # epoch 为整数
                loss = float(row[1])  # loss 为浮动值
                epochs.append(epoch)
                losses.append(loss)
        
        # 绘制损失曲线
        plt.plot(epochs, losses, label='Loss', color='blue')
        plt.xlabel('Epochs')  # x 轴标签
        plt.ylabel('Loss')    # y 轴标签
        plt.title('Training Loss Curve')  # 标题
        plt.legend()
        plt.grid(False)
        output_file = f"loss_curve_epoch1-{epochs[-1]}.png"
        plt.savefig(output_file)  # 保存为图片文件
    
    except Exception as e:
        print(f"Error while drawing loss: {e}")

def train(cavmae_ft, train_loader, val_loader, args):
    epochs=args.epochs
    epoch=1
    print("start training...")
    while epoch < epochs+1:
        model.train()
        begin_time = time.time()
        end_time = time.time()
        print('---------------')
        print(datetime.datetime.now())
        save_loss()
        if(epoch%10):
            draw_loss_curve()


def validate(model, val_loader, args, output_pred=False):
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
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                audio_output = model(a_input, v_input)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        stats = calculate_stats(audio_output.cpu(), target.cpu())

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, audio_output, target