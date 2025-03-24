import argparse
import traintest_finetune


# ============================解析参数===============================
# 创建 ArgumentParser 对象, 解析命令行参数
parser = argparse.ArgumentParser(description='Args for finetune')

# 训练超参数 
parser.add_argument('--data-train', type=str, help='path to train data csv')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--data-val', type=str, help='path to val data csv')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')

# 


# 将解析到的命令参数存储为args，后续通过args.<参数名> 可以使用解析出的参数值
args = parser.parse_args()
# =========================创建Dataloader============================

# =======================创建需要训练的模型===========================
# 参数初始化

# ==========================开始训练模型==============================
traintest_finetune.train()
