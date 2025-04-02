import numpy as np
from scipy import stats
from sklearn import metrics
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非 GUI 后端，适用于服务器或远程环境
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import os

import seaborn as sns
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    '''
    用于计算和存储某个数值的当前值val、计数count、总和sum、平均值sum
    '''

    def __init__(self):
        self.reset()

    # 重置
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 更新
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # ACC
    acc = metrics.accuracy_score(np.argmax(target, axis=1), np.argmax(output, axis=1))

    # Class-wise statistics
    for k in range(classes_num):

        # AP
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        try:
            # AUC
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1  # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
        except:
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': avg_precision,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
            print('class {:s} no true sample'.format(str(k)))
        stats.append(dict)

    return stats


def plot_precision_recall_curve(y_true, y_scores, class_name, save_dir):
    """
    绘制 Precision-Recall (PR) 曲线，并显示 AP (Average Precision)。

    :param y_true: 真实标签 (1D NumPy 数组)，二进制 (0 或 1)
    :param y_scores: 预测得分 (1D NumPy 数组)，通常是模型输出的概率
    :param class_name: 类别名称
    :param save_dir: 存放目录
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, marker='', markersize=4, color='#EB5757', label=f"AP = {ap_score:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({class_name})")
    plt.legend(loc='lower right')  # 固定图例位置
    plt.grid()
    # plt.show()
    save_path = os.path.join(save_dir, f"PR_curve_{class_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)


def plot_roc_curve(y_true, y_scores, class_name, save_dir):
    """
    绘制 ROC (Receiver Operating Characteristic) 曲线，并显示 AUC (Area Under Curve)。

    :param y_true: 真实标签 (1D NumPy 数组)，二进制 (0 或 1)
    :param y_scores: 预测得分 (1D NumPy 数组)，通常是模型输出的概率
    :param class_name: 类别名称
    :param save_dir: 存放目录
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, marker='', markersize=4, color='#2E75B5', label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # 参考线（随机分类器）
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve ({class_name})")
    plt.legend(loc='lower right')  # 固定图例位置
    plt.grid()
    # plt.show()
    save_path = os.path.join(save_dir, f"ROC_curve_{class_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)


def plot_confusion_matrix(save_dir, y_true: np.ndarray, y_pred: np.ndarray, class_names=None, normalize=False):
    """
    绘制混淆矩阵（支持归一化）

    参数:
        save_dir: 存放目录
        y_true (np.ndarray): 真实类别数组 (1D)
        y_pred (np.ndarray): 预测类别数组 (1D)
        class_names (list, 可选): 类别名称列表，默认为 None（自动生成 0,1,2,...）
        normalize (bool, 可选): 是否归一化（显示百分比）

    示例:
        y_true = np.array([0, 1, 2, 1, 2, 0])
        y_pred = np.array([0, 2, 2, 1, 0, 0])
        plot_confusion_matrix(y_true, y_pred, class_names=["A", "B", "C"], normalize=True)
    """

    # 计算混淆矩阵
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    # 是否归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)  # 每行归一化，计算分类准确率
        cm = np.nan_to_num(cm)  # 避免除零错误

    # 类别标签
    if class_names is None:
        class_names = [str(i + 1) for i in range(cm.shape[0])]

    # 绘制热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    # 设置标签
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    titlename = f"Confusion Matrix {('(normalized)' if normalize else '')}"
    plt.title(titlename)
    # plt.show()
    save_path = os.path.join(save_dir, f"CM{'_normalized' if normalize else ''}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)


if __name__ == "__main__":
    y_true = np.random.randint(0, 2, size=100)  # 真实标签
    y_scores = y_true * 0.1 + np.random.rand(100) * 0.9
    plot_precision_recall_curve(y_true, y_scores, class_name="1", save_dir="./test/")
    plot_roc_curve(y_true, y_scores, class_name="1", save_dir="./test/")
    y_scores = y_true * 0.3 + np.random.rand(100) * 0.7
    plot_precision_recall_curve(y_true, y_scores, class_name="1", save_dir="./test/")
    plot_roc_curve(y_true, y_scores, class_name="1", save_dir="./test/")