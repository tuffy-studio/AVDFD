import os
import random
import csv

# 定义 CSV 写入函数
def save_csv(file_path, data):
    with open(file_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video-path", "label"])  # 写入表头
        writer.writerows(data)  # 写入数据


def divide_dataset(dataset_root, label_mapping):
    video_data = []
    # 遍历数据集，收集视频路径及其标签
    for category, label in label_mapping.items():
        category_path = os.path.join(dataset_root, category)
        for root, _, files in os.walk(category_path):
            for file in files:
                if file.endswith(".mp4"):  # 只处理 MP4 视频文件
                    video_path = os.path.join(root, file).replace("\\", "/")  # 统一路径格式
                    video_data.append((video_path, label))

    # 随机打乱数据
    random.shuffle(video_data)

    # 计算训练集和测试集大小
    train_size = int(0.7 * len(video_data))

    # 划分训练集和测试集
    train_data = video_data[:train_size]
    test_data = video_data[train_size:]

    # 保存 CSV 文件
    train_csv_path = os.path.join(dataset_root, "train_4.csv")
    test_csv_path = os.path.join(dataset_root, "test_4.csv")

    save_csv(train_csv_path, train_data)
    save_csv(test_csv_path, test_data)

    print(f"训练集已保存: {train_csv_path}")
    print(f"测试集已保存: {test_csv_path}")


# 设置 FakeAVCeleb 数据集的根目录
dataset_root = "E:\\DOWNLOADS\\FAKEAVCELEB_V1.2\\FAKEAVCELEB_V1.2"  # 请修改为你的实际路径

# 定义真假类别映射
label_mapping_2 = {
    "RealVideo-RealAudio": "0",
    "RealVideo-FakeAudio": "1",
    "FakeVideo-RealAudio": "1",
    "FakeVideo-FakeAudio": "1",
}

# divide_dataset(dataset_root=dataset_root, label_mapping=label_mapping_2)


# 定义真假类别映射
label_mapping_4 = {
    "RealVideo-RealAudio": "0",
    "RealVideo-FakeAudio": "1",
    "FakeVideo-RealAudio": "2",
    "FakeVideo-FakeAudio": "3",
}

divide_dataset(dataset_root=dataset_root, label_mapping=label_mapping_4)