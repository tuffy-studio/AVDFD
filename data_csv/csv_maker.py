import os
import csv

# 定义要整理的文件夹路径
dir_test = '/root/autodl-tmp/0/'
#dir_train = '/home/home/wangyuxuan/jielun/graduation-project/data/fine-tune/train_video/0'

# 定义输出的CSV文件路径
#train_0_csv = '/home/home/wangyuxuan/jielun/graduation-project/data/train_0.csv'

test_0_csv = '/root/autodl-tmp/test_real.csv'

# 创建或覆盖CSV文件
with open(test_0_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['video_name', 'target'])  # 写入表头

    # 处理目录1中的mp4文件
    for filename in os.listdir(dir_test):
        if filename.endswith('.mp4'):
            video_path = os.path.join(dir_test, filename)
            writer.writerow([video_path, 0])  # 将文件路径和标签1写入CSV

print(f"CSV file has been created at: {test_0_csv}")

