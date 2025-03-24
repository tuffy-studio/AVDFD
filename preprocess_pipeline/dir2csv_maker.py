import os
import csv

def generate_video_csv(video_dir, output_csv, target_label=0):
    """
    遍历指定目录中的 .mp4 文件，并将文件路径及标签写入 CSV 文件。

    :param video_dir: 存放视频文件的目录
    :param output_csv: 生成的 CSV 文件路径
    :param target_label: 视频的标签（默认 0）
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # 确保 CSV 文件目录存在

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['video_name', 'target'])  # 写入表头

        # 遍历目录中的 MP4 文件
        for filename in os.listdir(video_dir):
            if filename.endswith('.mp4'):
                video_path = os.path.join(video_dir, filename)
                writer.writerow([video_path, target_label])  # 写入文件路径和标签

    print(f"CSV 文件已创建: {output_csv}")