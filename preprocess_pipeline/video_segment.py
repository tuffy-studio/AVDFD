import os
import math
import csv
import glob
from tqdm import tqdm  # 导入 tqdm 进度条库
from moviepy import VideoFileClip


def split_videos_from_csv(input_csv, output_csv, output_dir, segment_length=3.20):
    """
    根据CSV文件中的视频路径，将视频分割，并将分割后的视频路径与标签写入新的CSV文件。

    :param input_csv: 原始 CSV 文件，包含视频路径和标签
    :param output_csv: 输出 CSV 文件，存储分割后的视频路径和标签
    :param output_dir: 分割后视频的存储路径
    :param segment_length: 每个片段的时长（默认 3.20 秒）
    """
    os.makedirs(output_dir, exist_ok=True)  # 创建存储目录（如果不存在）

    # 读取原始 CSV 文件
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取表头（如果存在）
        video_data = list(reader)  # 读取所有数据

    # 先写入 CSV 文件的表头
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_name", "target"])  # 写入表头

    with tqdm(total=len(video_data), desc="Processing Videos", unit="video") as pbar:
        for video_name, target in video_data:
            if not os.path.exists(video_name):
                print(f"文件 {video_name} 不存在，跳过...")
                pbar.update(1)
                continue

            base_name = os.path.splitext(os.path.basename(video_name))[0]  # 获取不带后缀的文件名
            
            video = VideoFileClip(video_name)
            duration = video.duration  # 获取视频时长
            num_segments = math.ceil(duration / segment_length)  # 计算分割片段数

            for i in range(num_segments):
                start_time = i * segment_length
                end_time = min(start_time + segment_length, duration)  # 防止超出视频时长
                if end_time - start_time <= 2:
                    continue

                # 生成分割后的视频文件名
                output_file = os.path.join(output_dir, f"{base_name}_part_{i + 1}.mp4")
                print(output_file)
                
                if not os.path.exists(output_file):
                    # 进行视频分割
                    segment = video.subclipped(start_time, end_time)
                    segment.write_videofile(output_file, codec="libx264", audio_codec="aac", logger=None)
                else:
                    print(f"file {video_name} has already been segmented. Only need to write csv file.")

                # 追加写入新的 CSV 文件
                with open(output_csv, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([output_file, target])

            video.close()  # 释放资源
            pbar.update(1)

    print(f"视频分割完成，结果已保存至 {output_csv}")


def modify_csv(input_file, output_file, path=""):
    with open(input_file, mode="r", encoding="utf-8") as infile, open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 读取并写入表头
        header = next(reader)
        writer.writerow(header)

        # 处理每一行数据
        for row in reader:
            row[0] = row[0].split("FakeAVCeleb_v1.2/")[-1]  # 去掉前面的路径，只保留从 FakeAVCeleb_v1.2 开始的部分
            row[0] = "FakeAVCeleb_v1.2/" + row[0]  # 确保前缀一致
            row[0] = row[0].replace("\\", "/")  # 统一路径分隔符为 '/'
            row[0] = path + row[0]  # 追加路径前缀
            writer.writerow(row)

    print(f"CSV file {input_file} modification completed")


def merge_csv(file1, file2, output_file):
    with open(file1, mode="r", encoding="utf-8") as f1, open(file2, mode="r", encoding="utf-8") as f2, open(output_file, mode="w", encoding="utf-8", newline="") as out:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(out)

        # 读取并写入表头
        header1 = next(reader1)
        header2 = next(reader2)

        # 确保表头一致
        if header1 != header2:
            raise ValueError("CSV 文件的列名不匹配，无法合并！")

        writer.writerow(header1)  # 写入表头
        writer.writerows(reader1)  # 写入第一个文件的数据
        writer.writerows(reader2)  # 写入第二个文件的数据

    print(f"CSV 文件合并完成，结果保存在 {output_file}")


def batch_process_csv(input_dir, output_dir, segments_dir):
    """
    对文件夹中所有 CSV 文件执行 split_videos_from_csv()
    :param input_dir: 存放源 CSV 文件的目录
    :param output_dir: 存放处理后 CSV 文件的目录
    :param segments_dir: 片段存储目录
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 获取 input_dir 目录下所有 CSV 文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    for src_csv in csv_files:
        # 获取文件名
        filename = os.path.basename(src_csv)
        segments_csv = os.path.join(output_dir, filename)

        # 执行 split_videos_from_csv
        split_videos_from_csv(src_csv, segments_csv, segments_dir)
        print(f"Processed: {src_csv} → {segments_csv}")


if __name__ == "__main__":
    # 设定路径
    input_dir = "../train_test/dataset_process/output_LOCO_modified"
    output_dir = "../train_test/dataset_process/output_LOCO_modified_segmented"

    #segments_dir = "/home/home/wangyuxuan/jielun/FakeAVCeleb_v1.2/"

    segments_dir = "E:/downloads/FakeAVCeleb_v1.2/cross_manipulation_segmented"

    # 批量处理
    batch_process_csv(input_dir, output_dir, segments_dir)


    # modify_csv(input_file="../data/csv/ft_test_4_augment_segmented.csv",
    #            output_file="../data/csv/ft_test_4_augment_segmented_modified.csv",
    #            path="/root/autodl-tmp/")

    # modify_csv(input_file="../data/csv/ft_train_4_augment_segmented.csv",
    #            output_file="../data/csv/ft_train_4_augment_segmented_modified.csv",
    #            path="/root/autodl-tmp/")





    # modify_csv(input_file="../data/csv/ft_test_4_segmented.csv",
    #            output_file="../data/csv/ft_test_4_segmented_modified.csv",
    #            path="/home/home/wangyuxuan/jielun/")

    # merge_csv(file1="../data/csv/ft_train_4_segmented_modified.csv",
    #            file2="../data/csv/ft_test_4_segmented_modified.csv",
    #            output_file="../data/csv/ft_all_4_segmented_modified.csv")


