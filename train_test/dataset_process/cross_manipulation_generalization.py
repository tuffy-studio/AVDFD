import csv
import os
from collections import Counter
import glob
import random


def count_type_method_combinations(csv_file):
    combination_counts = Counter()

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            type_value = row.get('type', '').strip()  # 获取type列
            method_value = row.get('method', '').strip()  # 获取method列
            if type_value and method_value:  # 确保type和method都有值
                combination = f"{type_value}-{method_value}"  # 将type和method组合成一个字符串
                combination_counts[combination] += 1

    # 输出每种组合及其出现的次数
    for combination, count in combination_counts.items():
        print(f"{combination}: {count}")

    return combination_counts


def manipulation_types_divide(csv_file, AV_type, manipulation_type):

    input_file = csv_file  # 原始 CSV 文件
    output_dir = "output"  # 存储所有生成的 CSV 文件
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    output_file = os.path.join(output_dir, f"{AV_type}-{manipulation_type}.csv")  # 结果文件

    with open(input_file, mode='r', encoding='utf-8') as infile, \
            open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)  # 读取 CSV 文件，按列名解析
        writer = csv.writer(outfile)
        writer.writerow(["path", "label"])  # 写入表头
        for row in reader:
            file_path = os.path.join(row["path"], row["filename"]).replace("\\", "/")
            if row["type"]==AV_type and row["method"] == manipulation_type:
                writer.writerow([file_path, "0" if row["category"] == "A" else "1"])


def leave_one_out_test(csv_dir="output", output_dir="output_LOCO"):
    """
    Perform Leave-One-Class-Out (LOCO) cross-validation on the given CSV files.

    Args:
    - csv_dir (str): Path to the directory containing the class CSV files.
    - output_dir (str): Path to the output directory where the results will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # List of classes
    types = [
        ["RealVideo-RealAudio", "real"],
        ["RealVideo-FakeAudio", "rtvc"],
        ["FakeVideo-RealAudio", "wav2lip"],
        ["FakeVideo-FakeAudio", "wav2lip"],
        ["FakeVideo-FakeAudio", "faceswap-wav2lip"],
        ["FakeVideo-FakeAudio", "fsgan-wav2lip"]
    ]

    # Load the CSV files for each class
    data_files = {f"{AV}-{method}": os.path.join(csv_dir, f"{AV}-{method}.csv") for AV, method in types}

    # Always include "real" in training
    real_file = data_files["RealVideo-RealAudio-real"]

    # Perform LOCO (Leave-One-Class-Out) Cross-Validation
    for test_key in [key for key in data_files.keys() if key != "RealVideo-RealAudio-real"]:
        # Get the files to be used for training (excluding the current test class)
        train_files = [real_file] + [data_files[key] for key in data_files.keys() if
                                     key != test_key and key != "RealVideo-RealAudio-real"]

        # Create the training CSV
        train_output_file = os.path.join(output_dir, f"train_excl_{test_key}.csv")
        with open(train_output_file, mode='w', newline='', encoding='utf-8') as train_file:
            writer = csv.writer(train_file)
            writer.writerow(["path", "label"])  # Write header

            # Write data for training
            for file in train_files:
                with open(file, mode='r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    next(reader)  # Skip header
                    for row in reader:
                        writer.writerow(row)  # Write rows to training file

        # Create the testing CSV for the current test class
        test_file = data_files[test_key]
        test_output_file = os.path.join(output_dir, f"test_{test_key}.csv")
        with open(test_output_file, mode='w', newline='', encoding='utf-8') as test_file_out:
            writer = csv.writer(test_file_out)
            writer.writerow(["path", "label"])  # Write header

            with open(test_file, mode='r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)  # Write rows to test file

        print(f"Created train_excl_{test_key}.csv and test_{test_key}.csv")


def modify_csv(input_file, output_file, path=""):
    with open(input_file, mode="r", encoding="utf-8") as infile, open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 读取并写入表头
        header = next(reader)
        writer.writerow(header)

        # 处理每一行数据
        for row in reader:
            row[0] = row[0].split("FakeAVCeleb/")[-1]  # 去掉前面的路径，只保留从 FakeAVCeleb_v1.2 开始的部分
            row[0] = "FakeAVCeleb_v1.2/" + row[0]  # 确保前缀一致
            row[0] = path + row[0]  # 追加路径前缀
            row[0] = row[0].replace("\\", "/")  # 统一路径分隔符为 '/'
            writer.writerow(row)

    print(f"CSV file {input_file} modification completed")


def balance_csv(input_csv, output_csv):
    """
    使用随机过采样，使 0 和 1 的样本数相等，并保存到新的 CSV 文件。

    参数:
    - input_csv: str, 输入的 CSV 文件路径。
    - output_csv: str, 输出的 CSV 文件路径。
    """
    # 读取 CSV 数据
    with open(input_csv, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取表头
        data = list(reader)  # 读取数据
    
    # 按 label 分类
    data_0 = [row for row in data if row[1] == '0']
    data_1 = [row for row in data if row[1] == '1']

    # 找到较大类别的样本数
    max_samples = max(len(data_0), len(data_1))

    # 过采样（随机复制少数类样本）
    if len(data_0) < len(data_1):
        data_0 = data_0 * (max_samples // len(data_0)) + random.sample(data_0, max_samples % len(data_0))
    else:
        data_1 = data_1 * (max_samples // len(data_1)) + random.sample(data_1, max_samples % len(data_1))

    # 合并数据并打乱顺序
    balanced_data = data_0 + data_1
    random.shuffle(balanced_data)

    # 写入新的 CSV 文件
    with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入表头
        writer.writerows(balanced_data)  # 写入数据

    print(f"已生成均衡数据集：{output_csv}")


# the meta_data.csv file in FakeAVCeleb dataset
csv_file_path = r"E:\downloads\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\meta_data.csv"  # 替换为实际路径

# combination_statistics = count_type_method_combinations(csv_file)

# types = [AV_type, manipulation_type]
types = [["RealVideo-RealAudio", "real"],
         ["RealVideo-FakeAudio", "rtvc"],
         ["FakeVideo-RealAudio", "wav2lip"],
         ["FakeVideo-FakeAudio", "wav2lip"],
         ["FakeVideo-FakeAudio", "faceswap-wav2lip"],
         ["FakeVideo-FakeAudio", "fsgan-wav2lip"]]

# for AV_type, manipulation_type in types:
#     manipulation_types_divide(csv_file_path, AV_type, manipulation_type)

# leave_one_out_test()


# 修改csv中文件路径前缀

# input_dir = "./output_LOCO"
# output_dir = "./output_LOCO_modified"  # 处理后的 CSV 文件存放目录
# os.makedirs(output_dir, exist_ok=True)  # 如果不存在，创建输出目录  

# csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

# for input_file in csv_files:
#     filename = os.path.basename(input_file)  # 获取文件名
#     output_file = os.path.join(output_dir, filename)  # 生成输出文件路径

#     # modify_csv(input_file, output_file, path="/home/home/wangyuxuan/jielun/")  # 处理 CSV 文件
#     modify_csv(input_file, output_file, path="E:/downloads/FakeAVCeleb_v1.2/")



# 数据过采样

input_dir = "./output_LOCO_modified_segmented"  # 处理后的 CSV 文件存放目录
output_dir = "./output_LOCO_modified_segmented_oversampled"
os.makedirs(output_dir, exist_ok=True)  # 如果不存在，创建输出目录  

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

for input_file in csv_files:
    filename = os.path.basename(input_file)  # 获取文件名
    output_file = os.path.join(output_dir, filename)  # 生成输出文件路径
    balance_csv(input_file, output_file)
