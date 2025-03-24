import csv

# 初始化计数器
label_1_count = 0

# 打开 CSV 文件
with open('../data/csv/ft_train_4_segmented_modified.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    # 跳过标题行（如果有的话）
    next(csvreader)  # 如果没有标题行，删除这行

    # 遍历每一行
    for row in csvreader:
        # 假设 CSV 中的第二列是 'label'（索引1），且值为 '1' 时计数
        if row[1] == '0':  # 注意 '1' 是字符串类型
            label_1_count += 1

# 输出结果
print(f"Label为1的样本数: {label_1_count}")