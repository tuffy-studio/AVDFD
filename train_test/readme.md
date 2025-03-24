本文件夹中的文件为模型训练/验证时所使用的文件

## 各个文件/文件夹的作用

### finetune.sh
代替命令行，输入伪造检测阶段所需参数

### pretrain.sh
代替命令行输入，输入音视频表征学习阶段所需参数

### run_finetune.py
解析命令行参数，启动伪造检测训练

### run_pretrain.py
解析命令行参数，启动音视频表征学习训练

### traintest_finetune.py
伪造检测模型训练代码

### traintest_pretrain.py
音视频表征学习模块训练代码

### evaluation.py
对模型进行验证，包括绘制混淆矩阵、AP、AUC曲线等功能