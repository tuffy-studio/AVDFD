import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载 .wav 文件
file_path = 'segment_3.wav'  # 替换为你的文件路径
audio, sr = librosa.load(file_path, sr=None)  # sr=None 保持原始采样率

# 创建一个图形窗口并绘制波形图
plt.figure(figsize=(10, 6))
librosa.display.waveshow(audio, sr=sr, alpha=0.7)

# 设置标题和标签
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

# 保存波形图为文件
output_path = 'waveform_output.png'  # 保存的文件路径
plt.savefig(output_path, bbox_inches='tight')

# 输出保存成功的消息
print(f"Waveform saved to {output_path}")