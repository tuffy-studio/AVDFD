import numpy as np
import matplotlib.pyplot as plt
import wave


def plot_waveform(wav_file, start_time=0, end_time=None, save_path=None):
    """
    将 .wav 文件的指定时间范围内音频可视化为波形图。

    参数：
    - wav_file: str，.wav 文件的路径。
    - start_time: float，起始时间（秒），默认从 0 秒开始。
    - end_time: float，可选，结束时间（秒），默认到音频结束。
    - save_path: str，可选，图片保存路径（包括文件名和扩展名）。默认不保存。

    返回：
    - None
    """
    try:
        # 打开 .wav 文件
        with wave.open(wav_file, 'rb') as wav:
            # 提取基本参数
            n_channels = wav.getnchannels()  # 通道数
            sample_width = wav.getsampwidth()  # 采样宽度
            framerate = wav.getframerate()  # 采样率
            n_frames = wav.getnframes()  # 总帧数

            # 转换起止时间为帧数
            start_frame = int(start_time * framerate)
            end_frame = int(end_time * framerate) if end_time else n_frames

            # 检查范围有效性
            if start_frame < 0 or start_frame >= n_frames:
                raise ValueError("起始时间超出音频范围")
            if end_frame < 0 or end_frame > n_frames:
                raise ValueError("结束时间超出音频范围")
            if start_frame >= end_frame:
                raise ValueError("起始时间必须小于结束时间")

            # 跳转到起始帧并读取所需帧
            wav.setpos(start_frame)
            frames_to_read = end_frame - start_frame
            frames = wav.readframes(frames_to_read)

            # 将音频数据转换为 numpy 数组
            if sample_width == 1:  # 8-bit audio
                dtype = np.uint8
            elif sample_width == 2:  # 16-bit audio
                dtype = np.int16
            else:
                raise ValueError("不支持的采样宽度")

            audio_data = np.frombuffer(frames, dtype=dtype)

            # 如果是立体声（双通道），只取第一个通道的数据
            if n_channels > 1:
                audio_data = audio_data[::n_channels]

            # 创建时间轴
            time = np.linspace(start_time, start_time + frames_to_read / framerate, num=frames_to_read)

            # 绘制波形
            plt.figure(figsize=(12, 6))
            plt.plot(time, audio_data, color='blue')
            plt.title(
                f"Audio Waveform ({start_time:.2f}s to {end_time:.2f}s)" if end_time else f"Audio Waveform ({start_time:.2f}s to end)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.grid()

            # 保存或显示波形图
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Waveform saved to {save_path}")

            plt.show()

    except wave.Error as e:
        print(f"Error reading .wav file: {e}")
    except FileNotFoundError:
        print("The specified .wav file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 示例调用
plot_waveform('../demo/segment_demo/segment_3.wav', start_time=2, end_time=2.1, save_path='./utils_data/waveform.png')

