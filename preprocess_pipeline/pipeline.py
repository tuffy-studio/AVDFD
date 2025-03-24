import os
import math
import numpy as np
from PIL import Image
import cv2
import yaml
import mediapipe as mp
import torchaudio
torchaudio.set_audio_backend("librosa")
import ffmpeg
# ffmpeg_path = r"C:\Users\23950\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin\ffmpeg.exe"
import numpy as np
import torch
from moviepy import VideoFileClip
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
# 屏蔽 TensorFlow 的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: 全部信息, 1: 只显示 ERROR, 2: ERROR + WARNING, 3: 只显示 FATAL
from absl import logging
logging.set_verbosity(logging.ERROR)
import matplotlib.pyplot as plt
import sys
# 获取当前脚本所在的目录，添加环境变量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,"FaceX_Zoo/face_sdk"))
sys.path.append(os.path.join(BASE_DIR,"FaceX_Zoo/face_sdk/models/network_def"))




from FaceX_Zoo.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from FaceX_Zoo.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler




def split_video(input_file, segment_length=3.20):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    video = VideoFileClip(input_file)
    duration = video.duration
    num_segments = math.ceil(duration / segment_length)

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = start_time + segment_length

        if end_time > duration:
            end_time = duration

        print(f"segment {i+1} in {num_segments}: {start_time}-{end_time} in {duration}")
        segment = video.subclipped(start_time, end_time)

        output_file = f"{base_name}_part_{i + 1}.mp4"
        segment.write_videofile(output_file, codec="libx264", audio_codec="aac")


def extract_frames(input_file, num_frames=16, save_frames=False):
    # 加载视频文件
    video = VideoFileClip(input_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Video file '{input_file}' not found.")
    # 获取视频的时长和帧率
    duration = video.duration  # 视频时长（单位：秒）
    fps = video.fps  # 每秒帧数（FPS）
    # 计算总帧数
    total_frames = int(duration * fps)

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    #print(frame_indices)
    frames = []
    for idx in frame_indices:
        frame = video.reader.get_frame(idx/video.fps)  # 直接按帧索引获取
        frames.append(frame)
    # frames 是以 NumPy 数组 的形式返回的，表示视频的单帧图像
    if save_frames == True:
        save_dir = "./frames/"
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            # 将 NumPy 数组转换为 PIL 图像对象
            image = Image.fromarray(frame.astype('uint8'))  # 转换为 uint8 类型以确保像素值在0-255之间

            # 保存图片
            output_path = os.path.join(save_dir, f"frame_{i + 1}.jpg")
            image.save(output_path)
    print(frames[0].shape)
    return frames





def load_face_detection_model():
    # 生成绝对路径，确保无论在哪个目录调用，路径都正确
    config_path = os.path.join(BASE_DIR, 'FaceX_Zoo', 'face_sdk', 'config', 'model_conf.yaml')
    model_path = os.path.join(BASE_DIR, 'FaceX_Zoo', 'face_sdk', 'models')


    with open(config_path, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.SafeLoader)

    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]

    # 加载模型
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name, meta_file="model_meta.json")
    print("OK")
    model, cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)

    print("Face detection model loaded successfully.")
    return faceDetModelHandler



# 处理图片，提取脸部区域
def extract_face_regions(frames, faceDetModelHandler, show=False):
    face_frames = []

    for frame in frames:
        frame = frame.copy()
        try:
            dets = faceDetModelHandler.inference_on_image(frame)


            for box in dets:
                x_min, y_min, x_max, y_max = map(int, box[:4])

                # 裁剪人脸区域
                face_frame = frame[y_min:y_max, x_min:x_max]
                face_frames.append(face_frame)

                # 可视化检测结果
                if show:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Detected Face", frame_bgr)
                    cv2.waitKey(0)

        except Exception as e:
            print(f"Error processing frame: {e}")

    if show:
        cv2.destroyAllWindows()

    return face_frames


# 定义嘴唇关键点索引
MOUTH_EXTRA_POINTS = [1, 207, 427, 200, 287, 92, 165, 167, 164, 393, 391, 322, 410, 406, 182, 106, 273, 12, 57, 186, 61, 62, 185, 184, 183,
                      191, 40, 74, 42, 80, 39, 73, 41, 81, 37, 72, 38, 82, 0, 11, 12, 13, 267, 302, 268, 312, 269, 303, 271, 311, 270, 43, 321, 320, 404, 405, 314,
                      315, 16, 17, 84, 85, 180, 181, 90, 91, 146, 83, 18, 313, 335,5]


def extract_mouth_region(frame, show=False):
    mp_face_mesh = mp.solutions.face_mesh
    # 初始化 Face Mesh 模型
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # 转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                IAMGE_DIMENSIONS = (w, h)
                points = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]

                # 提取嘴唇区域的坐标
                mouth_points = [points[idx] for idx in MOUTH_EXTRA_POINTS]
                x_coords = [p[0] for p in mouth_points]
                y_coords = [p[1] for p in mouth_points]
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

                # 裁剪嘴唇区域
                mouth_region = frame[y_min:y_max, x_min:x_max]
        
                if show==True:
                    mouth_region_bgr = cv2.cvtColor(mouth_region, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("./frames/month.region.jpg", mouth_region_bgr)
                    cv2.imshow("Mouth Region", mouth_region_bgr)
                    cv2.waitKey(0)  # 等待按键操作

        return mouth_region


def extract_audio_from_video(video_file, output_audio_file):
    """从.mp4文件中提取音频并保存为.wav格式"""
    # 使用 ffmpeg 从视频中提取音频并保存为 wav 格式
    ffmpeg.input(video_file).output(output_audio_file, ac=1, ar='16k').run()  # ac=1 表示单声道，ar='16k' 设置采样率
    # print(f"Audio successfully extracted to {output_audio_file}.")


def wav2fbank(filename, melbins=128, target_length=1024):
    # 如果文件是 .mp4 格式，先提取音频
    if filename.endswith('.mp4'):
        temp_audio_file = filename.replace('.mp4', '.wav')
        if not os.path.exists(temp_audio_file):
            extract_audio_from_video(filename, temp_audio_file)
        else:
            pass
           # print(f"Audio file {temp_audio_file} already exists.")
        filename = temp_audio_file

    # 加载音频文件
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()

    try:
        # 尝试提取梅尔频率倒谱系数（MFCC）
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10)
    except Exception as e:
        # 捕获具体异常并输出错误信息
        print(f"Error in loading audio or computing fbank: {e}")
        # 返回默认的 fbank 值以避免崩溃
        fbank = torch.zeros([512, 128]) + 0.01
        print('There was an error loading the fbank. Returning default tensor.')

    # 调整 fbank 的大小到目标长度
    fbank = torch.nn.functional.interpolate(
        fbank.unsqueeze(0).transpose(1, 2), size=(target_length,),
        mode='linear', align_corners=False).transpose(1, 2).squeeze(0)

    # 处理完后删除临时 .wav 文件
    # if temp_audio_file and os.path.exists(temp_audio_file):
       # os.remove(temp_audio_file)
    
    return fbank


def plot_fbank(fbank):
    """绘制梅尔谱图"""
    plt.figure(figsize=(10, 4))
    plt.imshow(fbank.numpy().T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Filter Banks")
    plt.title("Mel Spectrogram")
    plt.show()


if __name__ == '__main__':
    split_video("./lqd.mp4")
    frames = extract_frames("../data/lqd_part_1.mp4")
    model = load_face_detection_model()
    frames=extract_face_regions(frames,model)
    for frame in frames:
        extract_mouth_region(frame, show=True)
    plot_fbank(wav2fbank("../data/lqd_part_1.mp4"))





    
