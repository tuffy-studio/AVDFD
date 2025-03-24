from moviepy.editor import VideoFileClip
import os

def split_video(input_file, output_dir, segment_length=4, num_segments=4):
    # 加载视频文件
    video = VideoFileClip(input_file)

    # 获取视频的总时长
    video_duration = video.duration
    print(f"Total video duration: {video_duration} seconds")

    # 如果视频长度小于 segment_length * num_segments，则调整 num_segments
    if video_duration < segment_length * num_segments:
        num_segments = int(video_duration // segment_length)

    # 按照 segment_length 将视频分割成多个部分
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = start_time + segment_length

        # 裁剪视频片段
        segment = video.subclip(start_time, min(end_time, video_duration))

        # 生成输出文件名
        output_file = f"{output_dir}/segment_{i + 1}.mp4"

        # 写入文件
        segment.write_videofile(output_file, codec="libx264", audio_codec="aac")
        print(f"Saved segment {i + 1} to {output_file}")

    # 释放资源
    video.close()

if __name__ == "__main__":
    input_video_path = "/home/home/wangyuxuan/jielun/graduation-project/data/demo/demo.mp4"  # 输入文件路径
    output_directory = "/home/home/wangyuxuan/jielun/graduation-project/data/demo/segment_demo"  # 输出文件夹路径
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 调用函数分割视频
    split_video(input_video_path, output_directory)
