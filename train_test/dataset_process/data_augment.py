import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# 数据增强管线
def build_augmentation_pipeline(config):
    transforms_list = []

    # 1. 高斯模糊 (Gaussian Blur)
    if config.AUG.USE_GAUSSIAN_BLUR:
        transforms_list.append(
            A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.01, 0.5), p=config.AUG.GAUSSIAN_BLUR_PROB)
        )

    # 2. 高斯噪声 (Gaussian Noise)
    if config.AUG.USE_GAUSSIAN_NOISE:
        transforms_list.append(
            A.GaussNoise(var_limit=(config.AUG.GAUSSIAN_NOISE_SIGMA**2, config.AUG.GAUSSIAN_NOISE_SIGMA**2),
                         p=config.AUG.GAUSSIAN_NOISE_PROB)
        )

    # 3. 颜色抖动 (Color Jitter)
    if config.AUG.USE_COLOR_JITTER:
        transforms_list.append(
            A.ColorJitter(
                p=config.AUG.COLOR_JITTER_PROB,
                brightness=config.AUG.COLOR_JITTER_BRIGHTNESS_RANGE,
                contrast=config.AUG.COLOR_JITTER_CONTRAST_RANGE,
                saturation=config.AUG.COLOR_JITTER_SATURATION_RANGE,
                hue=config.AUG.COLOR_JITTER_HUE_RANGE
            )
        )

    # 4. 锐化 (Sharpen)
    if config.AUG.USE_SHARPEN:
        transforms_list.append(
            A.Sharpen(p=config.AUG.SHARPEN_PROB, 
                      alpha=config.AUG.SHARPEN_ALPHA_RANGE, 
                      lightness=config.AUG.SHARPEN_LIGHTNESS_RANGE)
        )

    # 5. JPEG 压缩 (JPEG Compression)
    if config.AUG.USE_JPEG_COMPRESSION:
        transforms_list.append(
            A.ImageCompression(
                quality_lower=config.AUG.JPEG_MIN_QUALITY,
                quality_upper=config.AUG.JPEG_MAX_QUALITY,
                compression_type=A.ImageCompressionType.JPEG,
                p=config.AUG.JPEG_COMPRESSION_PROB
            )
        )

    # 6. WebP 压缩 (WebP Compression)
    if config.AUG.USE_WEBP_COMPRESSION:
        transforms_list.append(
            A.ImageCompression(
                quality_lower=config.AUG.WEBP_MIN_QUALITY,
                quality_upper=config.AUG.WEBP_MAX_QUALITY,
                compression_type=A.ImageCompressionType.WEBP,
                p=config.AUG.WEBP_COMPRESSION_PROB
            )
        )

    # 最后的转换为 PyTorch 张量
    transforms_list.append(ToTensorV2())

    # 将所有转换组合成一个流水线
    transform = A.Compose(transforms_list)

    return transform

# 示例配置对象
class Config:
    class AUG:
        # 各种增强操作的启用与否
        USE_GAUSSIAN_BLUR = True
        USE_GAUSSIAN_NOISE = True
        USE_COLOR_JITTER = True
        USE_SHARPEN = True
        USE_JPEG_COMPRESSION = True
        USE_WEBP_COMPRESSION = True

        GAUSSIAN_BLUR_PROB = 0.5
        GAUSSIAN_NOISE_PROB = 0.5
        GAUSSIAN_NOISE_SIGMA = 25
        COLOR_JITTER_PROB = 0.5
        COLOR_JITTER_BRIGHTNESS_RANGE = 0.2
        COLOR_JITTER_CONTRAST_RANGE = 0.2
        COLOR_JITTER_SATURATION_RANGE = 0.2
        COLOR_JITTER_HUE_RANGE = 0.1
        SHARPEN_PROB = 0.5
        SHARPEN_ALPHA_RANGE = (0.2, 1.0)
        SHARPEN_LIGHTNESS_RANGE = (0.2, 1.0)
        JPEG_MIN_QUALITY = 50
        JPEG_MAX_QUALITY = 90
        JPEG_COMPRESSION_PROB = 0.7
        WEBP_MIN_QUALITY = 50
        WEBP_MAX_QUALITY = 90
        WEBP_COMPRESSION_PROB = 0.7

# 读取图片，应用增强操作并输出
def apply_augmentations(image_path, config):
    # 读取图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

    # 获取增强管线
    transform = build_augmentation_pipeline(config)

    # 执行增强操作
    augmented = transform(image=image)
    augmented_image = augmented['image']

    # 显示增强后的图片
    augmented_pil = Image.fromarray(augmented_image.astype(np.uint8))
    augmented_pil.show()

    # 保存增强后的图片
    augmented_pil.save("augmented_image.jpg")

if __name__ == "__main__":
    config = Config()
    apply_augmentations('your_image.jpg', config)
