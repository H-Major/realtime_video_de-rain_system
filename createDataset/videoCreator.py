import os
import cv2


def images_to_video(dir_name, output_name, x, fps):
    # 获取目录中的所有文件
    images = [img for img in os.listdir(dir_name) if img.lower().endswith(".jpg")]
    images.sort()  # 按文件名排序

    if not images:
        print("目录中没有图片文件")
        return

    print(f"找到 {len(images)} 张图片")

    # 目标视频尺寸
    target_width, target_height = 1920, 1080
    print(f"目标视频尺寸: {target_width}x{target_height}")

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    video = cv2.VideoWriter(output_name, fourcc, fps, (target_width, target_height))

    total_frames = 0

    for image in images:
        img_path = os.path.join(dir_name, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片文件: {img_path}")
            continue
        # 调整图片大小
        resized_img = cv2.resize(img, (target_width, target_height))
        for _ in range(x):
            video.write(resized_img)
            total_frames += 1
            if total_frames % 100 == 0:
                print(f"已处理 {total_frames} 帧")

    video.release()
    print(f"视频已保存到 {output_name}")
    print(f"预期总帧数: {total_frames}")

    # 检查生成视频的实际帧数
    cap = cv2.VideoCapture(output_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"实际视频帧数: {frame_count}")


# 示例调用
dir_name = './images'
output_name = 'dash_board_norain_30f.mp4'
x = 30  # 每张图片连续1帧
fps = 30  # 每秒30帧

images_to_video(dir_name, output_name, x, fps)
