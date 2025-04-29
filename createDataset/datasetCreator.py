import moviepy.editor as mp
import cv2
import os


'''
python -m PyQt5.uic.pyuic UI_createDataset.ui -o UI_createDataset.py
'''

def create_directory1(directory_name):
    try:
        # 检查目录是否已存在
        if not os.path.exists(directory_name):
            # 如果不存在则创建目录
            os.makedirs(directory_name)
            print(f"目录 '{directory_name}' 创建成功！")
        else:
            print(f"目录 '{directory_name}' 已经存在。")
    except OSError as error:
        print(f"创建目录 '{directory_name}' 时出错: {error}")


def create_directory2(relative_path, directory_name):
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 组合相对路径和目标文件夹名称
        target_dir = os.path.join(current_dir, relative_path, directory_name)

        # 检查目标目录是否存在
        if not os.path.exists(target_dir):
            # 如果不存在则创建目录
            os.makedirs(target_dir)
            print(f"目录 '{directory_name}' 在相对路径 '{relative_path}' 下创建成功！")
        else:
            print(f"目录 '{directory_name}' 在相对路径 '{relative_path}' 下已经存在。")
    except OSError as error:
        print(f"创建目录 '{directory_name}' 时出错: {error}")


def create_file(relative_path, file_name):
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 组合相对路径和目标文件名
        target_file_path = os.path.join(current_dir, relative_path, file_name)

        # 检查目标文件是否存在
        if not os.path.exists(target_file_path):
            # 如果不存在则创建空文件
            with open(target_file_path, 'w') as file:
                print(f"文件 '{file_name}' 在相对路径 '{relative_path}' 下创建成功！")
        else:
            print(f"文件 '{file_name}' 在相对路径 '{relative_path}' 下已经存在。")
    except OSError as error:
        print(f"创建文件 '{file_name}' 时出错: {error}")


# Define a function to overlay video B on video A
def overlay_videos(video_a_path, video_b_path, output_path):
    # Load video A and get its properties
    video_a = mp.VideoFileClip(video_a_path)
    fps_a = video_a.fps
    duration_a = video_a.duration
    size_a = video_a.size

    # Load video B, remove the black background, and resize it to match video A's resolution
    video_b = mp.VideoFileClip(video_b_path).fx(mp.vfx.mask_color, color=[0, 0, 0], thr=100, s=5)
    video_b_resized = video_b.resize(newsize=size_a)

    # If video B is longer than video A, cut it to match video A's duration
    if video_b_resized.duration > duration_a:
        video_b_resized = video_b_resized.subclip(0, duration_a)
    # If video B is shorter than video A, loop it to match video A's duration
    else:
        video_b_resized = mp.concatenate_videoclips([video_b_resized] * int(duration_a // video_b_resized.duration + 1))
        video_b_resized = video_b_resized.subclip(0, duration_a)

    # Overlay video B on video A
    final_video = mp.CompositeVideoClip([video_a, video_b_resized.set_position(("center", "center"))])

    # Write the result to the output file
    final_video.write_videofile(output_path, fps=fps_a)


def save_frames(norain_video, rain_video, save_pth_1, save_pth_2, data, X):
    cap1 = cv2.VideoCapture(norain_video)
    cap2 = cv2.VideoCapture(rain_video)
    i = 1
    counter = 1
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        if i % X == 0:
            cv2.imwrite(save_pth_1 + str(counter) + '.jpg', frame1)
            cv2.imwrite(save_pth_2 + str(counter) + '.jpg', frame2)
            with open(data, 'a') as f:
                f.write(save_pth_1 + str(counter) + '.jpg' + '#' + save_pth_2 + str(counter) + '.jpg' + '\n')
            counter += 1
            print("saving frame " + str(counter))
        i += 1
    print("finished saving " + str(counter) + " frames")
    cap1.release()
    cap2.release()


dataset_name = "sight_and_squirrel"

norain_video = 'sight_and_squirrel_src.mp4'
rain_video = 'sight_and_squirrel.mp4'

make_out_video = 0

frame_step = 5

rain_video = 'heavy_rain.mp4'
cur_path = "./source_video/"
to_path = "./compositioned_video/"

data_pth = "./" + dataset_name + "/"
data_file = "train_data.txt"
norain_pth = "norain/"
rain_pth = "rain/"

create_directory1(dataset_name)
create_directory2(data_pth, "rain")
create_directory2(data_pth, "norain")

create_file(data_pth, data_file)
# 函数输入参数分别为：源视频、雨慕视频、输出视频 的路径
if make_out_video:
    overlay_videos(cur_path + norain_video, cur_path + rain_video, to_path + rain_video)

save_frames(to_path + rain_video, cur_path + norain_video, data_pth + rain_pth, data_pth + norain_pth, data_pth + data_file, frame_step)
