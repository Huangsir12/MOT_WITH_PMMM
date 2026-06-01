import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def visualize_sequence(image_dir, label_file, output_dir,
                        seq_name, image_files, fps=30, frame_width=1920, frame_height=1080, class_names=None):
    """
    可视化单个序列并生成视频
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出视频路径
    output_video_path = os.path.join(output_dir, f"{seq_name}.mp4")
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 处理标注文件
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
    else:
        print(f"标注文件 {label_file} 不存在!")
        return
    
    # 处理每一帧
    for img_file in tqdm(image_files, desc=f"Processing {seq_name}"):
        # 读取图片
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        for line in lines:
            parts = line.strip().split(",")
            if int(parts[0]) == int(img_file.split(".")[0]):
                track_id = int(parts[1])
                class_id = int(parts[7])
                bbox = list(map(float, parts[2:6]))
                x1, y1, w, h = bbox
                # 转换bbox坐标
                xmin, ymin, xmax, ymax = int(x1), int(y1), int(x1 + w), int(y1 + h)
                # 获取类别名称
                label = class_names[class_id] if class_names else str(class_id)
                text = f"{track_id} {label}"
                # 绘制边框 
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (220, 80, 40), 3)
                # 绘制文本
                cv2.rectangle(image, (xmin, ymin - 60), (xmin + 320, ymin), (220, 160, 100), -1) 
                cv2.putText(image, text, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        # 写入视频帧
        video_writer.write(image)
    
    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存到: {output_video_path}")

def visualize_all_sequences(data_dir, output_dir="output_videos", class_names=None):
    """
    可视化所有视频序列，每个序列生成一个单独的视频
    """
    for dir in os.listdir(data_dir):
    # 获取所有图片文件
        print(f"正在处理视频序列: {os.path.join(data_dir, dir)}")
        if dir == ".ipynb_checkpoints":
            continue
        image_dir = os.path.join(data_dir, dir, "img1")  # 图片文件夹路径
        label_file = os.path.join(data_dir, dir, "gt", "gt.txt")  # 标注文件夹路径
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort(key=lambda x: int(x.split(".")[0]))  # 按照文件名排序
        info_flie  = os.path.join(data_dir, dir, "seqinfo.ini")

        if not image_files:
            print("没有找到图片文件!")
            return
        print(f"视频序列{dir}找到 {len(image_files)}张图片")

        with open(info_flie, 'r') as file:
            lines = file.readlines()
        dataname = str(lines[1].strip().split(" ")[-1])
        output_dir = os.path.join(output_dir, dataname)
        fps = int(lines[3].split(" ")[-1])
        image_width = int(lines[5].split(" ")[-1])
        image_heigth = int(lines[6].split(" ")[-1])
        print(f"视频序列{dir}的fps为{fps}, 宽度为{image_width}, 高度为{image_heigth}")
        visualize_sequence(
            image_dir=image_dir,
            label_file=label_file,
            output_dir=output_dir,
            seq_name=f"{dir}",
            image_files=image_files,
            fps=fps,
            frame_width = image_width,
            frame_height = image_heigth,
            class_names=class_names
        )

# 使用示例
if __name__ == "__main__":
    # 设置路径
    data_dir = "data/datasets/Emporium/train"  # 图片文件夹路径
    output_dir = "runs/video/track"  # 输出视频目录
    
    # 类别名称 (根据你的数据集修改)
    class_names = ["cat", "person"]
    
    # 可视化所有序列
    visualize_all_sequences(
        data_dir=data_dir,
        output_dir=output_dir,
        class_names=class_names
    )

