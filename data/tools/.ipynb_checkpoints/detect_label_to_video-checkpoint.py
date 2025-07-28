import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# MOT17数据集的帧宽高
frame_wh_json = {"MOT17-01": [1920, 1080],
                 "MOT17-02": [1920, 1080],
                 "MOT17-03": [1920, 1080],
                 "MOT17-04": [1920, 1080],
                 "MOT17-05": [640, 480],
                 "MOT17-06": [640, 480],
                 "MOT17-07": [1920, 1080],
                 "MOT17-08": [1920, 1080],
                 "MOT17-09": [1920, 1080],
                 "MOT17-10": [1920, 1080],
                 "MOT17-11": [1920, 1080],
                 "MOT17-12": [1920, 1080],
                 "MOT17-13": [1920, 1080],
                 "MOT17-14": [1920, 1080]}


def yolo_to_pixels(bbox, img_width, img_height):
    """
    将YOLO格式的bbox(中心x,中心y,宽度,高度)转换为像素坐标(xmin,ymin,xmax,ymax)
    """
    x_center, y_center, w, h = bbox
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    xmin = int(x_center - w / 2)
    ymin = int(y_center - h / 2)
    xmax = int(x_center + w / 2)
    ymax = int(y_center + h / 2)
    return xmin, ymin, xmax, ymax

def group_files_by_sequence(files):
    """
    将文件按视频序列分组
    返回: 字典 {序列名: [排序后的文件列表]}
    """
    sequences = {}
    for f in files:
        # 分割序列名和帧号 (如 "MOT17-13_000376.jpg" -> "MOT17-13", "000376")
        parts = f.split('_')
        if len(parts) >= 2:
            seq_name = '_'.join(parts[:-1])  # 处理可能包含多个下划线的情况
            frame_num = parts[-1].split('.')[0]
            if seq_name not in sequences:
                sequences[seq_name] = []
            sequences[seq_name].append((int(frame_num), f))
    
    # 对每个序列的文件按帧号排序
    for seq in sequences:
        sequences[seq].sort(key=lambda x: x[0])
        sequences[seq] = [f[1] for f in sequences[seq]]  # 只保留文件名
    
    return sequences

def visualize_sequence(image_dir, label_dir, output_dir, seq_name, image_files, fps=30, class_names=None):
    """
    可视化单个序列并生成视频
    """
    if not image_files:
        print(f"序列 {seq_name} 没有图片文件!")
        return
    
    # 获取第一张图片的尺寸
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"无法读取图片: {first_image_path}")
        return
    
    if seq_name in frame_wh_json:
        frame_width, frame_height = frame_wh_json[seq_name]
    else:
        frame_width, frame_height = 1920, 1080
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出视频路径
    output_video_path = os.path.join(output_dir, f"{seq_name}.mp4")
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 处理每一帧
    for img_file in tqdm(image_files, desc=f"Processing {seq_name}"):
        # 读取图片
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 对应的标注文件路径
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    
                    # 转换bbox坐标
                    xmin, ymin, xmax, ymax = yolo_to_pixels(bbox, frame_width, frame_height)
                    # 获取类别名称
                    label = class_names[class_id] if class_names else str(class_id)
                    text = f"{class_id} {label}"
                    # 绘制边框 
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                    # 绘制文本
                    cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        # 写入视频帧
        video_writer.write(image)
    
    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存到: {output_video_path}")

def visualize_all_sequences(image_dir, label_dir, output_dir="output_videos", fps=30, class_names=None):
    """
    可视化所有视频序列，每个序列生成一个单独的视频
    """
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("没有找到图片文件!")
        return
    
    # 按序列分组
    sequences = group_files_by_sequence(image_files)
    
    print(f"找到 {len(sequences)} 个视频序列: {list(sequences.keys())}")
    
    # 处理每个序列
    for seq_name, seq_files in sequences.items():
        visualize_sequence(
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            seq_name=seq_name,
            image_files=seq_files,
            fps=fps,
            class_names=class_names
        )

# 使用示例
if __name__ == "__main__":
    # 设置路径
    image_dir = "data/datasets/MOT17_YOLO/train/images"  # 图片文件夹路径
    label_dir = "data/datasets/MOT17_YOLO/train/labels"  # 标注文件夹路径
    output_dir = "runs/video/det/MOT17-val_2025-0415-1733_train"  # 输出视频目录
    
    # 类别名称 (根据你的数据集修改)
    class_names = ["person"]
    
    # 可视化所有序列
    visualize_all_sequences(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        fps=30,
        class_names=class_names
    )

