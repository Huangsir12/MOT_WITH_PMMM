"""
跟踪结果可视化脚本
将MOT格式的跟踪结果txt文件叠加到视频上生成可视化视频

MOT格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

用法:
python track_result_to_video.py --video input.mp4 --txt results.txt --output output.mp4
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def load_tracking_results(txt_path):
    """
    加载MOT格式跟踪结果

    Args:
        txt_path: 跟踪结果txt文件路径

    Returns:
        dict: {frame_id: [(track_id, bbox, conf, class_id), ...]}
    """
    if not Path(txt_path).exists():
        raise FileNotFoundError(f"跟踪结果文件不存在: {txt_path}")

    results = defaultdict(list)

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7]) if len(parts) > 7 else 0

            bbox = (x, y, w, h)
            results[frame_id].append((track_id, bbox, conf, class_id))

    return results


def generate_colors(num_colors):
    """生成不同的颜色用于区分不同的track ID"""
    np.random.seed(42)
    colors = {}
    for i in range(num_colors):
        colors[i] = tuple(map(int, np.random.randint(0, 255, 3)))
    return colors


def draw_tracking_results(frame, detections, colors, class_names=None, show_conf=True):
    """
    在帧上绘制跟踪结果

    Args:
        frame: 视频帧
        detections: [(track_id, bbox, conf, class_id), ...]
        colors: ID颜色映射
        class_names: 类别名称列表
        show_conf: 是否显示置信度
    """
    for track_id, bbox, conf, class_id in detections:
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        # 获取颜色
        color = colors.get(track_id % len(colors), (0, 255, 0))

        # 绘制边框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"cls{class_id}"
        if show_conf:
            label = f"ID:{track_id} {class_name} {conf:.2f}"
        else:
            label = f"ID:{track_id} {class_name}"

        # 绘制标签背景
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # 绘制标签文本
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return frame


def visualize_tracking(video_path, txt_path, output_path, class_names=None, show_conf=True):
    """
    可视化跟踪结果

    Args:
        video_path: 输入视频路径
        txt_path: 跟踪结果txt文件路径
        output_path: 输出视频路径
        class_names: 类别名称列表
        show_conf: 是否显示置信度
    """
    # 检查输入文件
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 加载跟踪结果
    print(f"加载跟踪结果: {txt_path}")
    tracking_results = load_tracking_results(txt_path)
    print(f"加载了 {len(tracking_results)} 帧的跟踪结果")

    # 获取所有唯一的track ID用于生成颜色
    all_track_ids = set()
    for detections in tracking_results.values():
        for track_id, _, _, _ in detections:
            all_track_ids.add(track_id)

    print(f"唯一track ID数: {len(all_track_ids)}")
    colors = generate_colors(max(all_track_ids) + 1 if all_track_ids else 100)

    # 打开视频
    print(f"打开视频: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频属性: {frame_width}x{frame_height} @ {fps}fps, 总帧数: {total_frames}")

    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # 处理每一帧
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="处理视频帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 获取当前帧的跟踪结果
        detections = tracking_results.get(frame_id, [])

        # 绘制跟踪结果
        if detections:
            frame = draw_tracking_results(frame, detections, colors, class_names, show_conf)

        # 在左上角显示帧号和检测数
        info_text = f"Frame: {frame_id} | Detections: {len(detections)}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # 写入视频帧
        video_writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    video_writer.release()

    print(f"\n视频已保存到: {output_path}")
    print(f"处理了 {frame_id} 帧")


def main():
    parser = argparse.ArgumentParser(
        description='将MOT格式跟踪结果可视化到视频上',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python track_result_to_video.py --video input.mp4 --txt results.txt --output output.mp4

  # 指定类别名称
  python track_result_to_video.py --video input.mp4 --txt results.txt --output output.mp4 --classes person car

  # 不显示置信度
  python track_result_to_video.py --video input.mp4 --txt results.txt --output output.mp4 --no-conf
        """
    )

    parser.add_argument('--video', '-v', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1759/camera_002_0.mp4",
                       help='输入视频路径')
    parser.add_argument('--txt', '-t', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1759/camera_002_0_xiugai.txt",
                       help='MOT格式跟踪结果txt文件路径')
    parser.add_argument('--output', '-o', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1759/camera_002_0_result.mp4",
                       help='输出视频路径')
    parser.add_argument('--classes', nargs='+', type=str, default=["person"],
                       help='类别名称列表，例如: --classes person car bike')
    parser.add_argument('--no-conf', action='store_true',
                       help='不显示置信度')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("跟踪结果可视化")
    print(f"{'='*60}")
    print(f"输入视频: {args.video}")
    print(f"跟踪结果: {args.txt}")
    print(f"输出视频: {args.output}")
    if args.classes:
        print(f"类别名称: {args.classes}")
    print(f"显示置信度: {not args.no_conf}")
    print(f"{'='*60}\n")

    visualize_tracking(
        video_path=args.video,
        txt_path=args.txt,
        output_path=args.output,
        class_names=args.classes,
        show_conf=not args.no_conf
    )


if __name__ == '__main__':
    main()
