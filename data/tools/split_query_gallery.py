#!/usr/bin/env python3
"""
将bpbreid数据集随机分割为query和gallery
随机选择5000张图片作为query，剩余作为gallery
"""

import random
import shutil
from pathlib import Path


def split_dataset(src_dir, query_dir, gallery_dir, query_count=5000):
    """
    随机分割数据集为query和gallery

    Args:
        src_dir: 源目录路径
        query_dir: query目标目录
        gallery_dir: gallery目标目录
        query_count: query集的图片数量
    """
    if not src_dir.exists():
        print(f"错误: 源目录不存在 {src_dir}")
        return

    # 获取所有图片
    image_files = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
    total_count = len(image_files)

    print(f"源目录: {src_dir}")
    print(f"总图片数: {total_count}")

    if total_count < query_count:
        print(f"警告: 图片总数({total_count})少于query数量({query_count})")
        query_count = total_count

    # 随机打乱
    random.shuffle(image_files)

    # 分割
    query_files = image_files[:query_count]
    gallery_files = image_files[query_count:]

    print(f"\nQuery集: {len(query_files)} 张")
    print(f"Gallery集: {len(gallery_files)} 张")

    # 创建目标目录
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # 复制query图片
    print(f"\n复制到 {query_dir}...")
    for img_path in query_files:
        dst_path = query_dir / img_path.name
        shutil.copy2(img_path, dst_path)

    # 复制gallery图片
    print(f"复制到 {gallery_dir}...")
    for img_path in gallery_files:
        dst_path = gallery_dir / img_path.name
        shutil.copy2(img_path, dst_path)

    print("\n完成！")


def main():
    # 设置随机种子以便复现
    random.seed(42)

    # 定义路径
    base_dir = Path('/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/datasets/DaJixiang')
    src_dir = base_dir / 'train'
    query_dir = base_dir / 'query'
    gallery_dir = base_dir / 'test'

    print("=" * 60)
    print("开始分割数据集为query和gallery")
    print("=" * 60)

    split_dataset(src_dir, query_dir, gallery_dir, query_count=3000)

    print("\n" + "=" * 60)
    print("分割完成！")
    print(f"Query目录: {query_dir}")
    print(f"Gallery目录: {gallery_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
