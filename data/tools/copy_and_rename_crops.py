#!/usr/bin/env python3
"""
复制并重命名crops图片到bpbreid数据集
逐个相机复制，每次复制后对目标文件夹所有图片重命名
"""

import os
import json
import shutil
from pathlib import Path


def load_mapping(json_path):
    """加载JSON映射文件"""
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    # 将键转换为整数以便匹配
    return {int(k): v for k, v in mapping.items()}


def copy_camera_images(camera_name, src_base, dst_base):
    """
    复制相机图片到目标文件夹（保持原名）

    Args:
        camera_name: 相机名称
        src_base: 源目录基础路径
        dst_base: 目标目录基础路径

    Returns:
        复制的图片数量
    """
    src_dir = src_base / camera_name

    if not src_dir.exists():
        print(f"  警告: 源目录不存在 {src_dir}")
        return 0

    # 确保目标目录存在
    dst_base.mkdir(parents=True, exist_ok=True)

    # 复制所有图片
    image_files = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
    count = 0
    for img_path in image_files:
        dst_path = dst_base / img_path.name
        shutil.copy2(img_path, dst_path)
        count += 1

    return count


def rename_all_images_in_folder(dst_base, mapping):
    """
    对目标文件夹中的所有图片根据映射重命名
    未映射的ID从5000开始重新编号

    Args:
        dst_base: 目标目录路径
        mapping: ID映射字典

    Returns:
        (映射重命名数量, 未映射重编号数量)
    """
    if not dst_base.exists():
        print(f"  警告: 目标目录不存在 {dst_base}")
        return 0, 0

    # 获取所有图片
    image_files = list(dst_base.glob('*.jpg')) + list(dst_base.glob('*.png'))

    # 第一遍：收集所有需要重命名的信息
    rename_plan = []  # (old_path, new_name, old_id)
    unmapped_ids = set()

    for img_path in image_files:
        old_name = img_path.name
        parts = old_name.split('_')

        if len(parts) < 4:
            continue

        try:
            old_id = int(parts[0])

            if old_id in mapping:
                # 存在映射
                new_id = mapping[old_id]
                parts[0] = f"{new_id:04d}"
                new_name = '_'.join(parts)
                rename_plan.append((img_path, new_name, old_id, True))
            else:
                # 不存在映射，标记为需要重新编号
                unmapped_ids.add(old_id)
                rename_plan.append((img_path, old_name, old_id, False))
        except ValueError:
            continue

    # 为未映射的ID分配新编号（从5000开始）
    unmapped_id_mapping = {}
    next_id = 5000
    for old_id in sorted(unmapped_ids):
        unmapped_id_mapping[old_id] = next_id
        next_id += 1

    # 第二遍：更新未映射ID的新名称
    final_rename_plan = []
    for img_path, name, old_id, is_mapped in rename_plan:
        if not is_mapped and old_id in unmapped_id_mapping:
            parts = name.split('_')
            new_id = unmapped_id_mapping[old_id]
            parts[0] = f"{new_id:04d}"
            name = '_'.join(parts)
        final_rename_plan.append((img_path, name))

    # 执行重命名（使用临时文件名避免冲突）
    temp_suffix = '.tmp_rename'
    mapped_count = 0
    unmapped_count = 0

    # 第一步：重命名为临时文件名
    temp_renames = []
    for img_path, new_name in final_rename_plan:
        if img_path.name != new_name:
            temp_path = img_path.parent / (new_name + temp_suffix)
            img_path.rename(temp_path)
            temp_renames.append((temp_path, new_name))

            # 统计
            parts = img_path.name.split('_')
            old_id = int(parts[0])
            if old_id in mapping:
                mapped_count += 1
            else:
                unmapped_count += 1

    # 第二步：去掉临时后缀
    for temp_path, new_name in temp_renames:
        final_path = temp_path.parent / new_name
        temp_path.rename(final_path)

    return mapped_count, unmapped_count


def main():
    # 定义路径
    base_dir = Path('/root/autodl-tmp/MOT_WITH_PMMM/data/datasets')
    src_base = base_dir / 'bpbreid_dajixiang' / 'crops_202603'
    dst_base = base_dir / 'bpbreid_dajixiang' / 'bpbreid'
    link_label_dir = base_dir / 'link_dajixiang' / 'link_label'

    # 定义相机和对应的JSON映射文件
    camera_mappings = [
        ('camera_001', 'camera001_002.json'),
        ('camera_002', 'camera002_003.json'),
        ('camera_003', 'camera003_004.json'),
        ('camera_004', 'camera004_005.json'),
        ('camera_005', 'camera005_006.json'),
    ]

    print("=" * 60)
    print("开始处理图片复制和重命名")
    print("=" * 60)

    total_copied = 0

    # 处理 camera_001 到 camera_005
    for camera_name, json_file_name in camera_mappings:
        print(f"\n处理 {camera_name}:")

        # 步骤1: 复制图片
        count = copy_camera_images(camera_name, src_base, dst_base)
        print(f"  复制图片: {count}")
        total_copied += count

        # 步骤2: 根据映射重命名目标文件夹所有图片
        json_file = link_label_dir / json_file_name
        if json_file.exists():
            mapping = load_mapping(json_file)
            print(f"  映射文件: {json_file_name}")
            print(f"  映射数量: {len(mapping)}")

            mapped_count, unmapped_count = rename_all_images_in_folder(dst_base, mapping)
            print(f"  映射重命名: {mapped_count}")
            print(f"  未映射重编号: {unmapped_count} (从5000开始)")
        else:
            print(f"  警告: JSON映射文件不存在 {json_file}")

    # 处理 camera_006（直接复制，不重命名）
    print(f"\n处理 camera_006:")
    count = copy_camera_images('camera_006', src_base, dst_base)
    print(f"  直接复制: {count}")
    total_copied += count

    print("\n" + "=" * 60)
    print(f"处理完成！总共复制 {total_copied} 张图片")
    print(f"目标目录: {dst_base}")
    print("=" * 60)


if __name__ == '__main__':
    main()
