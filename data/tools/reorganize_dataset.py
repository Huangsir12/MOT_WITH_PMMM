"""
数据集重组脚本
将DaJixiang数据集的query和test中的图片随机移动到train中

处理流程:
1. 从query中随机选择3000张图片移动到train
2. 从test中随机选择6000张图片移动到train
3. 对masks/pifpaf和masks/pifpaf_maskrcnn_filtering中的对应文件进行相同操作
4. 记录所有移动的文件名

用法:
python reorganize_dataset.py --dataset-root /path/to/DaJixiang --dry-run
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def get_all_files(directory):
    """获取目录中的所有文件"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def move_files(src_dir, dst_dir, file_list, desc="Moving files", suffix_transform=None):
    """
    移动文件列表

    Args:
        src_dir: 源目录
        dst_dir: 目标目录
        file_list: 文件名列表
        desc: 进度条描述
        suffix_transform: 后缀转换函数，例如 lambda x: x + '.npy'
    """
    os.makedirs(dst_dir, exist_ok=True)

    moved_count = 0
    missing_count = 0

    for filename in tqdm(file_list, desc=desc):
        # 应用后缀转换
        if suffix_transform:
            src_filename = suffix_transform(filename)
        else:
            src_filename = filename

        src_path = os.path.join(src_dir, src_filename)
        dst_path = os.path.join(dst_dir, src_filename)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            moved_count += 1
        else:
            missing_count += 1
            print(f"  警告: 文件不存在 {src_path}")

    return moved_count, missing_count


def reorganize_dataset(dataset_root, query_count=3000, test_count=6000, dry_run=False):
    """
    重组数据集

    Args:
        dataset_root: 数据集根目录
        query_count: 从query移动的图片数量
        test_count: 从test移动的图片数量
        dry_run: 是否仅预览不实际移动
    """
    dataset_root = Path(dataset_root)

    # 定义目录路径
    query_dir = dataset_root / "query"
    test_dir = dataset_root / "test"
    train_dir = dataset_root / "train"

    pifpaf_query_dir = dataset_root / "masks" / "pifpaf" / "query"
    pifpaf_test_dir = dataset_root / "masks" / "pifpaf" / "test"
    pifpaf_train_dir = dataset_root / "masks" / "pifpaf" / "train"

    maskrcnn_query_dir = dataset_root / "masks" / "pifpaf_maskrcnn_filtering" / "query"
    maskrcnn_test_dir = dataset_root / "masks" / "pifpaf_maskrcnn_filtering" / "test"
    maskrcnn_train_dir = dataset_root / "masks" / "pifpaf_maskrcnn_filtering" / "train"

    # 检查目录是否存在
    print(f"\n{'='*60}")
    print("检查目录")
    print(f"{'='*60}")

    required_dirs = [
        query_dir, test_dir,
        pifpaf_query_dir, pifpaf_test_dir,
        maskrcnn_query_dir, maskrcnn_test_dir
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")
        print(f"✓ {dir_path}")

    # 获取所有文件
    print(f"\n{'='*60}")
    print("统计文件数量")
    print(f"{'='*60}")

    query_files = get_all_files(query_dir)
    test_files = get_all_files(test_dir)

    print(f"query目录文件数: {len(query_files)}")
    print(f"test目录文件数: {len(test_files)}")

    # 检查数量是否足够
    if len(query_files) < query_count:
        raise ValueError(f"query目录文件数({len(query_files)})少于需要移动的数量({query_count})")

    if len(test_files) < test_count:
        raise ValueError(f"test目录文件数({len(test_files)})少于需要移动的数量({test_count})")

    # 随机选择文件
    print(f"\n{'='*60}")
    print("随机选择文件")
    print(f"{'='*60}")

    random.seed(42)  # 设置随机种子以保证可重复性

    selected_query_files = random.sample(query_files, query_count)
    selected_test_files = random.sample(test_files, test_count)

    print(f"从query选择: {len(selected_query_files)} 个文件")
    print(f"从test选择: {len(selected_test_files)} 个文件")
    print(f"总计: {len(selected_query_files) + len(selected_test_files)} 个文件")

    # 保存文件列表
    log_dir = dataset_root / "reorganize_logs"
    log_dir.mkdir(exist_ok=True)

    query_log = log_dir / "moved_from_query.txt"
    test_log = log_dir / "moved_from_test.txt"

    with open(query_log, 'w') as f:
        f.write('\n'.join(sorted(selected_query_files)))

    with open(test_log, 'w') as f:
        f.write('\n'.join(sorted(selected_test_files)))

    print(f"\n文件列表已保存:")
    print(f"  {query_log}")
    print(f"  {test_log}")

    if dry_run:
        print(f"\n{'='*60}")
        print("Dry-run模式，不执行实际移动")
        print(f"{'='*60}")
        print("\n预览前10个将要移动的文件:")
        print("\n从query:")
        for f in selected_query_files[:10]:
            print(f"  {f}")
        print("\n从test:")
        for f in selected_test_files[:10]:
            print(f"  {f}")
        return

    # 执行移动操作
    print(f"\n{'='*60}")
    print("开始移动文件")
    print(f"{'='*60}")

    stats = {
        'query_main': {'moved': 0, 'missing': 0},
        'test_main': {'moved': 0, 'missing': 0},
        'query_pifpaf': {'moved': 0, 'missing': 0},
        'test_pifpaf': {'moved': 0, 'missing': 0},
        'query_maskrcnn': {'moved': 0, 'missing': 0},
        'test_maskrcnn': {'moved': 0, 'missing': 0}
    }

    # 1. 移动主目录的query文件
    print("\n1. 移动主目录 query -> train")
    moved, missing = move_files(query_dir, train_dir, selected_query_files, "Query主目录")
    stats['query_main']['moved'] = moved
    stats['query_main']['missing'] = missing

    # 2. 移动主目录的test文件
    print("\n2. 移动主目录 test -> train")
    moved, missing = move_files(test_dir, train_dir, selected_test_files, "Test主目录")
    stats['test_main']['moved'] = moved
    stats['test_main']['missing'] = missing

    # 3. 移动pifpaf的query文件（后缀: .jpg.confidence_fields.npy）
    print("\n3. 移动pifpaf query -> train")
    moved, missing = move_files(
        pifpaf_query_dir, pifpaf_train_dir, selected_query_files,
        "Query pifpaf",
        suffix_transform=lambda x: x + '.confidence_fields.npy'
    )
    stats['query_pifpaf']['moved'] = moved
    stats['query_pifpaf']['missing'] = missing

    # 4. 移动pifpaf的test文件（后缀: .jpg.confidence_fields.npy）
    print("\n4. 移动pifpaf test -> train")
    moved, missing = move_files(
        pifpaf_test_dir, pifpaf_train_dir, selected_test_files,
        "Test pifpaf",
        suffix_transform=lambda x: x + '.confidence_fields.npy'
    )
    stats['test_pifpaf']['moved'] = moved
    stats['test_pifpaf']['missing'] = missing

    # 5. 移动maskrcnn的query文件（后缀: .npy，去掉.jpg）
    print("\n5. 移动maskrcnn query -> train")
    moved, missing = move_files(
        maskrcnn_query_dir, maskrcnn_train_dir, selected_query_files,
        "Query maskrcnn",
        suffix_transform=lambda x: x.replace('.jpg', '.npy')
    )
    stats['query_maskrcnn']['moved'] = moved
    stats['query_maskrcnn']['missing'] = missing

    # 6. 移动maskrcnn的test文件（后缀: .npy，去掉.jpg）
    print("\n6. 移动maskrcnn test -> train")
    moved, missing = move_files(
        maskrcnn_test_dir, maskrcnn_train_dir, selected_test_files,
        "Test maskrcnn",
        suffix_transform=lambda x: x.replace('.jpg', '.npy')
    )
    stats['test_maskrcnn']['moved'] = moved
    stats['test_maskrcnn']['missing'] = missing

    # 打印统计信息
    print(f"\n{'='*60}")
    print("移动完成 - 统计信息")
    print(f"{'='*60}")

    total_moved = 0
    total_missing = 0

    for key, value in stats.items():
        print(f"{key:20s}: 移动 {value['moved']:5d}, 缺失 {value['missing']:5d}")
        total_moved += value['moved']
        total_missing += value['missing']

    print(f"{'='*60}")
    print(f"{'总计':20s}: 移动 {total_moved:5d}, 缺失 {total_missing:5d}")
    print(f"{'='*60}")

    # 验证结果
    print(f"\n{'='*60}")
    print("验证结果")
    print(f"{'='*60}")

    train_files = get_all_files(train_dir)
    pifpaf_train_files = get_all_files(pifpaf_train_dir)
    maskrcnn_train_files = get_all_files(maskrcnn_train_dir)

    print(f"主目录train文件数: {len(train_files)}")
    print(f"pifpaf train文件数: {len(pifpaf_train_files)}")
    print(f"maskrcnn train文件数: {len(maskrcnn_train_files)}")

    expected_count = query_count + test_count
    print(f"\n期望文件数: {expected_count}")

    if len(train_files) == expected_count:
        print("✓ 主目录train文件数正确")
    else:
        print(f"✗ 主目录train文件数不正确 (期望{expected_count}, 实际{len(train_files)})")

    print("\n完成！")


def main():
    parser = argparse.ArgumentParser(
        description='重组DaJixiang数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset-root', type=str,
                       default='/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/datasets/DaJixiang',
                       help='数据集根目录')
    parser.add_argument('--query-count', type=int, default=3000,
                       help='从query移动的图片数量')
    parser.add_argument('--test-count', type=int, default=6000,
                       help='从test移动的图片数量')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅预览不实际移动')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("DaJixiang数据集重组")
    print(f"{'='*60}")
    print(f"数据集根目录: {args.dataset_root}")
    print(f"从query移动: {args.query_count} 张")
    print(f"从test移动: {args.test_count} 张")
    print(f"Dry-run模式: {args.dry_run}")
    print(f"{'='*60}")

    reorganize_dataset(
        dataset_root=args.dataset_root,
        query_count=args.query_count,
        test_count=args.test_count,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
