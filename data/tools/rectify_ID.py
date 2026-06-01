"""
ID修正脚本 - 多阶段ID映射和分配

处理流程:
1. 从txt1加载初始已存在的ID集合
2. 处理txt2:
   - 使用json1映射（值->键）
   - 不在json1中的ID分配最小未使用ID
   - 记录所有映射关系，与json1合并
   - 更新已存在ID集合
3. 处理txt3:
   - 先用json2映射，成功的记为S1，失败的记为S2
   - S1继续用json1映射（使用合并后的映射）
   - S2分配最小未使用ID（保持相同ID映射到相同新ID）

用法:
python rectify_ID.py --txt1 results1.txt --txt2 results2.txt --txt3 results3.txt \
                     --json1 mapping1.json --json2 mapping2.json \
                     --output2 results2_rectified.txt --output3 results3_rectified.txt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_id_mapping(json_path):
    """加载ID映射字典 {key: value}"""
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return {int(k): int(v) for k, v in mapping.items()}


def create_reverse_mapping(id_mapping):
    """创建反向映射: value -> key"""
    return {v: k for k, v in id_mapping.items()}


def load_mot_results(txt_path):
    """加载MOT格式跟踪结果"""
    if not Path(txt_path).exists():
        raise FileNotFoundError(f"跟踪结果文件不存在: {txt_path}")

    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 9:
                data.append([float(x) for x in parts[:9]])
            elif len(parts) >= 7:
                row = [float(x) for x in parts[:7]]
                row.extend([0, 1])
                data.append(row)

    return np.array(data) if data else np.array([]).reshape(0, 9)


def get_unique_ids(mot_results):
    """获取MOT结果中的所有唯一ID"""
    if len(mot_results) == 0:
        return set()
    return set(int(x) for x in mot_results[:, 1])


def find_min_unused_id(existing_ids):
    """找到最小的未使用ID"""
    if not existing_ids:
        return 1
    candidate = 1
    while candidate in existing_ids:
        candidate += 1
    return candidate


def save_mot_results(mot_results, output_path):
    """保存MOT格式跟踪结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for row in mot_results:
            line = f"{int(row[0])},{int(row[1])},{int(row[2])},{int(row[3])},{int(row[4])},{int(row[5])},{row[6]:.6f},{int(row[7])},{int(row[8])}\n"
            f.write(line)


def process_txt2(txt2_results, json1_reverse, existing_ids):
    """
    处理txt2: 使用json1映射，不在映射中的分配新ID

    Args:
        txt2_results: txt2的MOT结果
        json1_reverse: json1的反向映射 {value: key}
        existing_ids: 已存在的ID集合（会被修改）

    Returns:
        numpy.ndarray: 修正后的结果
        dict: 新的映射关系 {old_id: new_id}
    """
    txt2_modified = txt2_results.copy()
    new_mappings = {}  # 记录所有映射关系

    # 获取txt2中的所有唯一ID
    txt2_unique_ids = get_unique_ids(txt2_results)

    print(f"\nTXT2原始唯一ID: {sorted(txt2_unique_ids)}")

    # 为每个唯一ID确定映射
    for old_id in txt2_unique_ids:
        if old_id in json1_reverse:
            # 在json1中找到映射
            new_id = json1_reverse[old_id]
            new_mappings[old_id] = new_id
            existing_ids.add(new_id)
        else:
            # 不在json1中，分配新ID
            new_id = find_min_unused_id(existing_ids)
            new_mappings[old_id] = new_id
            existing_ids.add(new_id)

    # 应用映射
    for i in range(len(txt2_modified)):
        old_id = int(txt2_modified[i, 1])
        txt2_modified[i, 1] = new_mappings[old_id]

    return txt2_modified, new_mappings


def process_txt3(txt3_results, json2_reverse, merged_json1_reverse, existing_ids):
    """
    处理txt3: 先用json2映射，再用合并后的json1映射

    Args:
        txt3_results: txt3的MOT结果
        json2_reverse: json2的反向映射 {value: key}
        merged_json1_reverse: 合并后的json1反向映射
        existing_ids: 已存在的ID集合（会被修改）

    Returns:
        numpy.ndarray: 修正后的结果
        dict: 统计信息
    """
    txt3_modified = txt3_results.copy()

    # 获取txt3中的所有唯一ID
    txt3_unique_ids = get_unique_ids(txt3_results)

    print(f"\nTXT3原始唯一ID: {sorted(txt3_unique_ids)}")

    # 第一步: 使用json2映射
    stage1_mappings = {}  # {old_id: intermediate_id}
    S1 = set()  # json2映射成功的ID
    S2 = set()  # json2映射失败的ID

    for old_id in txt3_unique_ids:
        if old_id in json2_reverse:
            intermediate_id = json2_reverse[old_id]
            stage1_mappings[old_id] = intermediate_id
            S1.add(old_id)
        else:
            S2.add(old_id)

    print(f"JSON2映射成功 (S1): {sorted(S1)}")
    print(f"JSON2映射失败 (S2): {sorted(S2)}")

    # 第二步: S1继续使用json1映射，S2分配新ID
    final_mappings = {}  # {old_id: final_id}

    # 处理S1
    for old_id in S1:
        intermediate_id = stage1_mappings[old_id]
        if intermediate_id in merged_json1_reverse:
            final_id = merged_json1_reverse[intermediate_id]
            final_mappings[old_id] = final_id
            existing_ids.add(final_id)
        else:
            # intermediate_id不在合并后的json1中，分配新ID
            final_id = find_min_unused_id(existing_ids)
            final_mappings[old_id] = final_id
            existing_ids.add(final_id)

    # 处理S2 - 相同的原始ID映射到相同的新ID
    for old_id in S2:
        if old_id not in final_mappings:
            new_id = find_min_unused_id(existing_ids)
            final_mappings[old_id] = new_id
            existing_ids.add(new_id)

    # 应用映射
    for i in range(len(txt3_modified)):
        old_id = int(txt3_modified[i, 1])
        txt3_modified[i, 1] = final_mappings[old_id]

    stats = {
        'S1': S1,
        'S2': S2,
        'stage1_mappings': stage1_mappings,
        'final_mappings': final_mappings
    }

    return txt3_modified, stats


def main():
    parser = argparse.ArgumentParser(
        description='多阶段ID修正脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--txt1', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1750/camera_001_0.txt",
                       help='第一个txt文件（提供初始已存在ID）')
    parser.add_argument('--txt2', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1759/camera_002_0.txt",
                       help='第二个txt文件（需要修正）')
    parser.add_argument('--txt3', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1843/camera_003_0.txt",
                       help='第三个txt文件（需要修正）')
    parser.add_argument('--json1', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/data/datasets/bpbreid_dajixiang/crops/camera_001_002.json",
                       help='第一个JSON映射文件')
    parser.add_argument('--json2', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/data/datasets/bpbreid_dajixiang/crops/camera_002_003.json",
                       help='第二个JSON映射文件')
    parser.add_argument('--output2', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1759/camera_002_0_xiugai.txt",
                       help='txt2修正后的输出路径')
    parser.add_argument('--output3', type=str, default="/root/autodl-tmp/MOT_WITH_PMMM/runs/track_reid/0326-1843/camera_003_0_xiugai.txt",
                       help='txt3修正后的输出路径')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示统计信息，不保存文件')

    args = parser.parse_args()

    # ========== 加载映射文件 ==========
    print(f"\n{'='*60}")
    print("步骤1: 加载映射文件")
    print(f"{'='*60}")
    json1_mapping = load_id_mapping(args.json1)
    json2_mapping = load_id_mapping(args.json2)
    json1_reverse = create_reverse_mapping(json1_mapping)
    json2_reverse = create_reverse_mapping(json2_mapping)

    print(f"JSON1映射规则数: {len(json1_mapping)}")
    print(f"JSON1反向映射示例: {dict(list(json1_reverse.items())[:5])}")
    print(f"JSON2映射规则数: {len(json2_mapping)}")
    print(f"JSON2反向映射示例: {dict(list(json2_reverse.items())[:5])}")

    # ========== 加载txt1，初始化已存在ID ==========
    print(f"\n{'='*60}")
    print("步骤2: 初始化已存在ID")
    print(f"{'='*60}")
    txt1_results = load_mot_results(args.txt1)
    existing_ids = get_unique_ids(txt1_results)

    print(f"TXT1: {len(txt1_results)} 检测框")
    print(f"初始已存在ID数: {len(existing_ids)}")
    print(f"初始已存在ID: {sorted(existing_ids)}")

    # ========== 处理txt2 ==========
    print(f"\n{'='*60}")
    print("步骤3: 处理TXT2")
    print(f"{'='*60}")
    txt2_results = load_mot_results(args.txt2)
    print(f"TXT2: {len(txt2_results)} 检测框")

    txt2_modified, txt2_mappings = process_txt2(txt2_results, json1_reverse, existing_ids)

    print(f"\nTXT2映射详情:")
    for old_id, new_id in sorted(txt2_mappings.items()):
        source = "JSON1" if old_id in json1_reverse else "新分配"
        print(f"  {old_id} -> {new_id} ({source})")

    print(f"\n处理后已存在ID数: {len(existing_ids)}")

    # 合并json1映射和txt2新映射
    merged_json1_reverse = json1_reverse.copy()
    for old_id, new_id in txt2_mappings.items():
        if old_id not in json1_reverse:
            # 将txt2的新映射加入到合并映射中
            merged_json1_reverse[old_id] = new_id

    print(f"合并后的JSON1映射规则数: {len(merged_json1_reverse)}")

    # 保存txt2
    if not args.dry_run:
        save_mot_results(txt2_modified, args.output2)
        print(f"\nTXT2已保存: {args.output2}")

    # ========== 处理txt3 ==========
    print(f"\n{'='*60}")
    print("步骤4: 处理TXT3")
    print(f"{'='*60}")
    txt3_results = load_mot_results(args.txt3)
    print(f"TXT3: {len(txt3_results)} 检测框")

    txt3_modified, txt3_stats = process_txt3(txt3_results, json2_reverse, merged_json1_reverse, existing_ids)

    print(f"\nTXT3处理详情:")
    print(f"阶段1 (JSON2映射):")
    for old_id in sorted(txt3_stats['S1']):
        intermediate_id = txt3_stats['stage1_mappings'][old_id]
        print(f"  {old_id} -> {intermediate_id}")

    print(f"\n阶段2 (最终映射):")
    for old_id, final_id in sorted(txt3_stats['final_mappings'].items()):
        if old_id in txt3_stats['S1']:
            intermediate_id = txt3_stats['stage1_mappings'][old_id]
            if intermediate_id in merged_json1_reverse:
                print(f"  {old_id} -> {intermediate_id} -> {final_id} (JSON2+JSON1)")
            else:
                print(f"  {old_id} -> {intermediate_id} -> {final_id} (JSON2+新分配)")
        else:
            print(f"  {old_id} -> {final_id} (新分配)")

    print(f"\n最终已存在ID数: {len(existing_ids)}")

    # 保存txt3
    if not args.dry_run:
        save_mot_results(txt3_modified, args.output3)
        print(f"\nTXT3已保存: {args.output3}")

    # ========== 最终统计 ==========
    print(f"\n{'='*60}")
    print("最终统计")
    print(f"{'='*60}")
    print(f"TXT2最终唯一ID数: {len(get_unique_ids(txt2_modified))}")
    print(f"TXT3最终唯一ID数: {len(get_unique_ids(txt3_modified))}")
    print(f"最终已存在ID数: {len(existing_ids)}")

    if args.dry_run:
        print("\nDry-run模式，未保存文件")
    else:
        print("\n完成！")


if __name__ == '__main__':
    main()
