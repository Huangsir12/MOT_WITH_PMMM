#!/usr/bin/env python3
"""
将gt.txt文件中每行的第一个数（帧号）都加1
"""

import os
import argparse
from pathlib import Path

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将gt.txt文件中每行的第一个数（帧号）都加1')
    parser.add_argument('--input', type=str, default='/root/autodl-tmp/MOT_WITH_PMMM/data/datasets/Emporium/train/door1/gt/gt.txt', help='输入gt.txt文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径，默认为覆盖原文件')
    args = parser.parse_args()
    
    # 处理输入输出路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入文件 {input_path} 不存在")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认为覆盖原文件，但先备份
        backup_path = input_path.with_suffix('.txt.bak')
        os.rename(input_path, backup_path)
        print(f"已备份原文件到 {backup_path}")
        output_path = input_path
    
    # 处理文件内容
    with open(backup_path if not args.output else input_path, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            modified_lines.append(line)
            continue
        
        # 分割每一行的数据
        parts = line.split(',')
        if not parts:
            modified_lines.append(line)
            continue
        
        # 将第一个数（帧号）加1
        try:
            frame_num = int(parts[0])
            parts[0] = str(frame_num + 1)
        except ValueError:
            print(f"警告：无法解析行 {line} 的帧号")
            modified_lines.append(line)
            continue
        
        # 重新组合行
        modified_line = ','.join(parts)
        modified_lines.append(modified_line)
    
    # 写入处理后的内容
    with open(output_path, 'w') as f:
        for line in modified_lines:
            f.write(line + '\n')
    
    print(f"处理完成！已将 {input_path} 中所有帧号加1")
    print(f"结果已保存到 {output_path}")
    return 0

if __name__ == '__main__':
    exit(main())
