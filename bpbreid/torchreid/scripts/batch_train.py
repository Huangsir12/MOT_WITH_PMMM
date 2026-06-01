"""
批量训练ReID模型脚本
轮流训练market1501和dajixiang数据集，使用不同的backbone和loss组合
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM/bpbreid")

from torchreid.scripts.reID_app import build_config, build_torchreid_model_engine
from torchreid.scripts.default_config import engine_run_kwargs


class TrainingConfig:
    """训练配置类"""
    def __init__(self, dataset, backbone, loss, config_file):
        self.dataset = dataset
        self.backbone = backbone
        self.loss = loss
        self.config_file = config_file

        # 根据loss类型确定model name
        if loss == 'triplet':
            self.model_name = backbone
        else:  # part_based
            self.model_name = 'bpbreid'

    def __repr__(self):
        return f"TrainingConfig(dataset={self.dataset}, backbone={self.backbone}, loss={self.loss})"


class TrainingResult:
    """训练结果类"""
    def __init__(self, config, log_dir, best_weight_path, metrics):
        self.config = config
        self.log_dir = log_dir
        self.best_weight_path = best_weight_path
        self.metrics = metrics
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'dataset': self.config.dataset,
            'backbone': self.config.backbone,
            'loss': self.config.loss,
            'model_name': self.config.model_name,
            'log_dir': self.log_dir,
            'best_weight_path': self.best_weight_path,
            'metrics': self.metrics
        }


def generate_training_configs():
    """生成所有训练配置组合"""
    datasets = [
        ('market1501', '/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/configs/bpbreid/bpbreid_market1501_train.yaml'),
        ('dajixiang', '/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/configs/bpbreid/bpbreid_dajixiang_train.yaml')
    ]

    backbones = ['hrnet32', 'osnet_x1_0', 'tosnet_x1_0']
    losses = ['part_based', 'triplet']

    configs = []
    for dataset_name, config_file in datasets:
        for backbone in backbones:
            for loss in losses:
                configs.append(TrainingConfig(dataset_name, backbone, loss, config_file))

    return configs


def train_single_config(config, args):
    """训练单个配置"""
    print(f"\n{'='*80}")
    print(f"开始训练: {config}")
    print(f"{'='*80}\n")

    # 根据backbone确定pretrained和max_epoch
    if 'tosnet' in config.backbone.lower():
        pretrained = False
        max_epoch = 100
    else:
        pretrained = True
        max_epoch = 60

    # 构建命令行参数对象
    class Args:
        def __init__(self):
            self.config_file = config.config_file
            self.sources = None
            self.targets = None
            self.transforms = None
            self.root = '/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/datasets'
            self.job_id = None
            self.save_dir = os.path.join(args.output_dir, f"{config.dataset}_{config.backbone}_{config.loss}")
            self.inference_enabled = False
            # 通过opts修改配置
            self.opts = [
                'model.bpbreid.backbone', config.backbone,
                'model.name', config.model_name,
                'loss.name', config.loss,
                'model.pretrained', str(pretrained),
                'train.max_epoch', str(max_epoch),
                'data.save_dir', os.path.join(args.output_dir, f"{config.dataset}_{config.model_name}_{config.loss}")
            ]

    train_args = Args()

    try:
        # 构建配置
        cfg = build_config(train_args, config.config_file)

        # 记录log目录和权重路径
        log_dir = cfg.data.save_dir
        best_weight_path = os.path.join(log_dir, 'model', 'model.pth.tar-best')

        # 构建模型和引擎
        engine, model = build_torchreid_model_engine(cfg)

        print(f'\n开始训练实验 {cfg.project.experiment_id}')
        print(f'Job ID: {cfg.project.job_id}')
        print(f'创建时间: {cfg.project.start_time}')
        print(f'Log目录: {log_dir}')
        print(f'Backbone: {config.backbone}, Loss: {config.loss}')
        print(f'Pretrained: {pretrained}, Max Epoch: {max_epoch}\n')

        # 运行训练并获取结果
        result_dict = engine.run(**engine_run_kwargs(cfg))

        print(f'\n训练完成: 实验 {cfg.project.experiment_id}')

        # 直接使用返回的结果
        metrics = {
            'mAP': result_dict.get('mAP'),
            'rank-1': result_dict.get('rank_1'),
            'rank-5': result_dict.get('rank_5'),
            'rank-10': result_dict.get('rank_10'),
            'rank-20': result_dict.get('rank_20'),
            'ssmd': result_dict.get('ssmd'),
            'cmc': {
                'rank-1': result_dict.get('rank_1'),
                'rank-5': result_dict.get('rank_5'),
                'rank-10': result_dict.get('rank_10'),
                'rank-20': result_dict.get('rank_20')
            }
        }

        # 打印提取的指标
        print(f'\n评估指标:')
        print(f'  mAP: {metrics.get("mAP", "N/A")}')
        print(f'  Rank-1: {metrics.get("rank-1", "N/A")}')
        print(f'  Rank-5: {metrics.get("rank-5", "N/A")}')
        print(f'  Rank-10: {metrics.get("rank-10", "N/A")}')
        print(f'  Rank-20: {metrics.get("rank-20", "N/A")}')
        print(f'  SSMD: {metrics.get("ssmd", "N/A")}')

        # 返回训练结果
        result = TrainingResult(config, log_dir, best_weight_path, metrics)
        return result

    except Exception as e:
        print(f"\n训练失败: {config}")
        print(f"错误信息: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return None


def save_results_to_markdown(results, output_file):
    """将训练结果保存为Markdown文档"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# ReID模型批量训练结果报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"总训练任务数: {len(results)}\n")
        f.write(f"成功任务数: {sum(1 for r in results if r is not None)}\n")
        f.write(f"失败任务数: {sum(1 for r in results if r is None)}\n\n")

        f.write("---\n\n")

        # 按数据集分组
        datasets = {}
        for result in results:
            if result is None:
                continue
            dataset = result.config.dataset
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(result)

        # 为每个数据集生成表格
        for dataset, dataset_results in datasets.items():
            f.write(f"## 数据集: {dataset.upper()}\n\n")

            # 表头
            f.write("| Backbone | Loss | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 | SSMD | Log目录 | 最佳权重 |\n")
            f.write("|----------|------|-----|--------|--------|---------|---------|------|---------|----------|\n")

            # 表格内容
            for result in dataset_results:
                metrics = result.metrics
                f.write(f"| {result.config.backbone} | {result.config.loss} | "
                       f"{metrics.get('mAP', 'N/A')} | "
                       f"{metrics.get('rank-1', 'N/A')} | "
                       f"{metrics.get('rank-5', 'N/A')} | "
                       f"{metrics.get('rank-10', 'N/A')} | "
                       f"{metrics.get('rank-20', 'N/A')} | "
                       f"{metrics.get('ssmd', 'N/A')} | "
                       f"`{result.log_dir}` | "
                       f"`{result.best_weight_path}` |\n")

            f.write("\n")

        # 详细信息
        f.write("---\n\n")
        f.write("## 详细训练信息\n\n")

        for i, result in enumerate(results, 1):
            if result is None:
                continue

            f.write(f"### 训练任务 {i}\n\n")
            f.write(f"- **训练时间**: {result.timestamp}\n")
            f.write(f"- **数据集**: {result.config.dataset}\n")
            f.write(f"- **Backbone**: {result.config.backbone}\n")
            f.write(f"- **Loss**: {result.config.loss}\n")
            f.write(f"- **Model Name**: {result.config.model_name}\n")
            f.write(f"- **Log目录**: `{result.log_dir}`\n")
            f.write(f"- **最佳权重**: `{result.best_weight_path}`\n")
            f.write(f"\n**评估指标**:\n")
            for metric_name, metric_value in result.metrics.items():
                if metric_name == 'cmc' and isinstance(metric_value, dict):
                    f.write(f"- CMC:\n")
                    for rank, value in metric_value.items():
                        f.write(f"  - {rank}: {value if value is not None else 'N/A'}\n")
                else:
                    f.write(f"- {metric_name}: {metric_value if metric_value is not None else 'N/A'}\n")
            f.write("\n")


def save_results_to_json(results, output_file):
    """将训练结果保存为JSON文件"""
    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_tasks': len(results),
        'successful_tasks': sum(1 for r in results if r is not None),
        'failed_tasks': sum(1 for r in results if r is None),
        'results': [r.to_dict() for r in results if r is not None]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='批量训练ReID模型')
    parser.add_argument('--output-dir', type=str, default='/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/torchreid/logs/20260326',
                       help='结果输出目录')
    parser.add_argument('--start-from', type=int, default=0,
                       help='从第几个配置开始训练（用于断点续训）')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅打印训练配置，不实际训练')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成所有训练配置
    configs = generate_training_configs()

    print(f"\n{'='*80}")
    print(f"批量训练ReID模型")
    print(f"{'='*80}")
    print(f"总共需要训练 {len(configs)} 个配置")
    print(f"从第 {args.start_from + 1} 个配置开始训练")
    print(f"结果输出目录: {output_dir}\n")

    # 打印所有配置
    print("训练配置列表:")
    for i, config in enumerate(configs, 1):
        pretrained = False if 'tosnet' in config.backbone.lower() else True
        max_epoch = 100 if 'tosnet' in config.backbone.lower() else 60
        print(f"  {i}. Dataset: {config.dataset}, Backbone: {config.backbone}, "
              f"Loss: {config.loss}, Model: {config.model_name}, "
              f"Pretrained: {pretrained}, Epochs: {max_epoch}")
    print()

    if args.dry_run:
        print("Dry run 模式，不执行实际训练")
        return

    # 训练所有配置
    results = []
    for i, config in enumerate(configs[args.start_from:], args.start_from):
        print(f"\n{'='*80}")
        print(f"进度: {i+1}/{len(configs)}")
        print(f"{'='*80}")

        result = train_single_config(config, args)
        results.append(result)

        # 每完成一个训练就保存一次结果（防止中途失败丢失数据）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_md_file = output_dir / f'training_results_progress.md'
        temp_json_file = output_dir / f'training_results_progress.json'
        save_results_to_markdown(results, temp_md_file)
        save_results_to_json(results, temp_json_file)

        print(f"\n进度已保存到: {temp_md_file}")

    # 保存最终结果
    final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_md_file = output_dir / f'training_results_final_{final_timestamp}.md'
    final_json_file = output_dir / f'training_results_final_{final_timestamp}.json'

    save_results_to_markdown(results, final_md_file)
    save_results_to_json(results, final_json_file)

    print(f"\n{'='*80}")
    print("所有训练任务完成！")
    print(f"{'='*80}")
    print(f"结果已保存到:")
    print(f"  - Markdown: {final_md_file}")
    print(f"  - JSON: {final_json_file}")

    # 打印统计信息
    successful = sum(1 for r in results if r is not None and r.metrics.get('mAP') is not None)
    failed = len(results) - successful
    print(f"\n训练统计:")
    print(f"  总任务数: {len(results)}")
    print(f"  成功: {successful}")
    print(f"  失败: {failed}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
