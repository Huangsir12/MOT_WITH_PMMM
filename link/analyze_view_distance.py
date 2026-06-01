"""
分析视角对特征距离分布的影响
比较同视角/不同视角、同ID/不同ID的特征距离分布
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from datetime import datetime

# 设置中文字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['WenQuanYi Bitmap Song']

sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from bpbreid.torchreid.scripts.reID_app import inference_reid_init
from classify.view.inference import ViewPointPredictor
from bpbreid.torchreid.metrics.distance import compute_distance_matrix_using_bp_features


class ViewDistanceAnalyzer:
    """视角-距离分布分析器"""

    def __init__(self, crops_dir, reid_config_file, view_model_path, label_dir):
        """
        Args:
            crops_dir: crops数据目录
            reid_config_file: ReID配置文件路径
            view_model_path: 视角分类模型路径
            label_dir: 标注文件目录
        """
        self.crops_dir = crops_dir
        self.label_dir = label_dir

        # 初始化ReID模型
        print("初始化ReID模型...")
        with open(reid_config_file, 'r') as file:
            doc = yaml.safe_load(file)
        doc['inference']['dataset_folder'] = self.crops_dir
        with open(reid_config_file, 'w') as file:
            yaml.safe_dump(doc, file, default_flow_style=None)
        self.reid_inference = inference_reid_init(reid_config_file)

        # 初始化视角分类模型
        print(f"初始化视角分类模型: {view_model_path}")
        self.view_predictor = ViewPointPredictor(view_model_path, device="cuda")

        # 视角类别定义
        self.view_classes = self._generate_view_classes()
        print(f"视角类别数: {len(self.view_classes)}")

    def _generate_view_classes(self):
        """生成视角类别映射"""
        orientations = ["face", "back"]
        angles = ["0°", "45°", "90°", "135°", "180°"]
        view_classes = []
        for ori in orientations:
            for angle in angles:
                view_classes.append(f"{ori}_{angle}")
        return view_classes

    def _get_view_class_id(self, orientation, angle):
        """根据朝向和角度获取视角类别ID"""
        view_key = f"{orientation}_{angle}"
        if view_key in self.view_classes:
            return self.view_classes.index(view_key)
        return -1

    def classify_and_get_info(self, image_folder):
        """
        对图片进行视角分类并获取track_id和view_id

        Returns:
            dict: {image_path: {'track_id': int, 'view_id': int}}
        """
        print(f"\n对 {image_folder} 进行视角分类...")
        image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))

        if len(image_paths) == 0:
            print(f"警告: {image_folder} 中没有找到图片")
            return {}

        # 批量预测视角
        results = self.view_predictor.predict_batch(image_paths, batch_size=32)

        # 构建信息字典
        info_dict = {}
        for result in results:
            path = str(result['path'])
            filename = os.path.basename(path)

            # 从文件名提取track_id
            try:
                track_id = int(filename.split('_')[0])
            except:
                print(f"警告: 无法从文件名提取track_id: {filename}")
                continue

            orientation = result['orientation']
            angle = result['angle']
            view_id = self._get_view_class_id(orientation, angle)

            info_dict[path] = {
                'track_id': track_id,
                'view_id': view_id
            }

        print(f"成功分类 {len(info_dict)} 张图片")
        return info_dict

    def extract_features(self, image_folder):
        """
        提取图片的ReID特征

        Returns:
            embeddings: [N, P, D] 特征张量
            visibility_scores: [N, P] 可见性分数
            image_paths: 图片路径列表
        """
        print(f"\n提取 {image_folder} 的ReID特征...")
        image_paths = sorted(list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png")))

        if len(image_paths) == 0:
            return None, None, []

        # 转换为字符串路径
        image_paths = [str(p) for p in image_paths]

        # 提取特征
        embeddings, visibility_scores, parts_masks = self.reid_inference.extract_part_based_features(
            self.reid_inference.extractor,
            image_paths,
            batch_size=16
        )

        print(f"特征形状: {embeddings.shape}")
        return embeddings, visibility_scores, image_paths

    def compute_distance_matrix(self, qf, gf, qf_vis, gf_vis):
        """
        计算距离矩阵

        Args:
            qf: query特征 [Nq, P, D]
            gf: gallery特征 [Ng, P, D]
            qf_vis: query可见性 [Nq, P]
            gf_vis: gallery可见性 [Ng, P]

        Returns:
            distmat: [Nq, Ng] 距离矩阵
        """
        print("\n计算距离矩阵...")

        # L2归一化
        qf = F.normalize(qf, p=2, dim=-1)
        gf = F.normalize(gf, p=2, dim=-1)

        # 计算距离矩阵
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(
            qf, gf, qf_vis, gf_vis,
            dist_combine_strat='mean',
            batch_size_pairwise_dist_matrix=5000,
            use_gpu=True,
            metric='euclidean'
        )

        distmat = distmat.cpu().numpy()
        print(f"距离矩阵形状: {distmat.shape}")
        return distmat

    def load_ground_truth(self, label_file):
        """加载标注文件"""
        with open(label_file, 'r') as f:
            gt = json.load(f)
        # 转换为整数键值对
        gt_int = {int(k): int(v) for k, v in gt.items()}
        return gt_int

    def group_pairs_by_view_and_id(self, query_info, gallery_info, ground_truth, distmat, query_paths, gallery_paths):
        """
        根据视角和ID将配对分组

        Groups:
            0: 同视角，同跟踪ID
            1: 同视角，不同跟踪ID
            2: 不同视角，同跟踪ID
            3: 不同视角，不同跟踪ID
            4: 同跟踪ID（不考虑视角）
            5: 不同跟踪ID（不考虑视角）

        Returns:
            dict: {group_id: [(query_idx, gallery_idx, distance)]}
        """
        print("\n根据视角和ID分组配对...")

        groups = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        for q_idx, q_path in enumerate(query_paths):
            q_info = query_info.get(q_path)
            if q_info is None:
                continue

            q_track_id = q_info['track_id']
            q_view_id = q_info['view_id']

            for g_idx, g_path in enumerate(gallery_paths):
                g_info = gallery_info.get(g_path)
                if g_info is None:
                    continue

                g_track_id = g_info['track_id']
                g_view_id = g_info['view_id']

                distance = distmat[q_idx, g_idx]

                # 判断是否同ID（根据ground truth）
                same_id = (q_track_id in ground_truth and ground_truth[q_track_id] == g_track_id)
                same_view = (q_view_id == g_view_id)

                # 分组（前4组：考虑视角）
                if same_view and same_id:
                    group_id = 0
                elif same_view and not same_id:
                    group_id = 1
                elif not same_view and same_id:
                    group_id = 2
                else:  # not same_view and not same_id
                    group_id = 3

                groups[group_id].append((q_idx, g_idx, distance))

                # 分组（后2组：不考虑视角）
                if same_id:
                    groups[4].append((q_idx, g_idx, distance))
                else:
                    groups[5].append((q_idx, g_idx, distance))

        # 打印每组数量
        print("\n各组配对数量:")
        group_names = {
            0: "同视角，同身份ID",
            1: "同视角，不同身份ID",
            2: "不同视角，同身份ID",
            3: "不同视角，不同身份ID",
            4: "同身份ID（不考虑视角）",
            5: "不同身份ID（不考虑视角）"
        }
        for group_id in range(6):
            print(f"  组{group_id} ({group_names[group_id]}): {len(groups[group_id])} 对")

        return groups

    def sample_and_analyze_distances(self, groups, sample_size=1000, bin_width=0.05):
        """
        从每组随机采样并分析距离分布

        Args:
            groups: 分组数据
            sample_size: 每组采样数量
            bin_width: 距离区间宽度

        Returns:
            dict: {group_id: {'distances': [], 'hist': [], 'bins': []}}
        """
        print(f"\n从每组采样 {sample_size} 对并分析距离分布...")

        results = {}

        for group_id in range(6):
            pairs = groups[group_id]

            # 提取距离
            distances = [pair[2] for pair in pairs]

            # 随机采样
            if len(distances) > sample_size:
                sampled_indices = np.random.choice(len(distances), sample_size, replace=False)
                sampled_distances = [distances[i] for i in sampled_indices]
            else:
                sampled_distances = distances
                print(f"  组{group_id}: 样本数不足{sample_size}，使用全部{len(distances)}个样本")

            # 计算直方图
            bins = np.arange(0, 1.6, bin_width)
            hist, bin_edges = np.histogram(sampled_distances, bins=bins)

            results[group_id] = {
                'distances': sampled_distances,
                'hist': hist,
                'bins': bin_edges
            }

            print(f"  组{group_id}: 采样{len(sampled_distances)}对, 距离范围[{min(sampled_distances):.3f}, {max(sampled_distances):.3f}]")

        return results

    def plot_distance_distributions(self, results, output_dir):
        """
        绘制距离分布柱状图

        Args:
            results: 分析结果
            output_dir: 输出目录（已包含时间戳子文件夹）
        """
        print("\n绘制距离分布图...")

        group_names = {
            0: "同身份ID，同视角",
            1: "不同身份ID，同视角",
            2: "同身份ID，不同视角",
            3: "不同身份ID，不同视角",
            4: "同身份ID（不考虑视角）",
            5: "不同身份ID（不考虑视角）"
        }

        colors = ['green', 'orange', 'blue', 'red', 'purple', 'brown']

        # 为每组绘制单独的图
        for group_id in range(6):
            fig, ax = plt.subplots(figsize=(10, 6))

            hist = results[group_id]['hist']
            bins = results[group_id]['bins']

            # 绘制柱状图
            ax.bar(bins[:-1], hist, width=bins[1]-bins[0],
                   color=colors[group_id], alpha=0.7, edgecolor='black')

            ax.set_xlabel('特征距离', fontsize=16)
            ax.set_ylabel('频数', fontsize=16)
            # ax.set_title(f'Group {group_id}: {group_names[group_id]}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 设置固定的坐标范围
            ax.set_xlim(0.2, 1.6)
            ax.set_ylim(0, 2000)

            # 添加统计信息
            distances = results[group_id]['distances']
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            median_dist = np.median(distances)

            stats_text = f'Mean: {mean_dist:.3f}\nStd: {std_dist:.3f}\nMedian: {median_dist:.3f}\nN: {len(distances)}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=14)

            plt.tight_layout()

            # 保存图片
            output_file = os.path.join(output_dir, f'distance_distribution_group{group_id}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  保存图片: {output_file}")

        # 绘制综合对比图1：前4组（考虑视角）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for group_id in range(4):
            ax = axes[group_id]

            hist = results[group_id]['hist']
            bins = results[group_id]['bins']

            ax.bar(bins[:-1], hist, width=bins[1]-bins[0],
                   color=colors[group_id], alpha=0.7, edgecolor='black')

            ax.set_xlabel('特征距离', fontsize=11)
            ax.set_ylabel('频数', fontsize=11)
            ax.set_title(f'Group {group_id}: {group_names[group_id]}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 设置固定的坐标范围
            ax.set_xlim(0.2, 1.6)
            ax.set_ylim(0, 2000)

            # 添加统计信息
            distances = results[group_id]['distances']
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            stats_text = f'Mean: {mean_dist:.3f}\nStd: {std_dist:.3f}\nN: {len(distances)}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

        plt.suptitle('Distance Distribution Comparison by View and ID', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存综合图1
        output_file = os.path.join(output_dir, 'distance_distribution_comparison_with_view.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  保存综合图1: {output_file}")

        # 绘制综合对比图2：后2组（不考虑视角）
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, group_id in enumerate([4, 5]):
            ax = axes[idx]

            hist = results[group_id]['hist']
            bins = results[group_id]['bins']

            ax.bar(bins[:-1], hist, width=bins[1]-bins[0],
                   color=colors[group_id], alpha=0.7, edgecolor='black')

            ax.set_xlabel('特征距离', fontsize=11)
            ax.set_ylabel('频数', fontsize=11)
            ax.set_title(f'Group {group_id}: {group_names[group_id]}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 设置固定的坐标范围
            ax.set_xlim(0.2, 1.6)
            ax.set_ylim(0, 2000)

            # 添加统计信息
            distances = results[group_id]['distances']
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            stats_text = f'Mean: {mean_dist:.3f}\nStd: {std_dist:.3f}\nN: {len(distances)}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

        plt.suptitle('Distance Distribution Comparison by ID Only', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存综合图2
        output_file = os.path.join(output_dir, 'distance_distribution_comparison_id_only.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  保存综合图2: {output_file}")

    def save_results(self, results, groups, output_dir):
        """
        保存分析结果到JSON文件

        Args:
            results: 分析结果
            groups: 分组数据
            output_dir: 输出目录（已包含时间戳子文件夹）
        """
        print("\n保存分析结果...")

        # 准备保存的数据
        save_data = {
            'groups': {}
        }

        for group_id in range(6):
            distances = results[group_id]['distances']

            # 转换numpy类型为Python原生类型
            distances_list = [float(d) for d in distances]

            save_data['groups'][group_id] = {
                'total_pairs': int(len(groups[group_id])),
                'sampled_pairs': int(len(distances)),
                'distances': distances_list,
                'statistics': {
                    'mean': float(np.mean(distances)),
                    'std': float(np.std(distances)),
                    'median': float(np.median(distances)),
                    'min': float(np.min(distances)),
                    'max': float(np.max(distances)),
                    'q25': float(np.percentile(distances, 25)),
                    'q75': float(np.percentile(distances, 75))
                }
            }

        # 保存到JSON
        output_file = os.path.join(output_dir, 'distance_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"  保存结果: {output_file}")

    def run_analysis(self, query_cam="camera_001", gallery_cam="camera_002",
                     label_file="camera001_002.json", output_dir="link/view_distance_analysis"):
        """
        运行完整分析流程

        Args:
            query_cam: query相机名称
            gallery_cam: gallery相机名称
            label_file: 标注文件名
            output_dir: 输出目录
        """
        print("="*80)
        print(f"开始分析: {query_cam} -> {gallery_cam}")
        print("="*80)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建带时间戳的子文件夹
        timestamped_output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(timestamped_output_dir, exist_ok=True)
        print(f"\n结果将保存到: {timestamped_output_dir}")

        # 构建路径
        query_folder = os.path.join(self.crops_dir, query_cam)
        gallery_folder = os.path.join(self.crops_dir, gallery_cam)
        label_path = os.path.join(self.label_dir, label_file)

        # 1. 视角分类
        query_info = self.classify_and_get_info(query_folder)
        gallery_info = self.classify_and_get_info(gallery_folder)

        # 2. 提取特征
        qf, qf_vis, query_paths = self.extract_features(query_folder)
        gf, gf_vis, gallery_paths = self.extract_features(gallery_folder)

        # 3. 计算距离矩阵
        distmat = self.compute_distance_matrix(qf, gf, qf_vis, gf_vis)

        # 4. 加载ground truth
        ground_truth = self.load_ground_truth(label_path)
        print(f"\n加载标注文件: {label_file}")
        print(f"标注匹配对数: {len(ground_truth)}")

        # 5. 分组
        groups = self.group_pairs_by_view_and_id(
            query_info, gallery_info, ground_truth,
            distmat, query_paths, gallery_paths
        )

        # 6. 采样并分析距离分布
        results = self.sample_and_analyze_distances(groups, sample_size=10000, bin_width=0.05)

        # 7. 绘制分布图
        self.plot_distance_distributions(results, timestamped_output_dir)

        # 8. 保存结果
        self.save_results(results, groups, timestamped_output_dir)

        print("\n" + "="*80)
        print("分析完成！")
        print(f"结果保存在: {timestamped_output_dir}")
        print("="*80)

        return results


if __name__ == "__main__":
    # 配置参数
    crops_dir = "data/datasets/bpbreid_dajixiang/crops_202603"
    label_dir = "data/datasets/link_dajixiang/link_label"
    reid_config_file = "bpbreid/configs/bpbreid/bpbreid_inference.yaml"
    view_model_path = "classify/view/checkpoints/best_model.pth"
    output_dir = "link/view_distance_analysis"

    # 创建分析器
    analyzer = ViewDistanceAnalyzer(
        crops_dir=crops_dir,
        label_dir=label_dir,
        reid_config_file=reid_config_file,
        view_model_path=view_model_path
    )

    # 运行分析
    results = analyzer.run_analysis(
        query_cam="camera_001",
        gallery_cam="camera_002",
        label_file="camera001_002.json",
        output_dir=output_dir
    )

    print("\n分析结果统计:")
    for group_id in range(6):
        distances = results[group_id]['distances']
        print(f"\n组{group_id}:")
        print(f"  样本数: {len(distances)}")
        print(f"  平均距离: {np.mean(distances):.4f}")
        print(f"  标准差: {np.std(distances):.4f}")
        print(f"  中位数: {np.median(distances):.4f}")
