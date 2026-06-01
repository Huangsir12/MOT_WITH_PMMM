"""
跨摄像头轨迹ID匹配评估脚本
比较origin方法和with_view两种方法的匹配准确性（精确率和召回率）
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from link_oringin import Link
from link_with_view import LinkWithView


class LinkEvaluator:
    """跨摄像头轨迹ID匹配评估器"""

    def __init__(self, crops_dir, label_dir, reid_config_file, view_model_path=None, only_run_metrics=False):
        """
        Args:
            crops_dir: crops数据目录
            label_dir: 标注文件目录
            reid_config_file: ReID配置文件路径
            view_model_path: 视角分类模型路径
        """
        self.crops_dir = crops_dir
        self.label_dir = label_dir
        self.reid_config_file = reid_config_file
        self.view_model_path = view_model_path

        # 定义相机对
        self.camera_pairs = [
            ("camera_001", "camera_002", "camera001_002.json"),
            ("camera_002", "camera_003", "camera002_003.json"),
            ("camera_003", "camera_004", "camera003_004.json"),
            ("camera_004", "camera_005", "camera004_005.json"),
            ("camera_005", "camera_006", "camera005_006.json"),
        ]
        
        if not only_run_metrics:
            # 初始化Link
            self.link = Link(
                input_dir=self.crops_dir,
                reid_config_file=self.reid_config_file
            )

            # 初始化LinkWithView
            self.viewlink = LinkWithView(
                input_dir=self.crops_dir,
                reid_config_file=self.reid_config_file,
                view_model_path=self.view_model_path,
            )

    def load_ground_truth(self, label_file):
        """
        加载标注文件

        Args:
            label_file: 标注文件路径

        Returns:
            dict: {query_id: gallery_id}
        """
        with open(label_file, 'r') as f:
            gt = json.load(f)

        # 转换为整数键值对
        gt_int = {int(k): int(v) for k, v in gt.items()}
        return gt_int

    def calculate_metrics(self, predictions, ground_truth):
        """
        计算精确率和召回率

        Args:
            predictions: 预测结果 {query_id: gallery_id}
            ground_truth: 标注结果 {query_id: gallery_id}

        Returns:
            dict: 包含precision, recall, f1, tp, fp, fn的字典
        """
        # 真阳性(TP): 预测正确的匹配对
        tp = 0
        # 假阳性(FP): 预测错误的匹配对
        fp = 0
        # 假阴性(FN): 应该匹配但未预测出的对
        fn = 0

        # 统计TP和FP
        for query_id, pred_gallery_id in predictions.items():
            if query_id in ground_truth:
                if ground_truth[query_id] == pred_gallery_id:
                    tp += 1
                else:
                    fp += 1
            else:
                # 预测了一个不在ground truth中的query_id
                fp += 1

        # 统计FN: ground truth中存在但未被预测的
        for query_id in ground_truth:
            if query_id not in predictions:
                fn += 1

        # 计算精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': len(ground_truth),
            'total_pred': len(predictions)
        }

    def evaluate_origin_method(self, query_folder, gallery_folder):
        """
        评估origin方法

        Args:
            query_folder: query集文件夹路径
            gallery_folder: gallery集文件夹路径

        Returns:
            dict: 匹配结果 {query_id: gallery_id}
        """
        print(f"\n{'='*60}")
        print(f"运行Origin方法")
        print(f"Query: {os.path.basename(query_folder)}")
        print(f"Gallery: {os.path.basename(gallery_folder)}")
        print(f"{'='*60}")

        # 获取ID统计
        query_pids_counts = self.link.get_counts(query_folder)
        print(f"{query_folder}中ID数量：{len(query_pids_counts)}")

        # 运行匹配
        matches = self.link.run_link(
            pids_counts=query_pids_counts,
            query_folder=query_folder,
            gallery_folder=gallery_folder
        )

        print(f"Origin方法匹配完成，匹配对数: {len(matches)}")
        return matches

    def evaluate_with_view_method1(self, query_folder, gallery_folder):
        """
        评估with_view方法1: 视角分类后分堆匹配

        Args:
            query_folder: query集文件夹路径
            gallery_folder: gallery集文件夹路径

        Returns:
            dict: 匹配结果 {query_id: gallery_id}
        """
        print(f"\n{'='*60}")
        print(f"运行With_View方法1: 视角分类后分堆匹配")
        print(f"Query: {os.path.basename(query_folder)}")
        print(f"Gallery: {os.path.basename(gallery_folder)}")
        print(f"{'='*60}")


        # 运行方法1
        matches = self.viewlink.run_link(method="method1", 
                                         query_dir=query_folder, 
                                         gallery_dir=gallery_folder, 
                                         distance_threshold=1.0)

        print(f"With_View方法1匹配完成，匹配对数: {len(matches)}")
        return matches

    def evaluate_with_view_method2(self, query_folder, gallery_folder):
        """
        评估with_view方法2: 视角特征串联匹配

        Args:
            query_folder: query集文件夹路径
            gallery_folder: gallery集文件夹路径

        Returns:
            dict: 匹配结果 {query_id: gallery_id}
        """
        print(f"\n{'='*60}")
        print(f"运行With_View方法2: 视角特征串联匹配")
        print(f"Query: {os.path.basename(query_folder)}")
        print(f"Gallery: {os.path.basename(gallery_folder)}")
        print(f"{'='*60}")

        # 运行方法2
        matches = self.viewlink.run_link(
            method="method2",
            query_dir=query_folder, 
            gallery_dir=gallery_folder, 
            distance_threshold=0.6,
        )

        print(f"With_View方法2匹配完成，匹配对数: {len(matches)}")
        return matches

    def save_results(self, results, output_file):
        """
        保存评估结果到JSON文件

        Args:
            results: 评估结果字典
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    def print_metrics(self, method_name, metrics):
        """打印评估指标"""
        print(f"\n{method_name} 评估结果:")
        print(f"  精确率 (Precision): {metrics['precision']:.4f} ({metrics['tp']}/{metrics['total_pred']})")
        print(f"  召回率 (Recall):    {metrics['recall']:.4f} ({metrics['tp']}/{metrics['total_gt']})")
        print(f"  F1分数:            {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

    def run_evaluation(self, output_dir="link/evaluation_results"):
        """
        运行完整评估流程

        Args:
            output_dir: 结果输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 存储所有结果
        all_results = {
            'origin': {'pairs': [], 'average': {}},
            # 'with_view_method1': {'pairs': [], 'average': {}},
            'with_view_method2': {'pairs': [], 'average': {}}
        }

        # 遍历所有相机对
        for query_cam, gallery_cam, label_file in self.camera_pairs:
            print(f"\n{'#'*80}")
            print(f"# 评估相机对: {query_cam} -> {gallery_cam}")
            print(f"{'#'*80}")

            # 构建路径
            query_folder = os.path.join(self.crops_dir, query_cam)
            gallery_folder = os.path.join(self.crops_dir, gallery_cam)
            label_path = os.path.join(self.label_dir, label_file)

            # 加载ground truth
            ground_truth = self.load_ground_truth(label_path)
            print(f"\n加载标注文件: {label_file}")
            print(f"标注匹配对数: {len(ground_truth)}")
            print(ground_truth)

            # 评估origin方法
            try:
                origin_matches = self.evaluate_origin_method(query_folder, gallery_folder)
                print(origin_matches)
                rectify_origin_matches = {query_id: gallery_info[0] for query_id, gallery_info in origin_matches.items()}

                origin_metrics = self.calculate_metrics(rectify_origin_matches, ground_truth)
                self.print_metrics("Origin方法", origin_metrics)

                all_results['origin']['pairs'].append({
                    'query': query_cam,
                    'gallery': gallery_cam,
                    'metrics': origin_metrics,
                    'matches': origin_matches
                })
            except Exception as e:
                print(f"Origin方法评估失败: {e}")
                all_results['origin']['pairs'].append({
                    'query': query_cam,
                    'gallery': gallery_cam,
                    'error': str(e)
                })

            # # 评估with_view方法1
            # try:
            #     method1_matches = self.evaluate_with_view_method1(query_folder, gallery_folder)
            #     print(method1_matches)
            #     method1_metrics = self.calculate_metrics(method1_matches, ground_truth)
            #     self.print_metrics("With_View方法1", method1_metrics)

            #     all_results['with_view_method1']['pairs'].append({
            #         'query': query_cam,
            #         'gallery': gallery_cam,
            #         'metrics': method1_metrics,
            #         'matches': method1_matches
            #     })
            # except Exception as e:
            #     print(f"With_View方法1评估失败: {e}")
            #     all_results['with_view_method1']['pairs'].append({
            #         'query': query_cam,
            #         'gallery': gallery_cam,
            #         'error': str(e)
            #     })

            # 评估with_view方法2
            try:
                method2_matches = self.evaluate_with_view_method2(query_folder, gallery_folder)
                print(method2_matches)
                method2_metrics = self.calculate_metrics(method2_matches, ground_truth)
                self.print_metrics("With_View方法2", method2_metrics)

                all_results['with_view_method2']['pairs'].append({
                    'query': query_cam,
                    'gallery': gallery_cam,
                    'metrics': method2_metrics,
                    'matches': method2_matches
                })
            except Exception as e:
                print(f"With_View方法2评估失败: {e}")
                all_results['with_view_method2']['pairs'].append({
                    'query': query_cam,
                    'gallery': gallery_cam,
                    'error': str(e)
                })

        # 计算平均性能
        print(f"\n{'#'*80}")
        print(f"# 计算平均性能")
        print(f"{'#'*80}")

        # for method_name in ['origin', 'with_view_method1', 'with_view_method2']:
        for method_name in ['origin', 'with_view_method2']:
            pairs = all_results[method_name]['pairs']

            # 过滤掉失败的评估
            valid_pairs = [p for p in pairs if 'metrics' in p]

            if len(valid_pairs) == 0:
                print(f"\n{method_name}: 没有有效的评估结果")
                continue

            # 计算平均值
            avg_precision = sum(p['metrics']['precision'] for p in valid_pairs) / len(valid_pairs)
            avg_recall = sum(p['metrics']['recall'] for p in valid_pairs) / len(valid_pairs)
            avg_f1 = sum(p['metrics']['f1'] for p in valid_pairs) / len(valid_pairs)
            total_tp = sum(p['metrics']['tp'] for p in valid_pairs)
            total_fp = sum(p['metrics']['fp'] for p in valid_pairs)
            total_fn = sum(p['metrics']['fn'] for p in valid_pairs)

            all_results[method_name]['average'] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'num_pairs': len(valid_pairs)
            }

            print(f"\n{method_name} 平均性能 (基于{len(valid_pairs)}对相机):")
            print(f"  平均精确率: {avg_precision:.4f}")
            print(f"  平均召回率: {avg_recall:.4f}")
            print(f"  平均F1分数: {avg_f1:.4f}")
            print(f"  总计 TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存结果
        output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        self.save_results(all_results, output_file)

        # 生成对比报告
        self.generate_comparison_report(all_results, output_dir, timestamp)

        return all_results

    def run_metrics(self, matches_results_file="link/evaluation_results/evaluation_results.json", output_dir="link/evaluation_results"):
        """
        加载现有的匹配结果，重新进行评估（不重新跑matches）

        Args:
            matches_results_file: 已保存的匹配结果JSON文件路径
            output_dir: 结果输出目录
        """
        print(f"\n{'='*80}")
        print(f"从已有结果文件加载匹配数据: {matches_results_file}")
        print(f"{'='*80}")

        # 加载已有的匹配结果
        if not os.path.exists(matches_results_file):
            raise FileNotFoundError(f"匹配结果文件不存在: {matches_results_file}")

        with open(matches_results_file, 'r', encoding='utf-8') as f:
            saved_results = json.load(f)

        # 存储重新评估的结果
        all_results = {
            'origin': {'pairs': [], 'average': {}},
            # 'with_view_method1': {'pairs': [], 'average': {}},
            'with_view_method2': {'pairs': [], 'average': {}}
        }

        # 遍历所有相机对
        for idx, (query_cam, gallery_cam, label_file) in enumerate(self.camera_pairs):
            print(f"\n{'#'*80}")
            print(f"# 重新评估相机对: {query_cam} -> {gallery_cam}")
            print(f"{'#'*80}")

            # 构建标注文件路径
            label_path = os.path.join(self.label_dir, label_file)

            # 加载ground truth
            ground_truth = self.load_ground_truth(label_path)
            print(f"\n加载标注文件: {label_file}")
            print(f"标注匹配对数: {len(ground_truth)}")

            # 对每种方法重新评估
            for method_name in ['origin', 'with_view_method2']:
                try:
                    # 从保存的结果中提取matches
                    saved_pair = saved_results[method_name]['pairs'][idx]

                    if 'error' in saved_pair:
                        print(f"{method_name} 原评估失败: {saved_pair['error']}")
                        all_results[method_name]['pairs'].append({
                            'query': query_cam,
                            'gallery': gallery_cam,
                            'error': saved_pair['error']
                        })
                        continue

                    # # 提取matches（字符串键转为整数）
                    # matches = {int(k): int(v) for k, v in saved_pair['matches'].items()}

                    # matches = saved_pair.get("matches", {})
                    if method_name == "origin":
                        matches = {query_id: gallery_info[0] for query_id, gallery_info in saved_pair['matches'].items()}
                    else:
                        matches = saved_pair['matches']

                    # print(f"matches:\n{matches}")
                    # print(f"ground_truth:\n{ground_truth}")
                    # 重新计算metrics
                    metrics = self.calculate_metrics(matches, ground_truth)
                    self.print_metrics(method_name, metrics)

                    all_results[method_name]['pairs'].append({
                        'query': query_cam,
                        'gallery': gallery_cam,
                        'metrics': metrics,
                        'matches': matches
                    })

                except Exception as e:
                    print(f"{method_name} 重新评估失败: {e}")
                    all_results[method_name]['pairs'].append({
                        'query': query_cam,
                        'gallery': gallery_cam,
                        'error': str(e)
                    })

        # 计算平均性能
        print(f"\n{'#'*80}")
        print(f"# 计算平均性能")
        print(f"{'#'*80}")

        for method_name in ['origin', 'with_view_method2']:
            pairs = all_results[method_name]['pairs']

            # 过滤掉失败的评估
            valid_pairs = [p for p in pairs if 'metrics' in p]

            if len(valid_pairs) == 0:
                print(f"\n{method_name}: 没有有效的评估结果")
                continue

            # 计算平均值
            avg_precision = sum(p['metrics']['precision'] for p in valid_pairs) / len(valid_pairs)
            avg_recall = sum(p['metrics']['recall'] for p in valid_pairs) / len(valid_pairs)
            avg_f1 = sum(p['metrics']['f1'] for p in valid_pairs) / len(valid_pairs)
            total_tp = sum(p['metrics']['tp'] for p in valid_pairs)
            total_fp = sum(p['metrics']['fp'] for p in valid_pairs)
            total_fn = sum(p['metrics']['fn'] for p in valid_pairs)

            all_results[method_name]['average'] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'num_pairs': len(valid_pairs)
            }

            print(f"\n{method_name} 平均性能 (基于{len(valid_pairs)}对相机):")
            print(f"  平均精确率: {avg_precision:.4f}")
            print(f"  平均召回率: {avg_recall:.4f}")
            print(f"  平均F1分数: {avg_f1:.4f}")
            print(f"  总计 TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存重新评估的结果（带时间戳）
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"re_evaluation_results_{timestamp}.json")
        self.save_results(all_results, output_file)

        # 生成对比报告（带时间戳）
        self.generate_comparison_report(all_results, output_dir, timestamp)

        return all_results

    def generate_comparison_report(self, results, output_dir, timestamp=None):
        """
        生成对比报告

        Args:
            results: 评估结果
            output_dir: 输出目录
            timestamp: 时间戳字符串（可选）
        """
        # 如果没有提供时间戳，生成一个
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_file = os.path.join(output_dir, f"comparison_report_{timestamp}.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("跨摄像头轨迹ID匹配方法对比报告\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write("="*80 + "\n\n")

            # 逐对比较
            f.write("1. 各相机对性能对比\n")
            f.write("-"*80 + "\n\n")

            num_pairs = len(self.camera_pairs)
            for i in range(num_pairs):
                query_cam, gallery_cam, _ = self.camera_pairs[i]
                f.write(f"相机对 {i+1}: {query_cam} -> {gallery_cam}\n")

                for method_name, method_label in [
                    ('origin', 'Origin方法'),
                    # ('with_view_method1', 'With_View方法1'),
                    ('with_view_method2', 'With_View方法2')
                ]:
                    pair_result = results[method_name]['pairs'][i]
                    if 'metrics' in pair_result:
                        m = pair_result['metrics']
                        f.write(f"  {method_label:20s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}\n")
                    else:
                        f.write(f"  {method_label:20s}: 评估失败\n")
                f.write("\n")

            # 平均性能对比
            f.write("\n2. 平均性能对比\n")
            f.write("-"*80 + "\n\n")
            f.write(f"{'方法':<20s} {'精确率':<12s} {'召回率':<12s} {'F1分数':<12s}\n")
            f.write("-"*80 + "\n")

            for method_name, method_label in [
                ('origin', 'Origin方法'),
                # ('with_view_method1', 'With_View方法1'),
                ('with_view_method2', 'With_View方法2')
            ]:
                avg = results[method_name]['average']
                if avg:
                    f.write(f"{method_label:<20s} {avg['precision']:<12.4f} {avg['recall']:<12.4f} {avg['f1']:<12.4f}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"\n对比报告已保存到: {report_file}")


if __name__ == "__main__":
    # 配置参数
    crops_dir = "data/datasets/bpbreid_dajixiang/crops_202603"
    label_dir = "data/datasets/link_dajixiang/link_label"
    reid_config_file = "bpbreid/configs/bpbreid/bpbreid_inference.yaml"
    view_model_path = "classify/view/checkpoints/best_model.pth"
    output_dir = "link/evaluation_results"

    only_run_metrics = False
    matches_results_file = "link/evaluation_results/evaluation_results.json"

    # 创建评估器
    evaluator = LinkEvaluator(
        crops_dir=crops_dir,
        label_dir=label_dir,
        reid_config_file=reid_config_file,
        view_model_path=view_model_path,
        only_run_metrics=only_run_metrics
    )

    # 运行评估
    if not only_run_metrics:
        results = evaluator.run_evaluation(output_dir=output_dir)
    else:
        results = evaluator.run_metrics(matches_results_file=matches_results_file, output_dir=output_dir)

    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)
