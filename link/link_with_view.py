"""
跨摄像头轨迹ID匹配 - 结合视角信息
实现两种方式：
方式一：视角分类后分堆匹配
方式二：视角特征串联匹配
"""

import os
import yaml
import sys
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")
from bpbreid.torchreid.scripts.reID_app import inference_reid_init
from tracking.pmmm_scripts.scripts import *
from classify.view.inference import ViewPointPredictor


class LinkWithView:
    """结合视角信息的跨摄像头轨迹ID匹配"""

    def __init__(self, input_dir, reid_config_file=None, view_model_path=None, query_folder=None, gallery_folder=None):
        """
        Args:
            input_dir: 数据集目录，包含query和gallery文件夹
            reid_config_file: ReID配置文件路径
            view_model_path: 视角分类模型路径
        """
        self.dataset_dir = input_dir

        # 初始化ReID模型
        if reid_config_file:
            with open(reid_config_file, 'r') as file:
                doc = yaml.safe_load(file)
            doc['inference']['dataset_folder'] = self.dataset_dir
            with open(reid_config_file, 'w') as file:
                yaml.safe_dump(doc, file, default_flow_style=None)
            self.reid_inference = inference_reid_init(reid_config_file)
        else:
            self.reid_inference = None

        # 初始化视角分类模型
        if view_model_path and os.path.exists(view_model_path):
            print(f"加载视角分类模型: {view_model_path}")
            self.view_predictor = ViewPointPredictor(view_model_path, device="cuda")
        else:
            print("警告: 未提供视角分类模型，将使用默认路径")
            default_path = "classify/view/checkpoints/best_model.pth"
            if os.path.exists(default_path):
                self.view_predictor = ViewPointPredictor(default_path, device="cuda")
            else:
                raise FileNotFoundError(f"视角分类模型未找到: {view_model_path or default_path}")

        # 视角类别定义 (朝向2类 × 角度5类 = 10种组合，但实际可能只有9类)
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
        return -1  # 未知类别

    def classify_images_by_view(self, image_folder):
        """
        对文件夹中的所有图片进行视角分类

        Returns:
            dict: {view_class_id: [image_paths]}
        """
        print(f"\n开始对 {image_folder} 进行视角分类...")
        image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))

        if len(image_paths) == 0:
            print(f"警告: {image_folder} 中没有找到图片")
            return {}

        # 批量预测
        results = self.view_predictor.predict_batch(image_paths, batch_size=32)

        # 按视角分类
        view_groups = defaultdict(list)
        for result in results:
            orientation = result['orientation']
            angle = result['angle']
            view_id = self._get_view_class_id(orientation, angle)
            view_groups[view_id].append(result['path'])

        # 统计
        print(f"视角分类完成，共 {len(image_paths)} 张图片")
        for view_id, paths in view_groups.items():
            if view_id >= 0:
                print(f"  视角 {self.view_classes[view_id]}: {len(paths)} 张")
            else:
                print(f"  未知视角: {len(paths)} 张")

        return view_groups

    def get_counts(self, folder_path):
        """统计指定文件夹中每个ID对应的图片数量"""
        pids_count = {}

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.png')):
                try:
                    parts = file_name.split('_')
                    track_id = int(parts[0])
                    pids_count[track_id] = pids_count.get(track_id, 0) + 1
                except (ValueError, IndexError):
                    print(f"跳过格式异常文件: {file_name}")

        return pids_count

    def get_frame_ranges(self, folder_path):
        """
        统计指定文件夹中每个ID的最小和最大帧号

        Args:
            folder_path: 数据集文件夹路径

        Returns:
            dict: {track_id: {'min_frame': int, 'max_frame': int}}
        """
        frame_ranges = {}

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.png')):
                try:
                    parts = file_name.split('_')
                    track_id = int(parts[0])
                    frame_num = int(parts[2])

                    if track_id not in frame_ranges:
                        frame_ranges[track_id] = {'min_frame': frame_num, 'max_frame': frame_num}
                    else:
                        frame_ranges[track_id]['min_frame'] = min(frame_ranges[track_id]['min_frame'], frame_num)
                        frame_ranges[track_id]['max_frame'] = max(frame_ranges[track_id]['max_frame'], frame_num)
                except (ValueError, IndexError):
                    print(f"跳过格式异常文件: {file_name}")

        return frame_ranges

    def clear_datasets_func(self, pids_counts, folder_path, min_count=5):
        """删除图片数量少于min_count的ID"""
        delete_ids = [str(tid) for tid, count in pids_counts.items() if count < min_count]

        if not delete_ids:
            print(f"所有ID的图片数量均≥{min_count}张，无需删除")
            return

        print(f"检测到需删除的ID（数量<{min_count}张）: {', '.join(delete_ids)}")
        deleted_count = 0

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.png')):
                try:
                    parts = file_name.split('_')
                    track_id = parts[-1].split('.')[0]
                    if track_id in delete_ids:
                        os.remove(os.path.join(folder_path, file_name))
                        deleted_count += 1
                except Exception as e:
                    continue

        print(f"删除完成，共删除 {deleted_count} 个文件")

    # ==================== 方式一：视角分类后分堆匹配 ====================

    def method1_split_by_view(self, query_dir, gallery_dir, distance_threshold=0.5):
        """
        方式一：视角分类后分堆匹配

        步骤：
        1. 视角分类：获取所有图片的视角分类值
        2. 归类：将gallery集和query集分堆，相同视角的图片放在一起
        3. 遍历匹配：遍历9个视角的小gallery集和query集，获取匹配结果
        4. 确定最终匹配结果：整合9个视角的匹配结果

        Args:
            distance_threshold: 距离阈值，默认0.5

        Returns:
            dict: {query_id: gallery_id} 最终匹配结果
        """
        print("\n" + "="*60)
        print("方式一：视角分类后分堆匹配")
        print("="*60)

        # 步骤1: 视角分类
        query_view_groups = self.classify_images_by_view(query_dir)
        gallery_view_groups = self.classify_images_by_view(gallery_dir)

        # 步骤2: 创建临时文件夹并归类
        temp_dir = os.path.join(self.dataset_dir, "temp_view_split")
        os.makedirs(temp_dir, exist_ok=True)

        view_match_results = {}  # {view_id: {query_id: [(gallery_id, distance)]}}

        # 步骤3: 遍历每个视角进行匹配
        for view_id in set(query_view_groups.keys()) | set(gallery_view_groups.keys()):
            if view_id < 0:  # 跳过未知视角
                continue

            query_paths = query_view_groups.get(view_id, [])
            gallery_paths = gallery_view_groups.get(view_id, [])

            if len(query_paths) == 0 or len(gallery_paths) == 0:
                print(f"\n视角 {self.view_classes[view_id]}: query或gallery为空，跳过")
                continue

            print(f"\n处理视角 {self.view_classes[view_id]}: query={len(query_paths)}, gallery={len(gallery_paths)}")

            # 创建临时视角文件夹
            view_temp_dir = os.path.join(temp_dir, f"view_{view_id}")
            view_query_dir = os.path.join(view_temp_dir, "query")
            view_gallery_dir = os.path.join(view_temp_dir, "gallery")
            os.makedirs(view_query_dir, exist_ok=True)
            os.makedirs(view_gallery_dir, exist_ok=True)

            # 复制文件到临时文件夹
            for path in query_paths:
                shutil.copy(path, view_query_dir)
            for path in gallery_paths:
                shutil.copy(path, view_gallery_dir)

            # 更新ReID配置指向临时文件夹
            original_dataset_folder = self.dataset_dir
            self.dataset_dir = view_temp_dir

            # 运行ReID匹配
            query_pids_counts = self.get_counts(view_query_dir)
            matches = self.reid_inference.run_tracking(query_pids_counts, view_query_dir, view_gallery_dir)

            # 恢复原始配置
            self.dataset_dir = original_dataset_folder

            # 保存该视角的匹配结果
            view_match_results[view_id] = {}
            for query_id, gallery_info in matches.items():
                if query_id not in view_match_results[view_id]:
                    view_match_results[view_id][query_id] = []

                view_match_results[view_id][query_id].append(gallery_info)  

            print(f"视角 {self.view_classes[view_id]} 匹配完成: {len(matches)} 对")

        # 步骤4: 整合所有视角的匹配结果
        final_matches = self._merge_view_matches(view_match_results, distance_threshold)

        # 清理临时文件夹
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        print(f"\n方式一匹配完成，最终匹配对数: {len(final_matches)}")
        return final_matches

    def _merge_view_matches(self, view_match_results, distance_threshold):
        """
        整合多个视角的匹配结果
        若同一个query_id匹配到不同gallery_id，则通过距离和数量综合确定
        """
        print("\n整合多视角匹配结果...")

        # 收集所有query_id的候选匹配
        query_candidates = defaultdict(list)  # {query_id: [(gallery_id, distance, view_id)]}

        for view_id, matches in view_match_results.items():
            for query_id, gallery_list in matches.items():
                for gallery_id, distance in gallery_list:
                    if distance < distance_threshold:
                        query_candidates[query_id].append((gallery_id, distance, view_id))

        # 对每个query_id确定最终匹配
        final_matches = {}
        print(query_candidates)
        for query_id, candidates in query_candidates.items():
            if len(candidates) == 0:
                continue

            # 按gallery_id分组
            gallery_votes = defaultdict(list)  # {gallery_id: [distances]}
            for gallery_id, distance, view_id in candidates:
                gallery_votes[gallery_id].append(distance)

            # 选择最佳匹配：优先考虑出现次数，其次考虑平均距离
            best_gallery_id = None
            best_score = float('inf')

            for gallery_id, distances in gallery_votes.items():
                count = len(distances)
                avg_distance = np.mean(distances)
                # 综合得分：距离越小越好，次数越多越好
                score = avg_distance / (count ** 0.5)  # 简单的综合策略

                if score < best_score:
                    best_score = score
                    best_gallery_id = gallery_id

            if best_gallery_id is not None:
                final_matches[query_id] = best_gallery_id

        return final_matches

    # ==================== 方式二：视角特征串联匹配 ====================

    def method2_concatenate_features(self, query_dir, gallery_dir, distance_threshold=0.58):
        """
        方式二：视角特征串联匹配

        步骤：
        1. 视角分类：获取所有图片的视角分类值
        2. 过滤：相同ID且视角相同的图片，只保留置信得分最高的图片
        3. 特征提取：获取所有crops的外观特征
        4. 特征组合：按固定顺序串联不同视角的外观特征
        5. 时空约束：提取每个ID的帧号范围
        6. 匹配：计算所有视角的平均距离，应用时空约束和最小匹配原则

        Args:
            distance_threshold: 匹配距离阈值，默认0.58

        Returns:
            dict: {query_id: gallery_id} 最终匹配结果
        """
        print("\n" + "="*60)
        print("方式二：视角特征串联匹配")
        print("="*60)

        # 步骤1: 视角分类
        query_view_info = self._classify_and_get_confidence(query_dir)
        gallery_view_info = self._classify_and_get_confidence(gallery_dir)

        # 步骤2: 过滤 - 每个ID每个视角只保留置信度最高的图片
        query_filtered = self._filter_by_confidence(query_view_info)
        gallery_filtered = self._filter_by_confidence(gallery_view_info)

        print(f"\n过滤后: query={len(query_filtered)} 张, gallery={len(gallery_filtered)} 张")

        # 步骤3: 特征提取
        query_features = self._extract_features_for_images(query_filtered)
        gallery_features = self._extract_features_for_images(gallery_filtered)

        # 步骤4: 特征组合 - 按视角顺序串联
        query_combined = self._combine_features_by_view(query_features, query_filtered)
        gallery_combined = self._combine_features_by_view(gallery_features, gallery_filtered)

        # 步骤5: 提取帧号范围（时空约束）
        print("\n提取帧号范围...")
        query_frame_ranges = self.get_frame_ranges(query_dir)
        gallery_frame_ranges = self.get_frame_ranges(gallery_dir)
        print(f"Query帧号范围: {len(query_frame_ranges)} 个ID")
        print(f"Gallery帧号范围: {len(gallery_frame_ranges)} 个ID")

        # 步骤6: 匹配（应用时空约束和最小匹配原则）
        matches = self._match_with_view_features(
            query_combined, gallery_combined,
            query_frame_ranges, gallery_frame_ranges,
            distance_threshold
        )

        print(f"\n方式二匹配完成，最终匹配对数: {len(matches)}")
        return matches

    def _classify_and_get_confidence(self, image_folder):
        """
        对图片进行视角分类并获取置信度

        Returns:
            list: [(image_path, track_id, view_id, confidence)]
        """
        image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))

        if len(image_paths) == 0:
            return []

        results = self.view_predictor.predict_batch(image_paths, batch_size=32)

        view_info = []
        for result in results:
            path = result['path']
            # 从文件名提取track_id
            filename = os.path.basename(path)
            try:
                track_id = int(filename.split('_')[0])
            except:
                continue

            orientation = result['orientation']
            angle = result['angle']
            view_id = self._get_view_class_id(orientation, angle)

            # 综合置信度
            confidence = (result['orientation_conf'] + result['angle_conf']) / 2

            view_info.append((path, track_id, view_id, confidence))

        return view_info

    def _filter_by_confidence(self, view_info):
        """
        过滤：相同ID且视角相同的图片，只保留置信度最高的

        Returns:
            list: [(image_path, track_id, view_id, confidence)]
        """
        # 按 (track_id, view_id) 分组
        groups = defaultdict(list)
        for item in view_info:
            path, track_id, view_id, confidence = item
            groups[(track_id, view_id)].append(item)

        # 每组只保留置信度最高的
        filtered = []
        for key, items in groups.items():
            best_item = max(items, key=lambda x: x[3])  # 按confidence排序
            filtered.append(best_item)

        return filtered

    def _extract_features_for_images(self, image_info):
        """
        提取图片的ReID特征
        使用 reid_inference 中已有的特征提取方法

        Args:
            image_info: [(image_path, track_id, view_id, confidence)]

        Returns:
            dict: {image_path: feature_vector}
        """
        print(f"\n提取 {len(image_info)} 张图片的ReID特征...")

        # 直接使用已有的特征提取方法
        batch_paths = [item[0] for item in image_info]

        # 使用 Inference 类的 extract_part_based_features 方法
        embeddings, visibility_scores, parts_masks = self.reid_inference.extract_part_based_features(
            self.reid_inference.extractor,
            batch_paths,
            batch_size=16
        )

        print(embeddings.shape)

        # 构建特征字典
        features_dict = {}
        for i, (path, _, _, _) in enumerate(image_info):
            # 取全局特征或前景特征 (第一个特征)
            feature = embeddings[i, 0, :].cpu().numpy()  # [D]
            features_dict[path] = feature

        print(f"特征提取完成，特征维度: {feature.shape}")
        return features_dict

    def _combine_features_by_view(self, features_dict, image_info):
        """
        按视角顺序串联特征

        Args:
            features_dict: {image_path: feature_vector}
            image_info: [(image_path, track_id, view_id, confidence)]

        Returns:
            dict: {track_id: {view_id: feature_vector}}
        """
        print("\n组合视角特征...")

        # 按track_id分组
        track_features = defaultdict(dict)
        for path, track_id, view_id, confidence in image_info:
            if path in features_dict:
                track_features[track_id][view_id] = features_dict[path]

        return track_features

    def _get_view_similarity_matrix(self):
        """
        定义视角间的相似度矩阵
        返回每对视角的相似度（值越大越相似）

        视角编码：
        - 0-4: face (0°, 45°, 90°, 135°, 180°)
        - 5-9: back (0°, 45°, 90°, 135°, 180°)

        Returns:
            dict: {(view_id1, view_id2): similarity_score}
        """
        similarity_matrix = {}

        for i in range(10):
            for j in range(10):
                ori_i = i // 5  # 0=face, 1=back
                angle_i = i % 5  # 0-4

                ori_j = j // 5
                angle_j = j % 5

                # 相同视角，相似度最高
                if i == j:
                    similarity = 1.0
                # 相同朝向，角度差异
                elif ori_i == ori_j:
                    angle_diff = abs(angle_i - angle_j)
                    similarity = 1.0 - angle_diff * 0.15  # 每差一个角度降低0.15
                # 不同朝向，相同角度
                elif angle_i == angle_j:
                    similarity = 0.5
                # 不同朝向，不同角度
                else:
                    angle_diff = abs(angle_i - angle_j)
                    similarity = 0.5 - angle_diff * 0.08

                similarity_matrix[(i, j)] = max(0.0, similarity)

        return similarity_matrix

    def _match_with_view_features(self, query_combined, gallery_combined,
                                   query_frame_ranges, gallery_frame_ranges,
                                   distance_threshold):
        """
        使用组合的视角特征进行匹配，应用时空约束和最小匹配原则

        时空约束：query集匹配ID的最大帧号不能大于gallery集匹配ID的最小帧号
        最小匹配原则：先获取所有满足条件的匹配对，从最小特征距离开始匹配，已匹配的gallery ID不能再次被匹配

        新策略：
        1. 使用固定的视角相似度矩阵
        2. 对每对query-gallery，按相似度从高到低选择视角对进行匹配
        3. 只要距离小于阈值，即作为备选
        4. 应用时空约束过滤
        5. 按距离排序，贪心匹配
        """
        print("\n计算视角特征距离并匹配...")

        # 获取视角相似度矩阵
        view_similarity = self._get_view_similarity_matrix()

        # 收集所有候选匹配对 (query_id, gallery_id, distance)
        candidate_matches = []

        for query_id, query_views in query_combined.items():
            # 检查query_id是否有帧号信息
            if query_id not in query_frame_ranges:
                print(f"警告: query ID {query_id} 没有帧号信息，跳过")
                continue

            query_max_frame = query_frame_ranges[query_id]['max_frame']

            for gallery_id, gallery_views in gallery_combined.items():
                # 检查gallery_id是否有帧号信息
                if gallery_id not in gallery_frame_ranges:
                    continue

                gallery_min_frame = gallery_frame_ranges[gallery_id]['min_frame']

                # 时空约束：query的最大帧号必须 <= gallery的最小帧号
                if query_max_frame > gallery_min_frame + 300:
                    continue

                # 获取query和gallery的可用视角
                query_view_ids = [v for v in query_views.keys() if query_views[v] is not None and not np.allclose(query_views[v], 0)]
                gallery_view_ids = [v for v in gallery_views.keys() if gallery_views[v] is not None and not np.allclose(gallery_views[v], 0)]

                if len(query_view_ids) == 0 or len(gallery_view_ids) == 0:
                    continue

                # 构建所有可能的视角对，并按相似度排序
                view_pairs = []
                for q_view in query_view_ids:
                    for g_view in gallery_view_ids:
                        similarity = view_similarity.get((q_view, g_view), 0.0)
                        view_pairs.append((q_view, g_view, similarity))

                # 按相似度从高到低排序
                view_pairs.sort(key=lambda x: x[2], reverse=True)

                # 按相似度顺序计算距离，取第一个满足条件的
                matched_distance = None
                used_views = []

                for q_view, g_view, similarity in view_pairs:
                    query_feat = query_views[q_view]
                    gallery_feat = gallery_views[g_view]

                    # 计算余弦距离
                    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
                    gallery_norm = gallery_feat / (np.linalg.norm(gallery_feat) + 1e-8)
                    cosine_sim = np.dot(query_norm, gallery_norm)
                    distance = 1 - cosine_sim

                    # 只要距离小于阈值，即作为备选
                    if distance < distance_threshold:
                        matched_distance = distance
                        used_views = [(q_view, g_view, similarity)]
                        break

                # 如果找到满足条件的匹配
                if matched_distance is not None:
                    candidate_matches.append((query_id, gallery_id, matched_distance, used_views))

        print(f"\n满足条件的候选匹配对数: {len(candidate_matches)}")

        # 按距离排序（从小到大）
        candidate_matches.sort(key=lambda x: x[2])

        # 贪心匹配：从最小距离开始，已匹配的gallery ID不能再次被匹配
        matched_gallery_ids = set()
        final_matches = {}

        for query_id, gallery_id, distance, used_views in candidate_matches:
            # 如果该gallery_id已经被匹配，跳过
            if gallery_id in matched_gallery_ids:
                continue

            # 匹配成功
            final_matches[query_id] = gallery_id
            matched_gallery_ids.add(gallery_id)

            view_info = f"视角对: {self.view_classes[used_views[0][0]]} <-> {self.view_classes[used_views[0][1]]} (相似度={used_views[0][2]:.2f})"
            print(f"匹配: query {query_id} -> gallery {gallery_id}, 距离={distance:.4f}, {view_info}")

        return final_matches

    def _compute_cross_view_distances(self, query_views, gallery_views, exclude_view_ids):
        """
        计算非相同视角之间的特征距离（优先相邻视角）

        Args:
            query_views: query的视角特征字典 {view_id: feature}
            gallery_views: gallery的视角特征字典 {view_id: feature}
            exclude_view_ids: 已经计算过的视角ID列表

        Returns:
            list: 按优先级排序的跨视角距离列表
        """
        cross_distances = []

        # 获取可用的query和gallery视角
        query_view_ids = [v for v in query_views.keys() if v not in exclude_view_ids]
        gallery_view_ids = [v for v in gallery_views.keys() if v not in exclude_view_ids]

        # 定义视角相邻关系（基于视角ID的接近程度）
        # 视角类别：face_0°, face_45°, face_90°, face_135°, face_180°, back_0°, back_45°, back_90°, back_135°, back_180°
        # 相邻定义：同朝向且角度相邻，或相同角度但朝向不同

        for q_view in query_view_ids:
            query_feat = query_views[q_view]
            if query_feat is None or np.allclose(query_feat, 0):
                continue

            for g_view in gallery_view_ids:
                gallery_feat = gallery_views[g_view]
                if gallery_feat is None or np.allclose(gallery_feat, 0):
                    continue

                # 计算余弦距离
                query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
                gallery_norm = gallery_feat / (np.linalg.norm(gallery_feat) + 1e-8)
                cosine_sim = np.dot(query_norm, gallery_norm)
                distance = 1 - cosine_sim

                # 计算视角相邻度（用于排序优先级）
                adjacency_score = self._compute_view_adjacency(q_view, g_view)

                cross_distances.append((distance, adjacency_score, q_view, g_view))

        # 按相邻度排序（相邻度越小越优先）
        cross_distances.sort(key=lambda x: x[1])

        # 返回距离列表
        return [d[0] for d in cross_distances]

    def _compute_view_adjacency(self, view_id1, view_id2):
        """
        计算两个视角之间的相邻度（值越小越相邻）

        视角编码：
        - 0-4: face (0°, 45°, 90°, 135°, 180°)
        - 5-9: back (0°, 45°, 90°, 135°, 180°)
        """
        # 提取朝向和角度
        orientation1 = view_id1 // 5  # 0=face, 1=back
        angle1 = view_id1 % 5         # 0-4 对应 0°, 45°, 90°, 135°, 180°

        orientation2 = view_id2 // 5
        angle2 = view_id2 % 5

        # 计算角度差异（考虑循环）
        angle_diff = min(abs(angle1 - angle2), 5 - abs(angle1 - angle2))

        # 计算朝向差异
        orientation_diff = abs(orientation1 - orientation2)

        # 相邻度 = 角度差异 + 朝向差异权重
        adjacency = angle_diff + orientation_diff * 2

        return adjacency

    # ==================== 主运行函数 ====================

    def run_link(self, method="method1", **kwargs):
        """
        运行跨摄像头轨迹ID匹配

        Args:
            method: "method1" 或 "method2"
            **kwargs: 方法特定的参数

        Returns:
            dict: {query_id: gallery_id} 匹配结果
        """
        if method == "method1":
            return self.method1_split_by_view(**kwargs)
        elif method == "method2":
            return self.method2_concatenate_features(**kwargs)
        else:
            raise ValueError(f"未知方法: {method}，请选择 'method1' 或 'method2'")


if __name__ == "__main__":
    # 配置参数
    config_file = "bpbreid/configs/bpbreid/bpbreid_inference.yaml"
    dataset_folder = "data/datasets/bpbreid_dajixiang/crops_202603"
    view_model_path = "classify/view/checkpoints/best_model.pth"

    query_folder = os.path.join(dataset_folder, "camera_001")
    gallery_folder = os.path.join(dataset_folder, "camera_002")

    # 初始化
    link = LinkWithView(
        input_dir=dataset_folder,
        reid_config_file=config_file,
        view_model_path=view_model_path,
    )

    # 可选：清理数据集
    clear_dataset = False
    if clear_dataset:
        query_pids_counts = link.get_counts(type="query")
        gallery_pids_counts = link.get_counts(type="gallery")
        link.clear_datasets_func(query_pids_counts, query_folder, min_count=3)
        link.clear_datasets_func(gallery_pids_counts, gallery_folder, min_count=5)

    # # 方式一：视角分类后分堆匹配
    # print("\n" + "="*80)
    # print("测试方式一：视角分类后分堆匹配")
    # print("="*80)
    # matches_method1 = link.run_link(method="method1", 
    #                                 query_dir=query_folder, 
    #                                 gallery_dir=gallery_folder, 
    #                                 distance_threshold=0.7)
    # print(f"\n方式一匹配结果示例: {dict(list(matches_method1.items())[:10])}")

    # 方式二：视角特征串联匹配
    print("\n" + "="*80)
    print("测试方式二：视角特征串联匹配")
    print("="*80)
    matches_method2 = link.run_link(method="method2", 
                                    query_dir=query_folder, 
                                    gallery_dir=gallery_folder, 
                                    distance_threshold=0.7)
    # print(f"\n方式二匹配结果示例: {dict(list(matches_method2.items())[:30])}")
    print(f"\n方式二匹配结果示例: {matches_method2}")

    # # 比较两种方法
    # print("\n" + "="*80)
    # print("两种方法对比")
    # print("="*80)
    # print(f"方式一匹配对数: {len(matches_method1)}")
    # print(f"方式二匹配对数: {len(matches_method2)}")

    # # 计算一致性
    # common_queries = set(matches_method1.keys()) & set(matches_method2.keys())
    # consistent = sum(1 for q in common_queries if matches_method1[q] == matches_method2[q])
    # if len(common_queries) > 0:
    #     consistency = consistent / len(common_queries) * 100
    #     print(f"两种方法的一致性: {consistency:.2f}% ({consistent}/{len(common_queries)})")
