import torch
import tqdm
import glob
import os
import os.path as osp
import numpy as np
from tabulate import tabulate
from torch.nn import functional as F
from torchreid.tools.feature_extractor import FeatureExtractor
from torchreid.tools.automized_data import AutomizedDataLoader
from torchreid.utils.constants import *
from .. import metrics
from ..metrics.distance import compute_distance_matrix_using_bp_features
from ..utils import plot_body_parts_pairs_distance_distribution, \
    plot_pairs_distance_distribution, re_ranking


class Inference():

    def __init__(self, cfg, model, writer):
        self.cfg = cfg
        self.model = model
        self.writer = writer
        self.dataset_folder = cfg.inference.dataset_folder
        self.dataset_name = cfg.inference.dataset_name
        self.use_gpu = cfg.use_gpu
        self.extractor = FeatureExtractor(
            cfg, 
            model_path=cfg.model.load_weights,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_classes=cfg.inference.num_classes,
            model=model
        )
        self.automized_dataloader = AutomizedDataLoader(self.dataset_folder, batch_size=8, num_workers=4)
        self.dist_combine_strat = cfg.test.part_based.dist_combine_strat
        self.batch_size_pairwise_dist_matrix = cfg.test.batch_size_pairwise_dist_matrix
        self.test_embeddings = cfg.model.bpbreid.test_embeddings
        self.detailed_ranking = cfg.test.detailed_ranking
    
    def display_individual_parts_ranking_performances(self, body_parts_distmat, cmc, g_camids, g_pids, mAP, q_camids,
                                                      q_pids, eval_metric):
        print('Parts embeddings individual rankings :')
        bp_offset = 0
        if GLOBAL in self.cfg.model.bpbreid.test_embeddings:
            bp_offset += 1
        if FOREGROUND in self.cfg.model.bpbreid.test_embeddings:
            bp_offset += 1
        table = []
        for bp in range(0, body_parts_distmat.shape[0]):  # TODO DO NOT TAKE INTO ACCOUNT -1 DISTANCES!!!!
            perf_metrics = metrics.evaluate_rank(
                body_parts_distmat[bp],
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric
            )
            title = 'p {}'.format(bp - bp_offset)
            if bp < bp_offset:
                if bp == 0:
                    if GLOBAL in self.config.model.bpbreid.test_embeddings:
                        title = GLOBAL
                    else:
                        title = FOREGROUND
                if bp == 1:
                    title = FOREGROUND
            mAP = perf_metrics['mAP']
            cmc = perf_metrics['cmc']
            table.append([title, mAP, cmc[0], cmc[4], cmc[9]])
        headers = ["embed", "mAP", "R-1", "R-5", "R-10"]
        print(tabulate(table, headers, tablefmt="fancy_grid", floatfmt=".3f"))
    
    def extract_test_embeddings(self, model_output):
            embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
            embeddings_list = []
            visibility_scores_list = []
            embeddings_masks_list = []

            for test_emb in self.test_embeddings:
                embds = embeddings[test_emb]
                embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
                if test_emb in bn_correspondants:
                    test_emb = bn_correspondants[test_emb]
                vis_scores = visibility_scores[test_emb]
                visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
                pt_masks = parts_masks[test_emb]
                embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

            assert len(embeddings) != 0

            embeddings = torch.cat(embeddings_list, dim=1)  # [N, P+2, D]
            visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+2]
            embeddings_masks = torch.cat(embeddings_masks_list, dim=1)  # [N, P+2, Hf, Wf]

            return embeddings, visibility_scores, embeddings_masks


    def extract_part_based_features(self, extractor, image_list, batch_size=4):

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        all_embeddings = []
        all_visibility_scores = []
        all_masks = []

        images_chunks = chunks(image_list, batch_size)
        for chunk in tqdm.tqdm(images_chunks):
            model_output = extractor(chunk)
            # embeddings, visibility_scores, masks, pixels_cls_scores = extractor(chunk)
            embeddings, visibility_scores, parts_masks = self.extract_test_embeddings(model_output)
            embeddings = embeddings.data.cpu().detach()
            visibility_scores = visibility_scores.cpu().detach()
            parts_masks = parts_masks.data.cpu().detach()

            all_embeddings.append(embeddings)
            all_visibility_scores.append(visibility_scores)
            all_masks.append(parts_masks)
            # pxl_scores.append(pixels_cls_scores)

        all_embeddings = torch.cat(all_embeddings, 0)
        all_visibility_scores = torch.cat(all_visibility_scores, 0)
        all_masks = torch.cat(all_masks, 0)
        return all_embeddings, all_visibility_scores, all_masks
        

    def extract_det_idx(self, img_path):
        # return int(os.path.basename(img_path).split("_")[0])
        return int(os.path.basename(img_path).split(".")[0].split("_")[-1])


    def extract_reid_features(self, input_folder):
        extractor = self.extractor
        # print("Looking for video folders with images crops in {}".format(base_folder))
        image_list = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, '*.jpg'))
        image_list.sort(key=self.extract_det_idx)
        print("{} images to process for folder {}".format(len(image_list), input_folder))
        features, visibility_scores, parts_masks = self.extract_part_based_features(extractor, image_list, batch_size=4)
        print('Done, obtained {} tensor'.format(features.shape))

        pids = [self.extract_det_idx(image_path) for image_path in image_list]
        pids = np.asarray(pids)

        camids = np.asarray([1] * len(image_list))

        # # dump to disk
        # video_name = os.path.splitext(os.path.basename(folder))[0]
        # parts_embeddings_filename = os.path.join(out_path, "embeddings_" + video_name + ".npy")
        # parts_visibility_scores_filanme = os.path.join(out_path, "visibility_scores_" + video_name + ".npy")
        # parts_masks_filename = os.path.join(out_path, "masks_" + video_name + ".npy")

        # os.makedirs(os.path.dirname(parts_embeddings_filename), exist_ok=True)
        # os.makedirs(os.path.dirname(parts_visibility_scores_filanme), exist_ok=True)
        # os.makedirs(os.path.dirname(parts_masks_filename), exist_ok=True)

        # np.save(parts_embeddings_filename, results['parts_embeddings'])
        # np.save(parts_visibility_scores_filanme, results['parts_visibility_scores'])
        # np.save(parts_masks_filename, results['parts_masks'])

        # print("features saved to {}".format(out_path))

        return features, pids, camids, visibility_scores, parts_masks
        

    def run(self,
            save_gallery_features,
            load_gallery_features,
            gallery_features_path,
            save_dir,
            normalize_feature=False,
            dist_metric='euclidean',
            rerank=False,
            ranks=[1, 5, 10, 20],
            visrank=False,
            visrank_topk=10,
            visrank_q_idx_list=[],
            visrank_count=10,
            ):

        self.writer.test_timer.start()

        test_loader = self.automized_dataloader.load_test_data()

        query_folder = osp.join(self.dataset_folder, 'query')
        gallery_folder = osp.join(self.dataset_folder, 'gallery')
        qf, q_pids, q_camids, qf_parts_visibility, q_parts_masks = self.extract_reid_features(query_folder)
        q_anns = []

        if load_gallery_features:
                gf, g_pids, g_camids, gf_parts_visibility, g_parts_masks = torch.load(gallery_features_path)
        else:
            gf, g_pids, g_camids, gf_parts_visibility, g_parts_masks = self.extract_reid_features(gallery_folder)
        g_anns = []

        if save_gallery_features:
            features_dir = osp.join(save_dir, 'features')
            print('Saving features to : ' + features_dir)
            if not osp.exists(features_dir):
                os.makedirs(features_dir)

            # TODO create if doesn't exist
            torch.save((gf, g_pids, g_camids, gf_parts_visibility, g_parts_masks), 
                       osp.join(features_dir, 'gallery_features.pt'), pickle_protocol=4)
        
        self.writer.performance_evaluation_timer.start()
        if normalize_feature:
            print('Normalizing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=-1)
            gf = F.normalize(gf, p=2, dim=-1)
        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(qf, gf, qf_parts_visibility,
                                                                                        gf_parts_visibility,
                                                                                        self.dist_combine_strat,
                                                                                        self.batch_size_pairwise_dist_matrix,
                                                                                        self.use_gpu, dist_metric)
        distmat = distmat.numpy()
        body_parts_distmat = body_parts_distmat.numpy()
        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq, body_parts_distmat_qq = compute_distance_matrix_using_bp_features(qf, qf, qf_parts_visibility, qf_parts_visibility,
                                                                    self.dist_combine_strat, self.batch_size_pairwise_dist_matrix,
                                                                    self.use_gpu, dist_metric)
            distmat_gg, body_parts_distmat_gg = compute_distance_matrix_using_bp_features(gf, gf, gf_parts_visibility, gf_parts_visibility,
                                                                        self.dist_combine_strat, self.batch_size_pairwise_dist_matrix,
                                                                        self.use_gpu, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        eval_metric = 'default'

        print('Computing CMC and mAP ...')
        eval_metrics = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_anns=q_anns,
            g_anns=g_anns,
            eval_metric=eval_metric
        )

        mAP = eval_metrics['mAP']
        cmc = eval_metrics['cmc']
        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

        for metric in eval_metrics.keys():
            if metric != 'mAP' and metric != 'cmc':
                val, size = eval_metrics[metric]
                if val is not None:
                    print('{:<20}: {:.2%} ({})'.format(metric, val, size))
                else:
                    print('{:<20}: not provided'.format(metric))

        # Parts ranking
        if self.detailed_ranking:
            self.display_individual_parts_ranking_performances(body_parts_distmat, cmc, g_camids, g_pids, mAP,
                                                                q_camids, q_pids, eval_metric)
        # TODO move below to writer
        plot_body_parts_pairs_distance_distribution(body_parts_distmat, q_pids, g_pids, "Query-gallery")
        print('Evaluate distribution of distances of pairs with same id vs different ids')
        same_ids_dist_mean, same_ids_dist_std, different_ids_dist_mean, different_ids_dist_std, ssmd = \
            plot_pairs_distance_distribution(distmat, q_pids, g_pids,
                                                "Query-gallery")  # TODO separate ssmd from plot, put plot in writer
        print("Positive pairs distance distribution mean: {:.3f}".format(same_ids_dist_mean))
        print("Positive pairs distance distribution standard deviation: {:.3f}".format(same_ids_dist_std))
        print("Negative pairs distance distribution mean: {:.3f}".format(different_ids_dist_mean))
        print("Negative pairs distance distribution standard deviation: {:.3f}".format(
            different_ids_dist_std))
        print("SSMD = {:.4f}".format(ssmd))
        

        if visrank:
            self.writer.visualize_rank(test_loader, self.dataset_name, distmat, save_dir,
                                        visrank_topk, visrank_q_idx_list, visrank_count,
                                        body_parts_distmat, qf_parts_visibility, gf_parts_visibility, q_parts_masks,
                                        g_parts_masks, mAP, cmc[0])

        self.writer.visualize_embeddings(qf, gf, q_pids, g_pids, test_loader, self.dataset_name,
                                            qf_parts_visibility, gf_parts_visibility, mAP, cmc[0])
        self.writer.performance_evaluation_timer.stop()
        return cmc, mAP, ssmd
    

    def run_tracking(self, pids_counts):
        self.writer.test_timer.start()

        query_folder = osp.join(self.dataset_folder, 'query')
        gallery_folder = osp.join(self.dataset_folder, 'gallery')
        qf, q_pids, q_camids, qf_parts_visibility, q_parts_masks = self.extract_reid_features(query_folder)
        gf, g_pids, g_camids, gf_parts_visibility, g_parts_masks = self.extract_reid_features(gallery_folder)

        self.writer.performance_evaluation_timer.start()
        if True:
            print('Normalizing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=-1)
            gf = F.normalize(gf, p=2, dim=-1)
        print('Computing distance matrix with metric={} ...'.format("euclidean"))
        distmat, body_parts_distmat = compute_distance_matrix_using_bp_features(qf, gf, qf_parts_visibility,
                                                                                        gf_parts_visibility,
                                                                                        self.dist_combine_strat,
                                                                                        self.batch_size_pairwise_dist_matrix,
                                                                                        self.use_gpu, "euclidean")
        distmat = distmat.numpy()
        body_parts_distmat = body_parts_distmat.numpy()
        
        print(q_pids)
        print(g_pids)
        print(distmat)
        indices = np.argsort(distmat, axis=1)
        print(indices)

        # get the first 10 ranks
        rank_length = min(10, len(g_pids))
        g_pids_ranks = [g_pids[indices[i, :rank_length]] for i in range(len(q_pids))]
        
        distmat_ranks = [distmat[i][indices[i, :rank_length]] for i in range(len(distmat))]
        print(g_pids_ranks)
        matched_pids = most_common(q_pids, g_pids_ranks, distmat_ranks, pids_counts, rank_length)
        
        print(matched_pids)

        return matched_pids

def most_common(q_pids, g_pids_ranks, distmat_ranks, pids_counts, rank_length):
    matched = {}
    index = 0
    assert len(g_pids_ranks) == sum(pids_counts.values())
    # pids_count 是query集中的类别字典，包含key-value为：pid和图片数量
    for i in range(len(pids_counts)):
        pid = q_pids[index]
        distmat_pid = {}
        for k in range(pids_counts[pid]):
            g_pids_rank = g_pids_ranks[index + k].tolist()
            distmat_rank = distmat_ranks[index + k].tolist()
            most_common_flag = False
            for j in range(4, rank_length+1):      # get the first i ranks
                rank_list_j = g_pids_rank[:j]
                most_common_gallery_id = max(set(rank_list_j), key=rank_list_j.count)
                if rank_list_j.count(most_common_gallery_id) / j >= 0.8:     # 众数频率是否超过0.8
                    most_common_flag = True
                    break
            if most_common_flag:
                most_common_index = [i for i in range(rank_length) if g_pids_rank[i] == most_common_gallery_id]
                most_common_distmat = [distmat_rank[item] for item in most_common_index]
                mean_distmat = sum(most_common_distmat[:3])/3.0
                print(mean_distmat)
                if mean_distmat < 0.7:          # distmat 平均特征距离是否小于0.7
                    distmat_pid[mean_distmat] = most_common_gallery_id    

        if len(distmat_pid) > 0:
            print(distmat_pid)
            matched[pid] = distmat_pid[min(distmat_pid.keys())]

        index = index + pids_counts[pid] 

    return matched