from __future__ import division, print_function, absolute_import

import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class AnChang110(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 21 (+1 for background).
        - images: 72 (train) + 40 (query) + 120 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'AnChang110'
    masks_base_dir = 'masks'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in AnChang110.masks_dirs:
            return None
        else:
            return AnChang110.masks_dirs[masks_dir]

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.masks_dir = masks_dir

        # allow alternative directory structure
        if not osp.isdir(self.dataset_dir):
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')


        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(AnChang110, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # print(img_paths)

        pid_container = set()
        for img_path in img_paths:
            pid = int(osp.basename(img_path).split('_')[0])
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid = int(osp.basename(img_path).split('_')[0])
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 21 # pid == 0 means background

            camid = 1 
            if relabel:
                pid = pid2label[pid]
            masks_path = self.infer_masks_path(img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         'masks_path': masks_path,
                         'camid': camid})
        return data
