import os
import yaml
from bpbreid.torchreid.scripts.reID_app import inference_reid_init
from datetime import datetime
from pmmm_scripts.scripts import *
import cv2

class TrackReid_PMMM:
    def __init__(self, output_dir, reid_congig_file=None, txt_name=None):

        if txt_name:
            self.output_dir = os.path.join(output_dir, txt_name)
        else:
            self.output_dir = os.path.join(output_dir, datetime.now().strftime("%m%d-%H%M"))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if reid_congig_file:
            with open(reid_congig_file, 'r') as file:
                doc = yaml.safe_load(file)

            doc['inference']['dataset_folder'] = self.output_dir

            with open(reid_congig_file, 'w') as file:
                yaml.safe_dump(doc, file, default_flow_style=None)

            self.reid_inference = inference_reid_init(reid_congig_file)
        else:
            self.reid_inference = None

        self.clops_output_dir = os.path.join(self.output_dir, "clops")
        if not os.path.exists(self.clops_output_dir):
            os.makedirs(self.clops_output_dir, exist_ok=True)

        self.gallery_dataset_dir = os.path.join(self.output_dir, "gallery")
        if not os.path.exists(self.gallery_dataset_dir):
            os.makedirs(self.gallery_dataset_dir, exist_ok=True)

        self.query_dataset_dir = os.path.join(self.output_dir, "query")
        if not os.path.exists(self.query_dataset_dir):
            os.makedirs(self.query_dataset_dir, exist_ok=True)

        self.advented, self.disappeared = {}, {}
        self.boxes_before, self.track_ids_before = [], []
    

    def processing_to_reid(self, frame, boxes, track_ids, frame_count, frame_width, frame_height, 
                           track_ids_before_all, abnormal_removed):
        matched_id = {}
        # save the boxes to the output directory
        for box, track_id in zip(boxes, track_ids):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box[:4])
            # print(f"track_id: {track_id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            crop = frame[y1:y2, x1:x2]  # Crop the detected object

            # Save the cropped image with the desired naming convention
            crop_name = f"frame_{frame_count}_ID_{track_id}.jpg"
            crop_path = os.path.join(self.clops_output_dir, crop_name)
            if crop.size == 0:
                print(f"警告：裁剪区域为空，边界框: {box}")
                continue
            cv2.imwrite(crop_path, crop)

        if frame_count > 1:
            if frame_count % 100 == 0:
                abnormal_removed = clear_gallery_dataset_frame(self.gallery_dataset_dir, frame_count)
            # Get abnormal track ids
            frame_abnormal_added, self.advented = get_abnormal_added_track_ids(self.advented, boxes, track_ids, 
                                                                track_ids_before_all, 
                                                                frame_width, frame_height)    
            frame_abnormal_removed, self.disappeared = get_abnormal_removed_track_ids(self.disappeared, 
                                                                    track_ids, 
                                                                    self.boxes_before, self.track_ids_before, 
                                                                    frame_width, frame_height)
            
            # add the abnormal removed track ids to the gallery dataset
            if frame_abnormal_removed:
                haven_added_ids = add_to_gallery_dataset(frame_count, self.clops_output_dir, self.gallery_dataset_dir, frame_abnormal_removed)
                abnormal_removed.extend(haven_added_ids)
                abnormal_removed = list(set(abnormal_removed))
                print(f"removed_track_id: {haven_added_ids}")
            if not abnormal_removed:
                frame_abnormal_added = []
            # add the abnormal added track ids to the query dataset
            if frame_abnormal_added:
                abnormal_removed, clear_ids = clear_gallery_dataset(self.gallery_dataset_dir, frame_count, abnormal_removed, track_ids)
                abnormal_removed = clear_gallery_dataset_frame(self.gallery_dataset_dir, frame_count)
                pids_counts = add_to_query_dataset(self.clops_output_dir, self.query_dataset_dir, frame_abnormal_added)
                print(f"clear_ids: {clear_ids}")
                print(f"added_track_id: {pids_counts}")
            
            gallery_counts = get_pictures_counts(self.gallery_dataset_dir)
            query_counts = get_pictures_counts(self.query_dataset_dir)
            # start to reID
            if query_counts > 0:
                # reID the gallery and query dataset
                print("gallery_counts: ", gallery_counts)
                print("query_counts: ", query_counts)
                print("query_ids: ", frame_abnormal_added)
                print("gallery_ids: ", abnormal_removed)
                
                if gallery_counts > 0:
                    matched_id = self.reid_inference.run_tracking(pids_counts)
                    matched_id_values = list(matched_id.values())
                    if matched_id_values:
                        print(matched_id_values)
                        for item in matched_id_values:
                            if item in abnormal_removed:
                                abnormal_removed.remove(item)
                        remove_from_ReID_dataset(self.gallery_dataset_dir, matched_id_values)
                remove_from_ReID_dataset(self.query_dataset_dir, frame_abnormal_added)
            
        self.boxes_before, self.track_ids_before = boxes, track_ids

        return matched_id, abnormal_removed