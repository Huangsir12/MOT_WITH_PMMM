import shutil
import os
import cv2


params_x = 0.06       # x轴异常边界比列
params_y = 0.06       # y轴异常边界比列
disappeared_consecutive_frames = 12        # 消失持续帧数
advented_consecutive_frames = 10           # 出现持续帧数
gallery_frame_interval = 6                 # 放入gallery集的同ids间隔帧数
query_frame_interval = 3                   # 放入query集的同ids间隔帧数
frame_clear_interval = 200                 # 据询问集图片帧相差100帧以上的帧清除


def get_abnormal_added_track_ids(advented_track_ids, boxes, track_ids, track_ids_before, frame_width, frame_height):
    # get the abnormal added track ids
    added =  []
    inter_disappear = []
    for track_id in advented_track_ids:
        if track_id in track_ids:
            advented_track_ids[track_id] += 1
            if advented_track_ids[track_id] >= advented_consecutive_frames:
                added.append(track_id)
        # else:
        #     inter_disappear.append(track_id)
        #     continue
    
    for track_id in added:          
        del advented_track_ids[track_id]
    # for track_id in inter_disappear:
    #     del advented_track_ids[track_id]
    # 如果新加入的跟踪id在track_ids_before中找不到，并且在boxes中不超出参数x, y的百分比
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box[:4])
        if track_id not in track_ids_before:
            if ((x1 > params_x*frame_width and x2 < (1-params_x)*frame_width) 
            and (y1 > params_y*frame_height and y2 < (1-params_y)*frame_height)):
                
                advented_track_ids[track_id] = 1
                
    return added, advented_track_ids


def get_abnormal_removed_track_ids(disappeared_track_ids, track_ids, boxes_before, track_ids_before, frame_width, frame_height):
    # get the abnormal removed track ids
    removed = []
    inter_advent = []
    # 如果消失帧数超过5帧，则认为跟踪已经消失
    for track_id in disappeared_track_ids:
        if track_id not in track_ids:
            disappeared_track_ids[track_id] += 1
            if disappeared_track_ids[track_id] >= disappeared_consecutive_frames:
                removed.append(track_id)
        else:
            inter_advent.append(track_id)
            continue

    for track_id in removed:          
        del disappeared_track_ids[track_id]
    for track_id in inter_advent:
        del disappeared_track_ids[track_id]

    for box, track_id in zip(boxes_before, track_ids_before):
        x1, y1, x2, y2 = map(int, box[:4])

        if track_id not in track_ids:
            if ((x1 > params_x*frame_width and x2 < (1-params_x)*frame_width) 
            and (y1 > params_y*frame_height and y2 < (1-params_y)*frame_height)):
                
                disappeared_track_ids[track_id] = 1

    return removed, disappeared_track_ids


def add_to_gallery_dataset(clops_output_dir, gallery_dataset_dir, abnormal_removed):
    havend_add_ids = []
    for track_id in abnormal_removed:
        file_name = []
        for file in os.listdir(clops_output_dir):
            frame_id  = int(file.split("_")[1])
            if file.endswith(f"ID_{track_id}.jpg") and (frame_id % gallery_frame_interval  == 0):
                file_name.append(file)
        if len(file_name) >=3:
            for item in file_name:
                shutil.copy(os.path.join(clops_output_dir, item), os.path.join(gallery_dataset_dir, item))
            havend_add_ids.append(track_id)
    return havend_add_ids    


def add_to_query_dataset(clops_output_dir, query_dataset_dir, abnormal_added):
    pids_count = {}
    for track_id in abnormal_added:
        file_name = []
        for file in os.listdir(clops_output_dir):
            frame_id  = int(file.split("_")[1])
            if file.endswith(f"ID_{track_id}.jpg") and (frame_id % query_frame_interval  == 0):
                file_name.append(file)
        for item in file_name:
            shutil.copy(os.path.join(clops_output_dir, item), os.path.join(query_dataset_dir, item))

        pids_count[track_id] = len(file_name)
    return pids_count

def get_pictures_counts(dataset_dir):
    return len(os.listdir(dataset_dir))

def remove_from_ReID_dataset(dataset_dir, remove_ids):
    for track_id in remove_ids:
        for file in os.listdir(dataset_dir):
            if file.endswith(f"ID_{track_id}.jpg"):
                file_path = os.path.join(dataset_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

def clear_gallery_dataset_frame(dataset_dir, frame_id):
    for file in os.listdir(dataset_dir):
        if frame_id - int(file.split("_")[1]) > frame_clear_interval:
            file_path = os.path.join(dataset_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def clear_gallery_dataset(dataset_dir, frame_id, pids):
    pid_frame_interval = {key:300 for key in pids}
    for file in os.listdir(dataset_dir):
        pid = int(file.split("_")[3].split(".")[0])
        frame_interval = frame_id - int(file.split("_")[1])
        if frame_interval < pid_frame_interval[pid]:
            pid_frame_interval[pid] = frame_interval
    
    clear_ids = [key for key in pid_frame_interval if pid_frame_interval[key] > 150]
    remove_from_ReID_dataset(dataset_dir, clear_ids)
    for item in clear_ids:
        pids.remove(item)
    return pids,  clear_ids


def renew_track_ids(track_results_ids, matched):
    matched_dict = {}
    for item in matched:
        matched_dict.update(item)
    updated_dict = {}
    # 遍历字典中的每个键值对
    for key, value in matched_dict.items():
        # 检查值是否为另一个键
        if value in matched_dict:
            # 如果是，更新键对应的值为对应的值
            updated_dict[key] = matched_dict[value]
        else:
            # 如果不是，直接添加到新字典
            updated_dict[key] = value
    
    updated_track_ids = []

    # 遍历每一帧的 track_ids
    for frame_track_ids in track_results_ids:
        updated_frame_track_ids = []
        # 遍历当前帧的每个 track_id
        for track_id in frame_track_ids:
            if track_id in updated_dict:
                new_track_id = updated_dict[track_id]
            else:
                new_track_id = track_id
            updated_frame_track_ids.append(new_track_id)
        updated_track_ids.append(updated_frame_track_ids)

    return updated_track_ids

    