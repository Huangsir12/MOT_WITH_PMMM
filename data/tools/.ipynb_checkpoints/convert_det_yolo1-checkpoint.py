import os 


label_data = "data/datasets/MOT17/train"
val_label_data = "runs/dets_n_embs/yolox_x_ablation/dets"
train_label_output_dir = "data/datasets/MOT17_YOLO/train/labels"
val_label_output_dir = "data/datasets/MOT17_YOLO/valid/labels"
if not os.path.exists(train_label_output_dir):
    os.makedirs(train_label_output_dir)
if not os.path.exists(val_label_output_dir):
    os.makedirs(val_label_output_dir)

# MOT17数据集的帧宽高
# MOT17数据集的帧宽高
frame_wh_json = {"MOT17-01": [1920, 1080],
                 "MOT17-02": [1920, 1080],
                 "MOT17-03": [1920, 1080],
                 "MOT17-04": [1920, 1080],
                 "MOT17-05": [640, 480],
                 "MOT17-06": [640, 480],
                 "MOT17-07": [1920, 1080],
                 "MOT17-08": [1920, 1080],
                 "MOT17-09": [1920, 1080],
                 "MOT17-10": [1920, 1080],
                 "MOT17-11": [1920, 1080],
                 "MOT17-12": [1920, 1080],
                 "MOT17-13": [1920, 1080],
                 "MOT17-14": [1920, 1080],}

for root, dirs, files in os.walk(label_data, topdown=False):
    for dir in dirs:
        train_det_label_path = os.path.join(label_data, dir, "det", "det_train_half.txt")
        val_det_label_path = os.path.join(val_label_data, f"{dir}.txt")
        if dir in frame_wh_json:
            frame_width, frame_height = frame_wh_json[dir]
        else:
            frame_width, frame_height = 1920, 1080
        frame_counts = 0
        frame_numbers = []
        if os.path.exists(train_det_label_path):
            with open(train_det_label_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split(",")
                frame_id = int(line[0])
                if frame_id not in frame_numbers:
                    frame_numbers.append(frame_id)
                    frame_counts += 1
                x, y, w, h = map(float, line[2:6])
                # Convert to YOLO format
                # YOLO format: class_id x_center y_center width height
                # Assuming class_id is 0 for all objects
                class_id = 0
                x_center = (x + w / 2) / frame_width
                y_center = (y + h / 2) / frame_height
                w1 = w / frame_width
                h1 = h / frame_height
                if x_center < 0 or y_center < 0 or w1 < 0 or h1 < 0:
                    print(f"invalid values: {x_center}, {y_center}, {w1}, {h1}")
                    continue
                if x_center > 1 or y_center > 1 or w1 > 1 or h1 > 1:
                    print(f"invalid values: {x_center}, {y_center}, {w1}, {h1}")
                    continue

                line_after = f"{class_id} {x_center:.6f} {y_center:.6f} {w1:.6f} {h1:.6f}\n"
                frame_id_after = str(frame_id).zfill(6)
                file_name= f"{dir}_{frame_id_after}.txt"
                # print(line_after)
                # print(f"{os.path.join(label_output_dir, file_name)}")
                with open(os.path.join(train_label_output_dir, file_name), "a") as f:
                    f.write(line_after)

        if os.path.exists(val_det_label_path):
            with open(val_det_label_path, "r") as f:
                next(f)
                lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                frame_id = int(float(line[0]))
                x1, y1, x2, y2 = map(float, line[1:5])
                # Convert to YOLO format
                # YOLO format: class_id x_center y_center width height
                # Assuming class_id is 0 for all objects
                class_id = 0
                x_center = (x1 + x2) / (2 * frame_width)
                y_center = (y1 + y2) / (2 * frame_height)
                w1 = (x2 - x1) / frame_width
                h1 = (y2 - y1) / frame_height
                if x_center < 0 or y_center < 0 or w1 < 0 or h1 < 0:
                    # print(line)
                    # print(f"invalid values: {x_center}, {y_center}, {w1}, {h1} in {val_det_label_path}")
                    continue
                if x_center > 1 or y_center > 1 or w1 > 1 or h1 > 1:
                    # print(line)
                    # print(f"invalid values: {x_center}, {y_center}, {w1}, {h1} in {val_det_label_path}")
                    continue

                line_after = f"{class_id} {x_center:.6f} {y_center:.6f} {w1:.6f} {h1:.6f}\n"
                frame_id_after = str(frame_id + frame_counts).zfill(6)
                file_name= f"{dir}_{frame_id_after}.txt"
                # print(line_after)
                # print(f"{os.path.join(val_label_output_dir, file_name)}")
                with open(os.path.join(val_label_output_dir, file_name), "a") as f:
                    f.write(line_after)
        
                
    
    