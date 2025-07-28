import os 


label_data = "data/datasets/MOT17/train"
train_label_output_dir = "data/datasets/MOT17_YOLO/train/labels"
val_label_output_dir = "data/datasets/MOT17_YOLO/valid/labels"
frame_height = 1080
frame_width = 1920

SPLITS = ["train", "val"]

for root, dirs, files in os.walk(label_data, topdown=False):
    for dir in dirs:
        train_det_label_path = os.path.join(label_data, dir, "det", "det_train_half.txt")
        val_det_label_path = os.path.join(label_data, dir, "det", "det_val_half.txt")
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
                lines = f.readlines()
            for line in lines:
                line = line.strip().split(",")
                frame_id = int(line[0])
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
                frame_id_after = str(frame_id + frame_counts).zfill(6)
                file_name= f"{dir}_{frame_id_after}.txt"
                # print(line_after)
                # print(f"{os.path.join(val_label_output_dir, file_name)}")
                with open(os.path.join(val_label_output_dir, file_name), "a") as f:
                    f.write(line_after)
        
                
    
    