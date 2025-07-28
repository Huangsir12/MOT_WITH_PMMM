import os
import shutil

picture_data_dir = "data/datasets/MOT17_YOLO/valid/images"
dst_output_dir = "data/datasets/val/gt/MOT17-val"


dict = {}
for file in os.listdir(picture_data_dir):
    if file.endswith('.jpg'):
        subfolder_name = file.split("_")[0]
        number = int(file.split("_")[1].split(".")[0])
        if subfolder_name not in dict:
            dict[subfolder_name] = number
        else:
            if number < dict[subfolder_name]:
                dict[subfolder_name] = number

for file in os.listdir(picture_data_dir):
    if file.endswith('.jpg'):
        # 构建完整的文件路径
        file_path = os.path.join(picture_data_dir, file)
        subfolder_name = file.split("_")[0]
        dst_filename = int(file.split("_")[1].split(".")[0]) - dict[subfolder_name] + 1
        dst_filename = str(dst_filename).zfill(6)

        # 构建目标文件路径
        dst_file_path = os.path.join(dst_output_dir, subfolder_name, "img1")
        if not os.path.exists(dst_file_path):
            os.makedirs(dst_file_path, exist_ok=True)
        dst_file_path = os.path.join(dst_file_path, f"{dst_filename}.jpg")
        print(f"Copied {file_path} to {dst_file_path}")
        # 复制文件
        shutil.copy(file_path, dst_file_path)