import os
import shutil

def copy_frames_by_25_interval(source_folder, target_folder):
    """
    从源文件夹筛选frame_*_ID_*.jpg格式图片，按帧号隔25帧复制到目标文件夹
    :param source_folder: 源图片文件夹路径
    :param target_folder: 目标保存文件夹路径
    """
    # 创建目标文件夹，已存在则不报错
    os.makedirs(target_folder, exist_ok=True)
    
    # 筛选源文件夹中符合格式的jpg图片，排除其他无关文件
    image_files = [f for f in os.listdir(source_folder) 
                   if f.endswith('.jpg') and 'frame_' in f and '_ID_' in f]
    
    # 用于存储(帧号, 文件名)的列表，后续按帧号排序
    frame_file_list = []
    for filename in image_files:
        try:
            # 核心：split分割字符串提取帧号，按格式frame_1_ID_1.jpg分割
            frame_part = filename.split('_')[1]  # 分割后取第2个元素（索引1）为帧号部分
            frame_num = int(frame_part)          # 转为整数帧号，用于数值判断和排序
            frame_file_list.append((frame_num, filename))
        except (IndexError, ValueError):
            # 跳过格式异常的文件（如分割失败、帧号非数字），不中断程序
            print(f"跳过格式异常文件：{filename}")
            continue
    
    # 按帧号从小到大排序，确保按视频帧顺序处理（关键，避免乱序）
    frame_file_list.sort()
    if not frame_file_list:
        print("未找到符合格式的图片文件！")
        return
    
    # 遍历排序后的帧文件，隔25帧复制（帧号 % 25 == 0 即每25帧取1张）
    for frame_num, filename in frame_file_list:
        if frame_num % 25 == 0:
            # 构建源文件和目标文件的完整路径
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            # 复制文件（保留文件元信息，比shutil.copy更推荐）
            shutil.copy2(source_path, target_path)
            print(f"已复制：{filename}（帧号：{frame_num}）")
    
    print(f"\n筛选完成！共处理{len(frame_file_list)}张有效图片，已复制至{target_folder}")

if __name__ == "__main__":
    # -------------------------- 需手动修改的路径 --------------------------
    SOURCE_FOLDER = "runs/track_reid/camera_dajixiang_006/clops"  # 源图片文件夹（frame_*_ID_*.jpg所在路径）
    TARGET_FOLDER = "data/datasets/bpbreid_dajixiang/camera_006"  # 目标保存文件夹（自动创建）
    # ----------------------------------------------------------------------
    copy_frames_by_25_interval(SOURCE_FOLDER, TARGET_FOLDER)