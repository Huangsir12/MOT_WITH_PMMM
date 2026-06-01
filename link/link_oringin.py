import os
import yaml
import sys
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")
from bpbreid.torchreid.scripts.reID_app import inference_reid_init

from tracking.pmmm_scripts.scripts import *


class Link:
    def __init__(self, input_dir, reid_config_file=None):
        
        self.dataset_dir = input_dir

        if reid_config_file:
            with open(reid_config_file, 'r') as file:
                doc = yaml.safe_load(file)

            doc['inference']['dataset_folder'] = self.dataset_dir

            with open(reid_config_file, 'w') as file:
                yaml.safe_dump(doc, file, default_flow_style=None)

            self.reid_inference = inference_reid_init(reid_config_file)
        else:
            self.reid_inference = None


    def get_counts(self, folder_path):
        """
        统计指定文件夹中，每个 ID 对应的图片数量
        文件名格式: frame_数字_ID_数字.jpg
        ID 定义: 文件名中 "ID_" 后面的数字
        
        Args:
            folder_path (str): 图片所在的文件夹路径
            
        Returns:
            dict: 键为 ID (字符串), 值为该 ID 对应的图片数量
        """
        pids_count = {}

        # print(len(os.listdir(folder_path)))
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            # 仅处理 .jpg 文件
            if file_name.lower().endswith('.jpg'):
                try:
                    # 分割文件名，提取 ID 部分
                    # 示例: 'frame_25_ID_1.jpg' -> ['frame', '25', 'ID', '1.jpg']
                    parts = file_name.split('_')
                    track_id = int(parts[-1].split('.')[0]) # 处理后缀 .jpg
                    # track_id = int(parts[0])
                    
                    # 更新计数
                    if track_id in pids_count:
                        pids_count[track_id] += 1
                    else:
                        pids_count[track_id] = 1
                except (ValueError, IndexError):
                    # 跳过格式不符合的文件
                    print(f"跳过格式异常文件: {file_name}")
                    continue
        
        # print(pids_count)
        return pids_count
    
    def clear_datasets_func(self, pids_counts, min_count=5, type="query"):
        """
        删除文件夹中，对应ID图片数量少于min_count的所有图片
        :param folder_path: 图片所在文件夹路径
        :param min_count: 最小保留数量，默认5张，少于该数量则删除对应ID所有图片
        :return: None，直接在原文件夹执行删除操作
        """

        if type == "query":
            folder_path = os.path.join(self.dataset_dir, "query")

        if type == "gallery":
            folder_path = os.path.join(self.dataset_dir, "gallery")
        
        delete_ids = [str(tid) for tid, count in pids_counts.items() if count < min_count]
        if not delete_ids:
            print(f"所有ID的图片数量均≥{min_count}张，无需删除")
            return
        
        print(f"检测到需删除的ID（数量<{min_count}张）: {', '.join(delete_ids)}")
        deleted_files = []  # 记录已删除的文件，用于最后日志输出
        error_files = []    # 记录删除失败的文件

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.jpg'):
                try:
                    # 提取当前文件的ID，匹配需删除的ID列表
                    parts = file_name.split('_')
                    track_id = parts[-1].split('.')[0]
                    
                    if track_id in delete_ids:
                        file_path = os.path.join(folder_path, file_name)
                        os.remove(file_path)  # 执行删除操作
                        deleted_files.append(file_name)
                except (ValueError, IndexError):
                    # 跳过格式异常的文件，不中断流程
                    continue
                except Exception as e:
                    # 捕获删除失败的异常（如文件被占用、权限不足）
                    error_files.append(f"{file_name} - 失败原因: {str(e)[:50]}")

        # 第四步：输出删除结果日志
        print("="*50)
        print(f"删除操作完成！")
        print(f"成功删除文件数: {len(deleted_files)}")
        if deleted_files:
            print(f"已删除文件示例: {', '.join(deleted_files[:5])}{'...' if len(deleted_files)>5 else ''}")
        if error_files:
            print(f"删除失败文件数: {len(error_files)}，失败详情: {error_files}")
        print("="*50)



    def run_link(self, pids_counts, query_folder, gallery_folder):
        matched_id = self.reid_inference.run_tracking(pids_counts, query_folder, gallery_folder)
        return matched_id


if __name__ == "__main__":

    config_file = "bpbreid/configs/bpbreid/bpbreid_inference.yaml"
    dataset_folder = "data/datasets/bpbreid_dajixiang/crops"
    clear_dataset = False

    query_folder = os.path.join(dataset_folder, "camera_002")
    gallery_folder = os.path.join(dataset_folder, "camera_003")

    link = Link(input_dir=dataset_folder,
                reid_config_file=config_file)
    query_pids_counts = link.get_counts(query_folder)
    gallery_pids_counts = link.get_counts(gallery_folder)

    if clear_dataset:
        link.clear_datasets_func(query_pids_counts, min_count=3, type="query")
        link.clear_datasets_func(gallery_pids_counts, min_count=5, type="gallery")
        query_pids_counts = link.get_counts(type="query")

    matches = link.run_link(pids_counts=query_pids_counts, 
                            query_folder=query_folder, 
                            gallery_folder=gallery_folder)
    print(matches)

    # {1: 9, 4: 5, 5: 6, 27: 941, 32: 19, 34: 29, 39: 42, 44: 222, 46: 374, 54: 66, 61: 135, 80: 68, 81: 127, 147: 52, 148: 155, 153: 158, 177: 167, 178: 173, 180: 184, 182: 176, 185: 175, 187: 178, 196: 195, 198: 5, 201: 200, 204: 205, 225: 237, 226: 232, 227: 243, 229: 236, 242: 243, 258: 363, 259: 274, 270: 277, 271: 279, 273: 716, 294: 337, 306: 375, 320: 371, 321: 721, 323: 393, 331: 393, 337: 398, 339: 397, 346: 401, 347: 417, 348: 393, 351: 573, 357: 442, 372: 386, 404: 757, 405: 492, 418: 884, 420: 498, 424: 513, 440: 478, 441: 485, 453: 523, 455: 520, 458: 516, 469: 537, 485: 602, 493: 89, 496: 603, 502: 573, 506: 631, 511: 757, 525: 274, 529: 625, 534: 513, 536: 514, 540: 618, 541: 616, 545: 638, 561: 631, 613: 660, 616: 757, 620: 831, 650: 742, 658: 308, 659: 721, 677: 721, 681: 765, 710: 807, 714: 810, 720: 198, 723: 822, 738: 833, 742: 824, 758: 513, 760: 838, 761: 836, 790: 573, 792: 859, 802: 891, 803: 892, 807: 757, 823: 918, 827: 573, 830: 929, 846: 931, 852: 921, 855: 919, 856: 917, 865: 937, 872: 970, 878: 955, 881: 996, 883: 997, 885: 397, 890: 1006, 895: 697, 902: 1016, 906: 1109, 908: 998, 911: 1000, 913: 1107, 916: 236, 931: 1011, 942: 1033, 946: 884, 956: 1048, 967: 1096, 973: 1128, 975: 1102, 978: 1149, 980: 1111, 986: 397, 987: 1139, 989: 397, 990: 1122, 991: 1142, 995: 217, 1013: 1168, 1015: 1170, 1019: 1148, 1036: 1157, 1040: 1153}