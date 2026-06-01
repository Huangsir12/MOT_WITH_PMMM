import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 导入公共模型模块
from model import (
    load_model,
    ID2AGE, ID2GENDER, ID2CLOTH, ID2BAG
)

# ======================== 单ID推理函数 ========================
def predict_single_pid(
    pid,
    img_dir,
    model_path,
    backbone_name="resnet50",
    fusion_type="mean",
    max_frames=10,
    device="cuda"
):
    """
    对单个行人ID进行属性预测
    Args:
        pid: 行人ID（字符串）
        img_dir: 图片文件夹路径
        model_path: 训练好的模型路径
        backbone_name: 骨干网络名称
        fusion_type: 融合方式
        max_frames: 最大帧数
        device: 推理设备
    Returns:
        预测结果字典（包含文本标签和置信度）
    """
    # 加载模型
    model = load_model(
        model_path=model_path,
        backbone_name=backbone_name,
        fusion_type=fusion_type,
        device=device
    )

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载该ID的多帧图片
    img_paths = glob.glob(os.path.join(img_dir, f"*_ID_{pid}.*"))
    if not img_paths:
        raise FileNotFoundError(f"未找到ID为 {pid} 的图片文件")

    # 按帧号排序（保证时序性）
    def extract_frame_num(path):
        fname = os.path.basename(path)
        for part in fname.split("_"):
            if part.startswith("frame"):
                return int(part.replace("frame", ""))
            if part.isdigit():
                return int(part)
        return 0
    img_paths = sorted(img_paths, key=extract_frame_num)

    # 预处理多帧
    frames = []
    for i, path in enumerate(img_paths[:max_frames]):
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            frames.append(img)
        except Exception as e:
            print(f"警告：读取帧 {path} 失败: {e}")

    # 填充到max_frames
    if len(frames) == 0:
        empty_frame = torch.zeros(3, 224, 224)
        frames = [empty_frame] * max_frames
    else:
        while len(frames) < max_frames:
            frames.append(torch.zeros_like(frames[0]))

    frames = torch.stack(frames).unsqueeze(0).to(device)  # [1, T, C, H, W]

    # 推理
    model.eval()
    with torch.no_grad():
        age_logits, gender_logits, cloth_logits, bag_logits = model(frames)

        # 转换为文本标签
        age_pred_id = age_logits.argmax(1).item()
        gender_pred_id = gender_logits.argmax(1).item()
        cloth_pred_id = cloth_logits.argmax(1).item()
        bag_pred_id = bag_logits.argmax(1).item()

        # 计算置信度
        age_conf = torch.softmax(age_logits, dim=1)[0, age_pred_id].item()
        gender_conf = torch.softmax(gender_logits, dim=1)[0, gender_pred_id].item()
        cloth_conf = torch.softmax(cloth_logits, dim=1)[0, cloth_pred_id].item()
        bag_conf = torch.softmax(bag_logits, dim=1)[0, bag_pred_id].item()

        result = {
            "pid": pid,
            "age": ID2AGE[age_pred_id],
            "gender": ID2GENDER[gender_pred_id],
            "cloth_style": ID2CLOTH[cloth_pred_id],
            "bag_type": ID2BAG[bag_pred_id],
            "age_confidence": round(age_conf, 4),
            "gender_confidence": round(gender_conf, 4),
            "cloth_style_confidence": round(cloth_conf, 4),
            "bag_type_confidence": round(bag_conf, 4)
        }
    return result

# ======================== 批量ID推理函数 ========================
def predict_batch_pids(
    pids,
    img_dir,
    model_path,
    backbone_name="resnet50",
    fusion_type="mean",
    max_frames=10,
    device="cuda"
):
    """
    批量预测多个行人ID的属性
    Args:
        pids: 行人ID列表（如 ["120", "121", "122"]）
        img_dir: 图片文件夹路径
        model_path: 训练好的模型路径
        backbone_name: 骨干网络名称
        fusion_type: 融合方式
        max_frames: 最大帧数
        device: 推理设备
    Returns:
        预测结果列表
    """
    results = []
    for pid in tqdm(pids, desc="批量推理中"):
        try:
            res = predict_single_pid(
                pid=pid,
                img_dir=img_dir,
                model_path=model_path,
                backbone_name=backbone_name,
                fusion_type=fusion_type,
                max_frames=max_frames,
                device=device
            )
            results.append(res)
        except Exception as e:
            print(f"警告：ID {pid} 推理失败: {e}")
            results.append({"pid": pid, "error": str(e)})
    return results

# ======================== 主函数（示例调用） ========================
if __name__ == "__main__":
    # 配置参数
    IMG_DIR = "./data/frames"          # 你的图片文件夹路径
    MODEL_PATH = "./checkpoints/best_model.pth"  # 训练好的模型路径
    BACKBONE = "resnet50"              # 需与训练时一致
    FUSION_TYPE = "mean"               # 需与训练时一致
    MAX_FRAMES = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 单ID推理示例
    print("=== 单ID推理示例 ===")
    pid = "120"  # 替换为你的目标ID
    try:
        result = predict_single_pid(
            pid=pid,
            img_dir=IMG_DIR,
            model_path=MODEL_PATH,
            backbone_name=BACKBONE,
            fusion_type=FUSION_TYPE,
            max_frames=MAX_FRAMES,
            device=DEVICE
        )
        print(f"行人ID {pid} 预测结果:")
        print(f"年龄: {result['age']} (置信度: {result['age_confidence']})")
        print(f"性别: {result['gender']} (置信度: {result['gender_confidence']})")
        print(f"服饰风格: {result['cloth_style']} (置信度: {result['cloth_style_confidence']})")
        print(f"提袋类型: {result['bag_type']} (置信度: {result['bag_type_confidence']})")
    except Exception as e:
        print(f"推理失败: {e}")

    # 2. 批量推理示例（可选）
    print("\n=== 批量推理示例 ===")
    batch_pids = ["120", "121", "122"]  # 替换为你的ID列表
    batch_results = predict_batch_pids(
        pids=batch_pids,
        img_dir=IMG_DIR,
        model_path=MODEL_PATH,
        backbone_name=BACKBONE,
        fusion_type=FUSION_TYPE,
        max_frames=MAX_FRAMES,
        device=DEVICE
    )
    for res in batch_results:
        if "error" in res:
            print(f"ID {res['pid']} 推理失败: {res['error']}")
        else:
            print(f"ID {res['pid']}: 年龄={res['age']}, 性别={res['gender']}, 服饰={res['cloth_style']}, 提袋={res['bag_type']}")