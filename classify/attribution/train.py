import os
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from model import load_pedestrian_pretrained

# 导入公共模型模块
from model import (
    MultiFrameMultiTaskModel,
    AGE2ID, GENDER2ID, CLOTH2ID, BAG2ID
)

# ======================== 数据集类 ========================
class PedestrianAttrDataset(Dataset):
    def __init__(self, img_dir, anno_path, mode="train", transform=None, max_frames=10):
        """
        Args:
            img_dir: 图片文件夹路径（如 "./data/frames"）
            anno_path: JSON标注文件路径（如 "./data/annotations.json"）
            mode: "train"/"val"/"test"
            transform: 图像预处理
            max_frames: 每个行人ID最多取多少帧（不足补0，超过截断）
        """
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        self.max_frames = max_frames
        
        # 1. 加载并解析JSON标注
        with open(anno_path, "r", encoding="utf-8-sig") as f:
            # content = f.read()
            anno_list = json.load(f)
        
        # 转换为 {pid: {age, gender, cloth_style, bag_type}} 格式
        self.anno_dict = {}
        for item in anno_list:
            pid = item["id"]  # 行人ID
            self.anno_dict[pid] = {
                "age": AGE2ID[item["age"]],
                "gender": GENDER2ID[item["gender"]],
                "cloth_style": CLOTH2ID[item["cloth_style"]],
                "bag_type": BAG2ID[item["bag_type"]]
            }
        
        # print(len(self.anno_dict))
        # 2. 构建ID到图片路径的映射
        self.pid_to_imgs = {}
        # 支持常见图片格式：jpg/jpeg/png
        img_extensions = [".jpg", ".jpeg", ".png"]
        for fname in os.listdir(img_dir):
            if any(fname.lower().endswith(ext) for ext in img_extensions):
                # 解析图片名：示例格式（兼容多种命名）
                # 支持格式1: frame_001_ID_120.jpg
                # 支持格式2: 001_frame_25_ID_120.png
                pid_str = int(fname.split("_")[0])
                img_path = os.path.join(img_dir, fname)
                if pid_str not in self.pid_to_imgs:
                    self.pid_to_imgs[pid_str] = []
                self.pid_to_imgs[pid_str].append(img_path)
        
       
        # 3. 过滤出有标注的ID，并划分训练/验证/测试集
        valid_pids = [pid for pid in self.pid_to_imgs if pid in self.anno_dict]
        print(f"有标注的ID个数：{len(valid_pids)}")
        
        # 划分比例：80%训练，10%验证，10%测试
        train_pids, temp_pids = train_test_split(valid_pids, test_size=0.2, random_state=42)
        val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)
        
        if self.mode == "train":
            self.selected_pids = train_pids
        elif self.mode == "val":
            self.selected_pids = val_pids
        else:
            self.selected_pids = test_pids

    def __len__(self):
        return len(self.selected_pids)

    def __getitem__(self, idx):
        pid = self.selected_pids[idx]
        img_paths = self.pid_to_imgs[pid]
        
        # 按帧号排序（保证时序性）
        def extract_frame_num(path):
            """从文件名提取帧号"""
            fname = os.path.basename(path)
            for part in fname.split("_"):
                if part.startswith("frame"):
                    return int(part.replace("frame", ""))
                if part.isdigit():
                    return int(part)
            return 0
        
        img_paths = sorted(img_paths, key=extract_frame_num)
        
        # 读取并预处理多帧图片
        frames = []
        for i, path in enumerate(img_paths[:self.max_frames]):
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                print(f"警告：读取图片 {path} 失败，跳过该帧: {e}")
        
        # 填充到max_frames（不足用全零张量补）
        if len(frames) == 0:
            # 极端情况：该ID无有效图片，创建空张量
            empty_frame = torch.zeros(3, 224, 224)
            frames = [empty_frame] * self.max_frames
        else:
            while len(frames) < self.max_frames:
                frames.append(torch.zeros_like(frames[0]))
        
        frames = torch.stack(frames)  # shape: [T, C, H, W]
        
        # 获取标注
        anno = self.anno_dict[pid]
        return frames, (anno["age"], anno["gender"], anno["cloth_style"], anno["bag_type"])

# ======================== 训练主函数 ========================
def main():
    # ======================== 配置参数 ========================
    IMG_DIR = "/root/autodl-tmp/MOT_WITH_PMMM/data/datasets/attribution_dajixiang/camera_001"          # 你的图片文件夹路径
    ANNO_PATH = "/root/autodl-tmp/MOT_WITH_PMMM/data/datasets/attribution_dajixiang/camera_001_annotations.json"  
    SAVE_DIR = "./checkpoints"         # 模型保存目录
    BACKBONE = "swin_tiny_patch4_window7_224"              # 可选：efficientnet_b3, swin_t, convnext_tiny
    FUSION_TYPE = "transformer"               # 可选：mean, max, lstm, transformer
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-4
    MAX_FRAMES = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    """
    🌟 首选方案（平衡精度 / 速度）
    BACKBONE = "efficientnet_b3"  # 精度接近ResNet50，速度快50%
    FUSION_TYPE = "mean"          # 简单稳定，适配轻量模型
    🎯 高精度方案（追求最优效果）
    BACKBONE = "swin_tiny_patch4_window7_224"  # 最高精度
    FUSION_TYPE = "transformer"                # 时序融合更优
    🚀 部署优先方案（边缘设备）
    BACKBONE = "mobilenetv3_large_100"  # 速度最快，显存最低
    FUSION_TYPE = "mean"                # 减少计算量
    🎯 行人属性专用方案（推荐）
    BACKBONE = "resnet50"  # 加载PA-100K预训练权重
    FUSION_TYPE = "mean"

    1、新手 / 快速验证：选 ResNet50，兼容性最好、最稳定；
    2、精度优先：选 Swin-Tiny/ConvNeXt-Tiny，搭配行人专用预训练权重；
    3、部署 / 速度优先：选 EfficientNet-B3/MobileNetV3，兼顾精度和速度；
    4、行人属性专用：优先加载 PA-100K/RAP 预训练的 ResNet50，是性价比最高的选择。
    
    """
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ======================== 数据预处理 ========================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ======================== 加载数据集 ========================
    train_dataset = PedestrianAttrDataset(
        img_dir=IMG_DIR,
        anno_path=ANNO_PATH,
        mode="train",
        transform=train_transform,
        max_frames=MAX_FRAMES
    )
    
    val_dataset = PedestrianAttrDataset(
        img_dir=IMG_DIR,
        anno_path=ANNO_PATH,
        mode="val",
        transform=val_transform,
        max_frames=MAX_FRAMES
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ======================== 初始化模型 ========================
    model = MultiFrameMultiTaskModel(
        backbone_name=BACKBONE,
        fusion_type=FUSION_TYPE
    ).to(DEVICE)
    
    # 多任务损失（各任务独立计算交叉熵）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 最佳验证准确率初始化
    best_val_acc = 0.0
    
    # ======================== 训练循环 ========================
    print(f"开始训练！使用设备: {DEVICE}, 骨干网络: {BACKBONE}, 融合方式: {FUSION_TYPE}")
    print(f"训练集数量: {len(train_dataset)}, 验证集数量: {len(val_dataset)}")
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for frames, (age, gender, cloth, bag) in train_pbar:
            frames = frames.to(device=DEVICE, non_blocking=True)
            age = age.to(device=DEVICE, non_blocking=True)
            gender = gender.to(device=DEVICE, non_blocking=True)
            cloth = cloth.to(device=DEVICE, non_blocking=True)
            bag = bag.to(device=DEVICE, non_blocking=True)
            
            # 前向传播
            age_logits, gender_logits, cloth_logits, bag_logits = model(frames)
            
            # 计算损失
            loss_age = criterion(age_logits, age)
            loss_gender = criterion(gender_logits, gender)
            loss_cloth = criterion(cloth_logits, cloth)
            loss_bag = criterion(bag_logits, bag)
            
            # 总损失（可根据任务重要性调整权重）
            total_loss = loss_age + loss_gender + loss_cloth + loss_bag
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * frames.size(0)
            train_pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = {
            "age": 0, "gender": 0, "cloth_style": 0, "bag_type": 0, "total": 0
        }
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        
        with torch.no_grad():
            for frames, (age, gender, cloth, bag) in val_pbar:
                frames = frames.to(device=DEVICE, non_blocking=True)
                age = age.to(device=DEVICE, non_blocking=True)
                gender = gender.to(device=DEVICE, non_blocking=True)
                cloth = cloth.to(device=DEVICE, non_blocking=True)
                bag = bag.to(device=DEVICE, non_blocking=True)
                
                # 前向传播
                age_logits, gender_logits, cloth_logits, bag_logits = model(frames)
                
                # 计算损失
                loss_age = criterion(age_logits, age)
                loss_gender = criterion(gender_logits, gender)
                loss_cloth = criterion(cloth_logits, cloth)
                loss_bag = criterion(bag_logits, bag)
                total_loss = loss_age + loss_gender + loss_cloth + loss_bag
                
                val_loss += total_loss.item() * frames.size(0)
                
                # 计算准确率
                batch_size = frames.size(0)
                correct["age"] += (age_logits.argmax(1) == age).sum().item()
                correct["gender"] += (gender_logits.argmax(1) == gender).sum().item()
                correct["cloth_style"] += (cloth_logits.argmax(1) == cloth).sum().item()
                correct["bag_type"] += (bag_logits.argmax(1) == bag).sum().item()
                correct["total"] += batch_size
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # 计算各任务准确率
        age_acc = correct["age"] / correct["total"]
        gender_acc = correct["gender"] / correct["total"]
        cloth_acc = correct["cloth_style"] / correct["total"]
        bag_acc = correct["bag_type"] / correct["total"]
        avg_acc = (age_acc + gender_acc + cloth_acc + bag_acc) / 4
        
        # 打印日志
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Age Acc: {age_acc:.4f} | Gender Acc: {gender_acc:.4f}")
        print(f"Cloth Style Acc: {cloth_acc:.4f} | Bag Type Acc: {bag_acc:.4f}")
        print(f"Average Acc: {avg_acc:.4f}")
        
        # 保存最佳模型
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_val_acc,
                "backbone_name": BACKBONE,
                "fusion_type": FUSION_TYPE
            }, save_path)
            print(f"保存最佳模型到 {save_path}，当前平均准确率: {best_val_acc:.4f}")
        
        # 学习率调度
        scheduler.step()
    
    # 保存最后一轮模型
    last_save_path = os.path.join(SAVE_DIR, "last_model.pth")
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "backbone_name": BACKBONE,
        "fusion_type": FUSION_TYPE
    }, last_save_path)
    print(f"\n训练完成！最后模型保存到 {last_save_path}")
    print(f"最佳验证平均准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()