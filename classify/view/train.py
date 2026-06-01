# train.py - 高精度视角分类训练脚本
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import timm  # 需要: pip install timm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# 配置参数
class Config:
    # 数据路径
    data_root = "/root/autodl-tmp/MOT_WITH_PMMM/data/datasets/view_dajixiang"  # 修改为您的数据路径
    
    # 模型参数
    model_name = "convnext_base.fb_in22k"  # ConvNeXt-Small, ImageNet-22k预训练
    img_size = 384  # ConvNeXt推荐尺寸
    num_orientations = 2  # face, back
    num_angles = 5  # 0, 45, 90, 135, 180
    
    # 训练参数
    batch_size = 32
    num_epochs = 100
    lr = 1e-4
    weight_decay = 0.05
    warmup_epochs = 5
    
    # 增强参数
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    label_smoothing = 0.1
    
    # 系统
    num_workers = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./checkpoints"
    seed = 42

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# 数据集类
class ViewPointDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # 角度映射
        self.angle_map = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4}
        self.orientation_map = {"face": 0, "back": 1}
        
        # 收集数据
        self.samples = []
        self._collect_samples()
        
        print(f"[{split}] 加载 {len(self.samples)} 张图片")
        
    def _collect_samples(self):
        # 遍历 face 和 back 文件夹
        for orientation in ["face", "back"]:
            for angle in [0, 45, 90, 135, 180]:
                folder = self.data_root / self.split / f"{orientation}_angle_{angle}"
                if not folder.exists():
                    print(f"警告: 文件夹不存在，跳过 {folder}")
                    continue

                # 获取该文件夹下所有图片
                images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))

                if len(images) == 0:
                    print(f"警告: {folder} 中没有图片")
                    continue

                for img_path in images:
                    self.samples.append({
                        "path": str(img_path),
                        "orientation": self.orientation_map[orientation],
                        "angle": self.angle_map[angle],
                        "orientation_name": orientation,
                        "angle_value": angle
                    })

        if len(self.samples) == 0:
            raise ValueError(f"错误: {self.split} 集中没有找到任何样本！请检查数据集路径和结构。")

        # 不需要再次划分，因为数据集已经有train/val/test文件夹
        # 如果split是train或val，直接使用对应文件夹的数据
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "orientation": torch.tensor(sample["orientation"], dtype=torch.long),
            "angle": torch.tensor(sample["angle"], dtype=torch.long),
            "path": sample["path"]
        }

# 数据增强
def get_transforms(split="train", img_size=384):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # 小角度旋转增强
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),  # 随机擦除
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# Mixup/Cutmix 实现
class MixupCutmix:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, images, orientations, angles):
        if np.random.rand() > self.prob:
            # 不使用mixup/cutmix时，返回None表示没有混合
            return images, orientations, angles, None, None, None

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        # Mixup
        if np.random.rand() < 0.5:
            mixed_images = lam * images + (1 - lam) * images[index]
            return mixed_images, orientations, angles, orientations[index], angles[index], lam
        else:
            # Cutmix
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            return images, orientations, angles, orientations[index], angles[index], lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# 多任务模型
class ViewPointNet(nn.Module):
    def __init__(self, model_name="convnext_base.fb_in22k", num_orientations=2, num_angles=5, pretrained=True):
        super().__init__()
        
        # 使用timm加载ConvNeXt
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            global_pool="avg"
        )
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            features = self.backbone(dummy)
            in_features = features.shape[1]
        
        print(f"Backbone特征维度: {in_features}")
        
        # 共享特征提取层
        self.feature_dropout = nn.Dropout(0.3)
        
        # 朝向分类头 (Face/Back)
        self.orientation_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_orientations)
        )
        
        # 角度分类头 (0/45/90/135/180)
        self.angle_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_angles)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.orientation_head, self.angle_head]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_dropout(features)
        
        orientation_logits = self.orientation_head(features)
        angle_logits = self.angle_head(features)
        
        return {
            "orientation": orientation_logits,
            "angle": angle_logits,
            "features": features
        }

# 损失函数 (带标签平滑)
class ViewPointLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, outputs, targets, targets_mixed=None, lam=None):
        if targets_mixed is None:
            loss_ori = self.criterion(outputs["orientation"], targets["orientation"])
            loss_angle = self.criterion(outputs["angle"], targets["angle"])
        else:
            # Mixup/Cutmix损失
            loss_ori = lam * self.criterion(outputs["orientation"], targets["orientation"]) + \
                      (1 - lam) * self.criterion(outputs["orientation"], targets_mixed["orientation"])
            loss_angle = lam * self.criterion(outputs["angle"], targets["angle"]) + \
                        (1 - lam) * self.criterion(outputs["angle"], targets_mixed["angle"])
        
        return loss_ori + loss_angle, {"orientation": loss_ori.item(), "angle": loss_angle.item()}

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, scheduler, mixup_cutmix, device, epoch):
    model.train()
    total_loss = 0
    correct_ori = 0
    correct_angle = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        orientations = batch["orientation"].to(device)
        angles = batch["angle"].to(device)
        
        # Mixup/Cutmix
        if epoch < Config.num_epochs - 10:  # 最后10轮不使用mixup
            images, ori_a, angle_a, ori_b, angle_b, lam = mixup_cutmix(images, orientations, angles)
            if ori_b is not None and angle_b is not None and lam is not None:
                targets_mixed = {"orientation": ori_b, "angle": angle_b}
            else:
                targets_mixed = None
                lam = None
                ori_a, angle_a = orientations, angles
        else:
            targets_mixed = None
            lam = None
            ori_a, angle_a = orientations, angles

        optimizer.zero_grad()
        outputs = model(images)

        targets = {"orientation": ori_a, "angle": angle_a}
        
        loss, loss_dict = criterion(outputs, targets, targets_mixed, lam)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        total_loss += loss.item()
        
        # 计算准确率 (使用原始标签)
        with torch.no_grad():
            pred_ori = outputs["orientation"].argmax(dim=1)
            pred_angle = outputs["angle"].argmax(dim=1)
            correct_ori += (pred_ori == orientations).sum().item()
            correct_angle += (pred_angle == angles).sum().item()
            total += orientations.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ori_acc": f"{100.*correct_ori/total:.2f}%",
            "angle_acc": f"{100.*correct_angle/total:.2f}%"
        })
    
    return total_loss / len(dataloader), 100. * correct_ori / total, 100. * correct_angle / total

# 验证函数
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_ori = 0
    correct_angle = 0
    correct_combined = 0
    total = 0
    
    all_ori_preds, all_ori_labels = [], []
    all_angle_preds, all_angle_labels = [], []
    
    for batch in tqdm(dataloader, desc="[Validate]"):
        images = batch["image"].to(device)
        orientations = batch["orientation"].to(device)
        angles = batch["angle"].to(device)
        
        outputs = model(images)
        targets = {"orientation": orientations, "angle": angles}
        
        loss, _ = criterion(outputs, targets)
        total_loss += loss.item()
        
        pred_ori = outputs["orientation"].argmax(dim=1)
        pred_angle = outputs["angle"].argmax(dim=1)
        
        correct_ori += (pred_ori == orientations).sum().item()
        correct_angle += (pred_angle == angles).sum().item()
        correct_combined += ((pred_ori == orientations) & (pred_angle == angles)).sum().item()
        total += orientations.size(0)
        
        all_ori_preds.extend(pred_ori.cpu().numpy())
        all_ori_labels.extend(orientations.cpu().numpy())
        all_angle_preds.extend(pred_angle.cpu().numpy())
        all_angle_labels.extend(angles.cpu().numpy())
    
    metrics = {
        "loss": total_loss / len(dataloader),
        "ori_acc": 100. * correct_ori / total,
        "angle_acc": 100. * correct_angle / total,
        "combined_acc": 100. * correct_combined / total,
        "ori_preds": all_ori_preds,
        "ori_labels": all_ori_labels,
        "angle_preds": all_angle_preds,
        "angle_labels": all_angle_labels
    }
    
    return metrics

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 主函数
def main():
    set_seed(Config.seed)
    os.makedirs(Config.save_dir, exist_ok=True)
    
    # 数据加载
    print(f"数据集路径: {Config.data_root}")
    print(f"检查数据集结构...")

    # 检查数据集是否存在
    if not os.path.exists(Config.data_root):
        raise FileNotFoundError(f"数据集路径不存在: {Config.data_root}")

    train_dataset = ViewPointDataset(Config.data_root, "train", get_transforms("train", Config.img_size))
    val_dataset = ViewPointDataset(Config.data_root, "val", get_transforms("val", Config.img_size))

    print(f"\n数据集统计:")
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, 
                             num_workers=Config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False,
                           num_workers=Config.num_workers, pin_memory=True)
    
    # 模型
    model = ViewPointNet(Config.model_name, Config.num_orientations, Config.num_angles).to(Config.device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.2f}M, 可训练: {trainable_params/1e6:.2f}M")
    
    # 损失和优化器
    criterion = ViewPointLoss(Config.label_smoothing)
    
    # 分层学习率
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": Config.lr * 0.1},  # Backbone使用较小学习率
        {"params": head_params, "lr": Config.lr}
    ], weight_decay=Config.weight_decay)
    
    # 学习率调度
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixup/Cutmix
    mixup_cutmix = MixupCutmix(Config.mixup_alpha, Config.cutmix_alpha)
    
    # 训练循环
    best_combined_acc = 0
    history = {"train_loss": [], "val_loss": [], "ori_acc": [], "angle_acc": [], "combined_acc": []}
    
    for epoch in range(1, Config.num_epochs + 1):
        train_loss, train_ori_acc, train_angle_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, mixup_cutmix, Config.device, epoch
        )
        
        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, Config.device)
        
        print(f"\nEpoch {epoch}/{Config.num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Ori: {train_ori_acc:.2f}%, Angle: {train_angle_acc:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Ori: {val_metrics['ori_acc']:.2f}%, "
              f"Angle: {val_metrics['angle_acc']:.2f}%, Combined: {val_metrics['combined_acc']:.2f}%")
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["ori_acc"].append(val_metrics["ori_acc"])
        history["angle_acc"].append(val_metrics["angle_acc"])
        history["combined_acc"].append(val_metrics["combined_acc"])
        
        # 保存最佳模型
        if val_metrics["combined_acc"] > best_combined_acc:
            best_combined_acc = val_metrics["combined_acc"]

            # 只保存可序列化的配置参数
            config_dict = {
                "data_root": Config.data_root,
                "model_name": Config.model_name,
                "img_size": Config.img_size,
                "num_orientations": Config.num_orientations,
                "num_angles": Config.num_angles,
                "batch_size": Config.batch_size,
                "num_epochs": Config.num_epochs,
                "lr": Config.lr,
                "weight_decay": Config.weight_decay
            }

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_combined_acc": best_combined_acc,
                "config": config_dict
            }, os.path.join(Config.save_dir, "best_model.pth"))
            print(f"✓ 保存最佳模型，Combined Acc: {best_combined_acc:.2f}%")
            
            # 绘制混淆矩阵
            plot_confusion_matrix(
                val_metrics["ori_labels"], val_metrics["ori_preds"],
                ["face", "back"], os.path.join(Config.save_dir, "confusion_orientation.png"),
                "Orientation Confusion Matrix"
            )
            plot_confusion_matrix(
                val_metrics["angle_labels"], val_metrics["angle_preds"],
                ["0°", "45°", "90°", "135°", "180°"],
                os.path.join(Config.save_dir, "confusion_angle.png"),
                "Angle Confusion Matrix"
            )
        
        # 定期保存
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "history": history
            }, os.path.join(Config.save_dir, f"checkpoint_epoch{epoch}.pth"))
    
    # 保存训练历史
    with open(os.path.join(Config.save_dir, "history.json"), "w") as f:
        json.dump(history, f)
    
    print(f"\n训练完成！最佳Combined Accuracy: {best_combined_acc:.2f}%")

if __name__ == "__main__":
    main()