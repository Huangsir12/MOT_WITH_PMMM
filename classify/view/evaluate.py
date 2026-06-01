# evaluate.py - 模型评估脚本
import torch
import json
from pathlib import Path
from train import ViewPointDataset, get_transforms, ViewPointNet, Config
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, data_root):
    """全面评估模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = ViewPointNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 加载测试数据
    dataset = ViewPointDataset(data_root, "val", get_transforms("val", 384))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 收集预测
    all_ori_preds, all_ori_labels = [], []
    all_angle_preds, all_angle_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            
            ori_preds = outputs["orientation"].argmax(dim=1).cpu().numpy()
            angle_preds = outputs["angle"].argmax(dim=1).cpu().numpy()
            
            all_ori_preds.extend(ori_preds)
            all_ori_labels.extend(batch["orientation"].numpy())
            all_angle_preds.extend(angle_preds)
            all_angle_labels.extend(batch["angle"].numpy())
    
    # 计算指标
    print("=" * 50)
    print("朝向分类报告 (Orientation)")
    print(classification_report(all_ori_labels, all_ori_preds, 
                               target_names=["face", "back"], digits=4))
    
    print("\n角度分类报告 (Angle)")
    print(classification_report(all_angle_labels, all_angle_preds,
                               target_names=["0°", "45°", "90°", "135°", "180°"], digits=4))
    
    # 组合准确率
    combined_correct = sum((o1 == o2) and (a1 == a2) 
                          for o1, o2, a1, a2 in zip(all_ori_preds, all_ori_labels, 
                                                   all_angle_preds, all_angle_labels))
    print(f"\n组合准确率 (Combined Accuracy): {100.*combined_correct/len(all_ori_labels):.2f}%")
    
    # 混淆矩阵可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 朝向混淆矩阵
    cm_ori = np.zeros((2, 2))
    for t, p in zip(all_ori_labels, all_ori_preds):
        cm_ori[t][p] += 1
    
    sns.heatmap(cm_ori, annot=True, fmt="g", cmap="Blues", ax=axes[0],
                xticklabels=["face", "back"], yticklabels=["face", "back"])
    axes[0].set_title("Orientation Confusion Matrix")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    
    # 角度混淆矩阵
    cm_angle = np.zeros((5, 5))
    for t, p in zip(all_angle_labels, all_angle_preds):
        cm_angle[t][p] += 1
    
    sns.heatmap(cm_angle, annot=True, fmt="g", cmap="Blues", ax=axes[1],
                xticklabels=["0°", "45°", "90°", "135°", "180°"],
                yticklabels=["0°", "45°", "90°", "135°", "180°"])
    axes[1].set_title("Angle Confusion Matrix")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150)
    print("\n评估图表已保存至: evaluation_results.png")

if __name__ == "__main__":
    evaluate_model("./checkpoints/best_model.pth", "./dataset")