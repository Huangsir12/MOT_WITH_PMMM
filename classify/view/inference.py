# inference.py - 高精度推理脚本
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from .train import ViewPointNet, Config  # 导入训练时的模型定义

class ViewPointPredictor:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = 384
        
        # 加载配置
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 初始化模型
        self.model = ViewPointNet(
            model_name="convnext_base.fb_in22k",
            num_orientations=2,
            num_angles=5,
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # 标签映射
        self.orientation_labels = {0: "face", 1: "back"}
        self.angle_labels = {0: "0°", 1: "45°", 2: "90°", 3: "135°", 4: "180°"}
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"模型加载成功！来自 Epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"最佳验证精度: {checkpoint.get('best_combined_acc', 'unknown')}")
    
    @torch.no_grad()
    def predict(self, image_path, return_probs=False):
        """
        单张图片预测
        
        Args:
            image_path: 图片路径或PIL Image
            return_probs: 是否返回概率分布
        """
        # 加载图片
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        orig_image = image.copy()
        
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        outputs = self.model(input_tensor)
        
        # 获取预测结果
        ori_probs = F.softmax(outputs["orientation"], dim=1)
        angle_probs = F.softmax(outputs["angle"], dim=1)
        
        ori_conf, ori_pred = torch.max(ori_probs, dim=1)
        angle_conf, angle_pred = torch.max(angle_probs, dim=1)
        
        # 组合置信度
        combined_conf = (ori_conf.item() + angle_conf.item()) / 2
        
        result = {
            "orientation": {
                "class": self.orientation_labels[ori_pred.item()],
                "confidence": ori_conf.item(),
                "class_id": ori_pred.item()
            },
            "angle": {
                "class": self.angle_labels[angle_pred.item()],
                "confidence": angle_conf.item(),
                "class_id": angle_pred.item()
            },
            "combined_confidence": combined_conf,
            "is_confident": combined_conf > 0.8  # 置信度阈值
        }
        
        if return_probs:
            result["orientation"]["probabilities"] = ori_probs[0].cpu().numpy().tolist()
            result["angle"]["probabilities"] = angle_probs[0].cpu().numpy().tolist()
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=16):
        """批量预测"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_images.append(self.transform(img))
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            outputs = self.model(batch_tensor)
            
            ori_probs = F.softmax(outputs["orientation"], dim=1)
            angle_probs = F.softmax(outputs["angle"], dim=1)
            
            _, ori_preds = torch.max(ori_probs, dim=1)
            _, angle_preds = torch.max(angle_probs, dim=1)
            
            for j, path in enumerate(batch_paths):
                results.append({
                    "path": str(path),
                    "orientation": self.orientation_labels[ori_preds[j].item()],
                    "angle": self.angle_labels[angle_preds[j].item()],
                    "orientation_conf": ori_probs[j].max().item(),
                    "angle_conf": angle_probs[j].max().item()
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """可视化预测结果"""
        # 预测
        result = self.predict(image_path, return_probs=True)
        
        # 加载图片
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 原图 + 预测结果
        axes[0].imshow(img)
        title = f"Orientation: {result['orientation']['class']} ({result['orientation']['confidence']:.2%})\n"
        title += f"Angle: {result['angle']['class']} ({result['angle']['confidence']:.2%})"
        color = "green" if result['is_confident'] else "orange"
        axes[0].set_title(title, color=color, fontsize=12, weight='bold')
        axes[0].axis("off")
        
        # 概率分布
        ori_probs = result['orientation']['probabilities']
        angle_probs = result['angle']['probabilities']
        
        x = np.arange(len(ori_probs))
        axes[1].bar(x - 0.2, ori_probs, 0.4, label='Orientation', alpha=0.8)
        axes[1].bar(x + 0.2, angle_probs, 0.4, label='Angle', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Face/0°', 'Back/45°', '-/90°', '-/135°', '-/180°'])
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Probability Distribution')
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return result

def main():
    parser = argparse.ArgumentParser(description="视角分类推理")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model.pth", help="模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径或文件夹")
    parser.add_argument("--output", type=str, default="./results", help="输出文件夹")
    parser.add_argument("--visualize", action="store_true", help="是否可视化结果")
    parser.add_argument("--batch", action="store_true", help="批量推理")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 初始化预测器
    print("正在加载模型...")
    predictor = ViewPointPredictor(args.model, args.device)
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # 批量推理
        image_paths = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpeg"))
        print(f"找到 {len(image_paths)} 张图片")
        
        results = predictor.predict_batch(image_paths)
        
        # 保存结果
        output_json = output_dir / "predictions.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 统计
        ori_dist = {}
        angle_dist = {}
        for r in results:
            ori_dist[r['orientation']] = ori_dist.get(r['orientation'], 0) + 1
            angle_dist[r['angle']] = angle_dist.get(r['angle'], 0) + 1
        
        print("\n=== 统计结果 ===")
        print("朝向分布:", ori_dist)
        print("角度分布:", angle_dist)
        print(f"详细结果保存至: {output_json}")
        
        # 可视化部分结果
        if args.visualize:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            for i, img_path in enumerate(image_paths[:10]):  # 只可视化前10张
                predictor.visualize_prediction(img_path, vis_dir / f"vis_{i}.png")
    
    else:
        # 单张推理
        result = predictor.predict(input_path, return_probs=True)
        
        print("\n=== 预测结果 ===")
        print(f"朝向: {result['orientation']['class']} (置信度: {result['orientation']['confidence']:.2%})")
        print(f"角度: {result['angle']['class']} (置信度: {result['angle']['confidence']:.2%})")
        print(f"综合置信度: {result['combined_confidence']:.2%}")
        print(f"是否可信: {'是' if result['is_confident'] else '否'}")
        
        if args.visualize:
            predictor.visualize_prediction(input_path, output_dir / "prediction.png")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()