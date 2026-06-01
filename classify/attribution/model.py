import torch
import torch.nn as nn
import timm
import re

# ======================== 标签映射配置（公共） ========================
# 年龄标签映射（9类）
AGE_LABELS = [
    '0-2', '3-6', '7-15', '16-25', '26-35', 
    '36-45', '46-55', '56-65', '66岁以上'
]
AGE2ID = {label: idx for idx, label in enumerate(AGE_LABELS)}
ID2AGE = {idx: label for idx, label in enumerate(AGE_LABELS)}

# 性别标签映射（2类）
GENDER_LABELS = ['男', '女']
GENDER2ID = {label: idx for idx, label in enumerate(GENDER_LABELS)}
ID2GENDER = {idx: label for idx, label in enumerate(GENDER_LABELS)}

# 服饰风格映射（5类）
CLOTH_LABELS = ["高端", "时尚", "正式", "休闲", "工作服（门店，美团等）"]
CLOTH2ID = {label: idx for idx, label in enumerate(CLOTH_LABELS)}
ID2CLOTH = {idx: label for idx, label in enumerate(CLOTH_LABELS)}

# 提袋类型映射（5类）
BAG_LABELS = ["无", "手提包", "双肩包", "单肩包", "塑料袋"]
BAG2ID = {label: idx for idx, label in enumerate(BAG_LABELS)}
ID2BAG = {idx: label for idx, label in enumerate(BAG_LABELS)}

# ======================== 多帧融合多任务模型 ========================
class MultiFrameMultiTaskModel(nn.Module):
    def __init__(self, backbone_name="resnet50", fusion_type="mean"):
        super().__init__()
        self.fusion_type = fusion_type
        self.backbone_name = backbone_name
        
        # 加载预训练骨干网络（去掉分类头）
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=True,  # 加载ImageNet预训练权重
            num_classes=0     # 不使用默认分类头
        )
        self.feature_dim = self.backbone.num_features  # 骨干网络输出特征维度
        
        # 多帧融合层
        if fusion_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=self.feature_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            fusion_dim = self.feature_dim  # 双向LSTM输出维度 = hidden_size * 2
        elif fusion_type == "transformer":
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.feature_dim,
                    nhead=8,
                    batch_first=True
                ),
                num_layers=2
            )
            fusion_dim = self.feature_dim
        else:  # mean/max
            fusion_dim = self.feature_dim
        
        # 多任务分类头
        self.age_head = nn.Linear(fusion_dim, len(AGE_LABELS))
        self.gender_head = nn.Linear(fusion_dim, len(GENDER_LABELS))
        self.cloth_style_head = nn.Linear(fusion_dim, len(CLOTH_LABELS))
        self.bag_type_head = nn.Linear(fusion_dim, len(BAG_LABELS))

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] - 批次大小×帧数×通道×高度×宽度
        Returns:
            各任务的logits
        """
        B, T, C, H, W = x.shape
        
        # 1. 骨干网络提取单帧特征
        x = x.view(B*T, C, H, W)  # 合并批次和帧维度: [B*T, C, H, W]
        frame_features = self.backbone(x)  # [B*T, D]
        frame_features = frame_features.view(B, T, -1)  # [B, T, D]
        
        # 2. 多帧融合
        if self.fusion_type == "mean":
            fused_features = frame_features.mean(dim=1)  # [B, D]
        elif self.fusion_type == "max":
            fused_features = frame_features.max(dim=1)[0]  # [B, D]
        elif self.fusion_type == "lstm":
            lstm_out, _ = self.lstm(frame_features)  # [B, T, D]
            fused_features = lstm_out.mean(dim=1)    # [B, D]
        elif self.fusion_type == "transformer":
            trans_out = self.transformer_encoder(frame_features)  # [B, T, D]
            fused_features = trans_out.mean(dim=1)                # [B, D]
        else:
            raise ValueError(f"不支持的融合方式: {self.fusion_type}")
        
        # 3. 多任务预测
        age_logits = self.age_head(fused_features)
        gender_logits = self.gender_head(fused_features)
        cloth_style_logits = self.cloth_style_head(fused_features)
        bag_type_logits = self.bag_type_head(fused_features)
        
        return age_logits, gender_logits, cloth_style_logits, bag_type_logits

# 模型加载辅助函数（公共）
def load_model(model_path, backbone_name="resnet50", fusion_type="mean", device="cuda"):
    """加载训练好的模型"""
    model = MultiFrameMultiTaskModel(
        backbone_name=backbone_name,
        fusion_type=fusion_type
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# def load_pedestrian_pretrained(model, pretrained_path="./pretrained/resnet50_pa100k.pth"):
#     checkpoint = torch.load(pretrained_path, map_location="cpu")
#     # 适配权重命名（去掉分类头）
#     state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith("backbone."):
#             new_state_dict[k.replace("backbone.", "")] = v
#         elif not k.startswith("head."):  # 跳过原分类头
#             new_state_dict[k] = v
#     # 加载权重（忽略分类头不匹配）
#     model.backbone.load_state_dict(new_state_dict, strict=False)
#     return model


def load_pedestrian_pretrained(model, pretrained_path, device="cuda"):
    """
    加载行人属性专用预训练权重（兼容PA-100K/RAP/TorchReid格式）
    Args:
        model: 初始化的MultiFrameMultiTaskModel
        pretrained_path: 行人预训练权重路径
        device: 加载设备
    Returns:
        加载好权重的模型
    """
    # 加载权重文件
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 处理不同格式的权重（兼容多种开源仓库命名）
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 权重名适配（去掉前缀/调整命名）
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉backbone.前缀（如backbone.conv1.weight → conv1.weight）
        k = re.sub(r"^backbone\.|^module\.backbone\.", "", k)
        # 去掉encoder.前缀（Transformer类模型）
        k = re.sub(r"^encoder\.", "", k)
        # 跳过分类头权重（保留骨干网络权重）
        if not any(head in k for head in ["fc", "head", "classifier"]):
            if k in model.backbone.state_dict():
                new_state_dict[k] = v
    
    # 加载权重（strict=False忽略不匹配的层）
    model.backbone.load_state_dict(new_state_dict, strict=False)
    print(f"成功加载行人预训练权重: {pretrained_path}")
    print(f"加载的权重层数: {len(new_state_dict)}/{len(model.backbone.state_dict())}")
    
    return model