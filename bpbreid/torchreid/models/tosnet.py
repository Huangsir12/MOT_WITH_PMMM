"""T-OSNet: Transformer-enhanced Omni-Scale Network for Person Re-Identification

Architecture:
- Stage 1-3: Original OSBlock (multi-scale CNN feature extraction)
- Stage 4: OS-Transformer Block (hybrid CNN + lightweight self-attention)
- Head: Global Average Pooling + BN + Linear (same as OSNet)
"""

from __future__ import division, absolute_import
import errno
import math
import os
import warnings
from collections import OrderedDict

import gdown
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'tosnet_x1_0', 'tosnet_x0_75', 'tosnet_x0_5', 'tosnet_x0_25'
]

pretrained_urls = {
    'tosnet_x1_0': 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'tosnet_x0_75': 'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'tosnet_x0_5': 'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'tosnet_x0_25': 'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs'
}


##########
# Basic layers (reuse from OSNet)
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# OSNet building blocks
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# Lightweight Multi-Head Self-Attention
##########
class LightweightMHSA(nn.Module):
    """Token-space multi-head self-attention with pre-norm."""

    def __init__(self, channels, num_heads=2, pooled_size=None):
        super(LightweightMHSA, self).__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.channels = channels
        self.num_heads = num_heads
        self.pooled_size = pooled_size
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        original_size = (H, W)

        if self.pooled_size is not None:
            pooled_h = min(self.pooled_size[0], H)
            pooled_w = min(self.pooled_size[1], W)
            x = F.adaptive_avg_pool2d(x, (pooled_h, pooled_w))
            H, W = pooled_h, pooled_w

        tokens = x.flatten(2).transpose(1, 2)
        norm_tokens = self.norm(tokens)
        attn_tokens, _ = self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + attn_tokens
        out = tokens.transpose(1, 2).reshape(B, C, H, W)

        if self.pooled_size is not None and (H, W) != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

        return out


class LightTokenFFN(nn.Module):
    """Lightweight token FFN with pre-norm."""

    def __init__(self, dim, ratio=2.0):
        super(LightTokenFFN, self).__init__()
        hidden_dim = max(dim, int(dim * ratio))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        tokens = x.flatten(2).transpose(1, 2)
        residual = tokens
        tokens = self.norm(tokens)
        tokens = self.fc1(tokens)
        tokens = self.act(tokens)
        tokens = self.fc2(tokens)
        tokens = residual + tokens
        return tokens.transpose(1, 2).reshape_as(x)


##########
# OS-Transformer Block
##########
class OSTransformerBlock(nn.Module):
    """OS-Transformer Block v2: true parallel CNN and attention branches."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=2,
        channel_reduction=None,
        IN=False,
        pooled_size=None,
        ffn_ratio=2.0,
        **kwargs
    ):
        super(OSTransformerBlock, self).__init__()

        self.osblock = OSBlock(in_channels, out_channels, IN=IN, **kwargs)
        attn_channels = out_channels
        if channel_reduction is not None and channel_reduction > 1:
            attn_channels = max(32, out_channels // channel_reduction)

        self.attn_proj = Conv1x1Linear(in_channels, attn_channels)
        self.mhsa = LightweightMHSA(attn_channels, num_heads=num_heads, pooled_size=pooled_size)
        self.ffn = LightTokenFFN(attn_channels, ratio=ffn_ratio)
        self.attn_expand = Conv1x1Linear(attn_channels, out_channels)

        self.fusion = Conv1x1Linear(out_channels * 2, out_channels)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.attn_scale = nn.Parameter(torch.zeros(1))
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        cnn_feat = self.osblock(x)

        attn_feat = self.attn_proj(x)
        attn_feat = self.mhsa(attn_feat)
        attn_feat = self.ffn(attn_feat)
        attn_feat = self.attn_expand(attn_feat)

        fused = torch.cat([cnn_feat, self.attn_scale * attn_feat], dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_bn(fused)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = fused + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# T-OSNet Architecture
##########
class TOSNet(nn.Module):
    """Transformer-enhanced Omni-Scale Network (T-OSNet).

    Architecture:
        - Stage 1-3: Original OSBlock (shallow layers for local features)
        - Stage 4: OS-Transformer Block (deep layer for global semantic features)
        - Head: Global Average Pooling + BN + Linear (same as OSNet)

    Args:
        num_classes: Number of identity classes
        blocks: List of block types for each stage
        layers: List of number of blocks in each stage
        channels: List of channel numbers for each stage
        feature_dim: Dimension of final feature vector
        loss: Loss type ('softmax', 'triplet', 'part_based')
        num_heads: Number of attention heads in OS-Transformer blocks
        channel_reduction: Channel reduction ratio for attention (None or int)
        IN: Whether to use Instance Normalization
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        loss='softmax',
        num_heads=2,
        channel_reduction=None,
        IN=False,
        **kwargs
    ):
        super(TOSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss

        # Convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stage 1-3: Original OSBlock (shallow layers)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN,
            num_heads=num_heads,
            channel_reduction=channel_reduction
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
            num_heads=num_heads,
            channel_reduction=channel_reduction
        )

        # Stage 4: OS-Transformer Block (deep layer)
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
            num_heads=max(1, num_heads),
            channel_reduction=max(4, channel_reduction or 4),
            pooled_size=(8, 4),
            ffn_ratio=1.5
        )

        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.fc = self._construct_fc_layer(
            feature_dim, channels[3], dropout_p=None
        )

        # Identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False,
        num_heads=2,
        channel_reduction=None,
        pooled_size=None,
        ffn_ratio=2.0
    ):
        layers = []

        # First block
        if block == OSTransformerBlock:
            layers.append(
                block(
                    in_channels,
                    out_channels,
                    num_heads=num_heads,
                    channel_reduction=channel_reduction,
                    IN=IN,
                    pooled_size=pooled_size,
                    ffn_ratio=ffn_ratio
                )
            )
        else:
            layers.append(block(in_channels, out_channels, IN=IN))

        # Remaining blocks
        for i in range(1, layer):
            if block == OSTransformerBlock:
                layers.append(
                    block(
                        out_channels,
                        out_channels,
                        num_heads=num_heads,
                        channel_reduction=channel_reduction,
                        IN=IN,
                        pooled_size=pooled_size,
                        ffn_ratio=ffn_ratio
                    )
                )
            else:
                layers.append(block(out_channels, out_channels, IN=IN))

        # Spatial downsampling
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if hasattr(self.conv4, 'modules'):
            for module in self.conv4.modules():
                if isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    if module.in_proj_bias is not None:
                        nn.init.constant_(module.in_proj_bias, 0)
                    nn.init.constant_(module.out_proj.weight, 0)
                    if module.out_proj.bias is not None:
                        nn.init.constant_(module.out_proj.bias, 0)
                elif isinstance(module, LightTokenFFN):
                    nn.init.xavier_uniform_(module.fc1.weight)
                    nn.init.constant_(module.fc1.bias, 0)
                    nn.init.constant_(module.fc2.weight, 0)
                    nn.init.constant_(module.fc2.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        # For part-based models like BPBreID, always return feature maps
        if return_featuremaps or self.loss == 'part_based':
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, key=''):
    """Initializes T-OSNet with matching OSNet pretrained weights."""

    def _get_torch_home():
        env_torch_home = 'TORCH_HOME'
        env_xdg_cache_home = 'XDG_CACHE_HOME'
        default_cache_dir = '~/.cache'
        return os.path.expanduser(
            os.getenv(
                env_torch_home,
                os.path.join(os.getenv(env_xdg_cache_home, default_cache_dir), 'torch')
            )
        )

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    filename = key.replace('tosnet', 'osnet') + '_imagenet.pt'
    cached_file = os.path.join(model_dir, filename)
    fallback_file = os.path.join(
        '/root/autodl-tmp/MOT_WITH_PMMM/bpbreid/weights/bpbreid_pretrained_model',
        filename
    )
    if os.path.exists(fallback_file):
        cached_file = fallback_file

    osnet_key = key.replace('tosnet', 'osnet')
    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file, map_location='cpu')
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    skipped_prefixes = ('conv4.', 'classifier.')

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith(skipped_prefixes):
            discarded_layers.append(k)
            continue
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded into T-OSNet '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print(
            'Successfully loaded shared pretrained weights from "{}" for {}'
            .format(cached_file, key)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded due to unmatched keys, '
                'new transformer layers, or classifier size: {}'.format(discarded_layers)
            )


##########
# Instantiation
##########
def tosnet_x1_0(num_classes=1000, pretrained=False, loss='softmax', num_heads=2, channel_reduction=None, **kwargs):
    """T-OSNet x1.0 (standard width)

    Args:
        num_classes: Number of identity classes
        pretrained: Whether to load pretrained weights (not supported yet)
        loss: Loss type
        num_heads: Number of attention heads (default: 2)
        channel_reduction: Channel reduction for attention (default: None, no reduction)
    """
    model = TOSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSTransformerBlock],  # Stage 4 uses OS-Transformer
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        num_heads=num_heads,
        channel_reduction=channel_reduction,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='tosnet_x1_0')
    return model


def tosnet_x0_75(num_classes=1000, pretrained=False, loss='softmax', num_heads=2, channel_reduction=None, **kwargs):
    """T-OSNet x0.75 (medium width)"""
    model = TOSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSTransformerBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        num_heads=num_heads,
        channel_reduction=channel_reduction,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='tosnet_x0_75')
    return model


def tosnet_x0_5(num_classes=1000, pretrained=False, loss='softmax', num_heads=2, channel_reduction=None, **kwargs):
    """T-OSNet x0.5 (small width)"""
    model = TOSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSTransformerBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        num_heads=num_heads,
        channel_reduction=channel_reduction,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='tosnet_x0_5')
    return model


def tosnet_x0_25(num_classes=1000, pretrained=False, loss='softmax', num_heads=2, channel_reduction=None, **kwargs):
    """T-OSNet x0.25 (very small width)"""
    model = TOSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSTransformerBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        num_heads=num_heads,
        channel_reduction=channel_reduction,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='tosnet_x0_25')
    return model
