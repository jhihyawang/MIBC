import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cross_atten import IpsiCrossViewFusion
from BilateralFusion import BilateralFusion


class SiameseResNetRuleModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True,
                 num_classes=3, architecture='baseline'):
        super().__init__()

        self.backbone_name = backbone_name
        self.architecture = architecture
        self.num_classes = num_classes

        # ---------------- Backbone ----------------
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 512
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 1280
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 1536
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 2048
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()
            
        elif backbone_name == 'convnext_small':
            self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()
            
        elif backbone_name == 'convnext_base':
            self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 1024
            self.backbone.classifier[2] = nn.Identity()
            
        else:
            raise ValueError(f"不支援的骨幹網路: {backbone_name}")


        # global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 🔹 左右乳 三分類 head（不使用 MIL attention）
        self.breast_classifier = nn.Linear(self.feature_dim, self.num_classes)

        # cross-view 模組仍可保留/關閉
        if architecture == "cross_atten":
            self.cross_ipsi = IpsiCrossViewFusion(dim=self.feature_dim, heads=4)
        else:
            self.cross_ipsi = None

        self.bilateral_fusion = BilateralFusion(dim=self.feature_dim)

    # ---------------- feature extractor ----------------
    def forward_one_view(self, x):
        """處理單張影像提取特徵"""
        # x shape: (Batch_Size * 4, 3, H, W)
        
        if 'resnet' in self.backbone_name:
            # ResNet 架構
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            # x = self.global_pool(x)
            # x = torch.flatten(x, 1)
            
        elif 'efficientnet' in self.backbone_name:
            # EfficientNet 架構
            x = self.backbone.features(x)
            # x = self.global_pool(x)
            # x = torch.flatten(x, 1)
            
        elif 'convnext' in self.backbone_name:
            # ConvNeXt 架構
            x = self.backbone.features(x)
            # x = self.global_pool(x)
            # x = torch.flatten(x, 1)
            
        return x # 回傳 feature map，稍後再池化和展平 (Batch, Feature_Dim, H', W')

    # --------------- 視角→左右乳（不含 MIL）---------------
    def _aggregate_left_right_features(self, pooled):
        """
        pooled: (B,4,C) 視角順序: [L-CC, R-CC, L-MLO, R-MLO]
        👉 去除 MIL，改為 simple average
        """
        # 左乳使用 L-CC (0) + L-MLO (2)
        left_feature = (pooled[:, 0] + pooled[:, 2]) / 2.0   # (B,C)

        # 右乳使用 R-CC (1) + R-MLO (3)
        right_feature = (pooled[:, 1] + pooled[:, 3]) / 2.0  # (B,C)

        return left_feature, right_feature

    # --------------- exam-rule 組合 ----------------
    def _compute_exam_probs_from_breast(self, left_logits, right_logits):
        """
        left_logits / right_logits: (B,3)
        label: 0=不確定 1=正常 2=惡性
        """
        pL = F.softmax(left_logits, dim=-1)
        pR = F.softmax(right_logits, dim=-1)

        # E=2: 只要任一側是 2
        p2 = 1.0 - (1.0 - pL[:, 2]) * (1.0 - pR[:, 2])

        # no 2
        no2 = (1.0 - pL[:, 2]) * (1.0 - pR[:, 2])

        # 兩邊都是 1
        both1 = pL[:, 1] * pR[:, 1]

        # E=0: 無2 且 至少一邊是0
        p0 = no2 - both1

        # 其餘為 1
        p1 = 1.0 - p0 - p2

        exam_probs = torch.stack([p0, p1, p2], dim=-1)

        exam_probs = torch.clamp(exam_probs, 1e-7, 1 - 1e-7)
        exam_probs = exam_probs / exam_probs.sum(dim=-1, keepdim=True)

        return exam_probs

    # ---------------- forward ----------------
    def forward(self, x):
        # x: (B,4,3,H,W)
        B, V, C, H, W = x.shape

        x = x.view(B * V, C, H, W)
        fmap = self.forward_one_view(x)                    # (B*4,C,Hf,Wf)
        _, C2, Hf, Wf = fmap.shape
        fmap = fmap.view(B, 4, C2, Hf, Wf)                 # (B,4,C,Hf,Wf)

        # ---- cross-attention（可選）----
        if self.architecture == "cross_atten" and self.cross_ipsi is not None:
            feats = {
                "L-CC":  fmap[:, 0],
                "R-CC":  fmap[:, 1],
                "L-MLO": fmap[:, 2],
                "R-MLO": fmap[:, 3],
            }
            feats = self.cross_ipsi(feats)

            h_cm_left, h_cm_right = self.bilateral_fusion(feats["L-CC"], feats["R-CC"])
            h_mc_left, h_mc_right = self.bilateral_fusion(feats["L-MLO"], feats["R-MLO"])

            fmap = torch.stack(
                [h_cm_left, h_cm_right, h_mc_left, h_mc_right], dim=1
            )

        # ---- global pooling ----
        pooled = self.global_pool(fmap).view(B, 4, self.feature_dim)  # (B,4,C)

        # ---- simple average per breast（NO MIL）----
        left_feat, right_feat = self._aggregate_left_right_features(pooled)

        # ---- 左右乳三分類 ----
        left_logits = self.breast_classifier(left_feat)
        right_logits = self.breast_classifier(right_feat)

        # ---- exam-rule ----
        exam_probs = self._compute_exam_probs_from_breast(left_logits, right_logits)

        return exam_probs, left_logits, right_logits
