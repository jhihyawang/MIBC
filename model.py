import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from IpsilateralFusion import IpsiCrossViewFusion
from BilateralFusion import BilateralFusion

class SiameseResNetRuleModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True,
                 num_classes=3, architecture='baseline',
                 concate_method='concat', decision_rule='max'): 
        super().__init__()

        self.backbone_name = backbone_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.concate_method = concate_method
        self.decision_rule = decision_rule

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
            self.backbone = models.resnet101(
                weights=models.ResNet101_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1280
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1536
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(
                weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()
            
        elif backbone_name == 'convnext_small':
            self.backbone = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()
            
        elif backbone_name == 'convnext_base':
            self.backbone = models.convnext_base(
                weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1024
            self.backbone.classifier[2] = nn.Identity()
            
        else:
            raise ValueError(f"不支援的骨幹網路: {backbone_name}")

        # global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- cross-view module ----------------
        if architecture == "cross_view" or architecture == "ipsi": 
            self.cross_ipsi = IpsiCrossViewFusion(dim=self.feature_dim, heads=4)
        else:
            self.cross_ipsi = None

        if architecture == "cross_view" or architecture == "bi":
            self.bilateral_fusion = BilateralFusion(dim=self.feature_dim)
        else:
            self.bilateral_fusion = None

        # ---------------- classifiers ----------------
        # 兩視角（CC + MLO）concat → breast-level classifier
        # 這裡的 concat 是針對「單側乳房」的 2 視角 (L-CC, L-MLO) 或 (R-CC, R-MLO)

        if self.concate_method == 'concat':
            self.breast_classifier = nn.Linear(self.feature_dim * 2, self.num_classes)

        elif self.concate_method == 'concat_linear':
            self.breast_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 2, self.num_classes)
            )

        elif self.concate_method == 'concat_mlp':
            hidden_dim = self.feature_dim  # 例如 2048

            self.breast_classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 2),      # 穩定訓練
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, self.num_classes)
            )

    # ---------------- feature extractor ----------------
    def forward_one_view(self, x):
        """處理單張影像提取特徵"""
        # x shape: (Batch_Size * 4, 3, H, W)
        
        if 'resnet' in self.backbone_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
        elif 'efficientnet' in self.backbone_name:
            x = self.backbone.features(x)
            
        elif 'convnext' in self.backbone_name:
            x = self.backbone.features(x)
            
        return x  # (Batch, Feature_Dim, H', W')

    # ---------------- forward ----------------
    def forward(self, x):
        # x: (B,4,3,H,W)
        B, V, C, H, W = x.shape

        x = x.view(B * V, C, H, W)
        fmap = self.forward_one_view(x)                    # (B*4,C,Hf,Wf)
        _, C2, Hf, Wf = fmap.shape
        fmap = fmap.view(B, 4, C2, Hf, Wf)                 # (B,4,C,Hf,Wf)

        # ---- cross-attention（可選）----
        if self.architecture == "cross_view":
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

        elif self.architecture == "ipsi":
            feats = {
                "L-CC":  fmap[:, 0],
                "R-CC":  fmap[:, 1],
                "L-MLO": fmap[:, 2],
                "R-MLO": fmap[:, 3],
            }
            feats = self.cross_ipsi(feats)

            fmap = torch.stack(
                [feats["L-CC"], feats["R-CC"], feats["L-MLO"], feats["R-MLO"]], dim=1
            )

        elif self.architecture == "bi":
            feats = {
                "L-CC":  fmap[:, 0],
                "R-CC":  fmap[:, 1],
                "L-MLO": fmap[:, 2],
                "R-MLO": fmap[:, 3],
            }
            h_cm_left, h_cm_right = self.bilateral_fusion(feats["L-CC"], feats["R-CC"])
            h_mc_left, h_mc_right = self.bilateral_fusion(feats["L-MLO"], feats["R-MLO"])

            fmap = torch.stack(
                [h_cm_left, h_cm_right, h_mc_left, h_mc_right], dim=1
            )

        # global pooling
        # fmap: (B, 4, C, Hf, Wf)
        B, V, C2, Hf, Wf = fmap.shape
        fmap_flat = fmap.view(B * V, C2, Hf, Wf)        # (B*4, C, Hf, Wf)

        pooled = self.global_pool(fmap_flat)           # (B*4, C, 1, 1)
        pooled = pooled.view(B, V, self.feature_dim)   # (B, 4, C)

        # 四視角 左右分別 concat → exam-level feature
        l_feat = torch.cat([pooled[:, 0], pooled[:, 2]], dim=1)  # (B, 2*C_f)
        r_feat = torch.cat([pooled[:, 1], pooled[:, 3]], dim=1)  # (B, 2*C_f)

        # breast-level logits
        L_logits = self.breast_classifier(l_feat)  # (B, num_classes)
        R_logits = self.breast_classifier(r_feat)  # (B, num_classes)

        # ---------- 轉成機率 ----------
        L_prob = F.softmax(L_logits, dim=1)  # (B, num_classes)
        R_prob = F.softmax(R_logits, dim=1)  # (B, num_classes)

        if self.decision_rule == 'max':
            # 對每個 class 取左右乳中較大的機率，再 renormalize
            m = torch.max(L_prob, R_prob)                   # (B, num_classes)
            exam_prob = m / (m.sum(dim=1, keepdim=True) + 1e-8)

        elif self.decision_rule == 'rule':
            # ----- 機率版臨床規則 -----
            # L_prob, R_prob: (B, 3) 對應 class 0,1,2

            pL0 = L_prob[:, 0]
            pL1 = L_prob[:, 1]
            pL2 = L_prob[:, 2]

            pR0 = R_prob[:, 0]
            pR1 = R_prob[:, 1]
            pR2 = R_prob[:, 2]

            # 1) exam = 2：至少一側為 2
            exam_p2 = 1.0 - (1.0 - pL2) * (1.0 - pR2)

            # 2) exam = 0：有一側為 0 且另一側非 2
            #    (L=0,R=0), (L=0,R=1), (L=1,R=0)
            exam_p0 = pL0 * pR0 + pL0 * pR1 + pL1 * pR0

            # 3) exam = 1：剩下的機率
            exam_p1 = 1.0 - exam_p0 - exam_p2

            # 數值穩定處理（避免極小負數）
            exam_prob = torch.stack([exam_p0, exam_p1, exam_p2], dim=1)  # (B, 3)
            exam_prob = torch.clamp(exam_prob, min=1e-8)
            exam_prob = exam_prob / exam_prob.sum(dim=1, keepdim=True)

        else:
            raise ValueError(f"不支援的決策規則: {self.decision_rule}")

        exam_log_prob = torch.log(exam_prob + 1e-8)

        return exam_log_prob, L_prob, R_prob, L_logits, R_logits
