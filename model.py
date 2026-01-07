import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from IpsilateralFusion import IpsiCrossViewFusion
from BilateralFusion import BilateralFusion
from nyu_layers import resnet22_nyu, load_nyu_pretrained_weights

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
        if backbone_name == 'resnet22_nyu':
            # NYU breast cancer classifier ResNet22
            # 注意：NYU 原始模型使用 1 通道 (grayscale)，我們需要適配
            # 我們創建 1 通道模型並載入權重，然後轉換第一層以適配 3 通道輸入
            self.backbone = resnet22_nyu(input_channels=1)  # 先創建 1 通道模型載入權重
            self.feature_dim = 256  # NYU ResNet22 輸出維度
            self._nyu_weights_loaded = False  # 標記是否載入了 NYU 權重
            
        elif backbone_name == 'resnet50':
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
            
        # 載入 NYU 預訓練權重（如果使用 resnet22_nyu）
        self.nyu_weights_loaded = False

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
                # nn.LayerNorm(self.feature_dim * 2),      # 拔掉
                # nn.Dropout(p=0.5),                
                nn.Linear(self.feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, self.num_classes)
            )

    def load_nyu_pretrained(self, weights_path):
        """
        載入 NYU breast cancer classifier 的預訓練權重
        
        Args:
            weights_path: NYU 權重檔案路徑 (.p 或 .pth)
                        推薦使用: models/ImageOnly__ModeImage_weights.p
                        
        支援的 NYU 權重類型:
        - ImageOnly__ModeImage_weights.p: 僅使用影像 (推薦)
        - ImageHeatmaps__ModeImage_weights.p: 影像+熱圖 (需額外處理)
        """
        if self.backbone_name != 'resnet22_nyu':
            raise ValueError("只有 resnet22_nyu 架構支援 NYU 預訓練權重")
            
        print(f"載入 NYU ResNet22 預訓練權重: {weights_path}")
        
        # 檢查權重類型
        if "ImageOnly" in weights_path:
            print("使用 Image-only 預訓練權重 (推薦)")
        elif "ImageHeatmaps" in weights_path:
            print("警告: 使用 Image+heatmaps 權重，可能需要額外適配")
        
        # 載入權重到backbone (1 通道模型)
        load_nyu_pretrained_weights(self.backbone, weights_path, view='L-CC')
        
        # 轉換第一個卷積層以支持 3 通道輸入
        self._adapt_first_conv_for_rgb()
        
        self.nyu_weights_loaded = True
        print("✅ NYU 預訓練權重載入完成")

    def _adapt_first_conv_for_rgb(self):
        """
        將 NYU 的 1 通道第一層卷積轉換為 3 通道，支援 RGB 輸入
        使用權重複製策略：將 1 通道權重複製 3 次並平均
        """
        first_conv = self.backbone.first_conv
        if first_conv.in_channels == 1:
            # 獲取原始權重 (16, 1, 7, 7)
            old_weight = first_conv.weight.data
            
            # 創建新的 3 通道卷積層
            new_conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # 將 1 通道權重擴展到 3 通道（複製 3 次並除以 3 保持數值規模）
            with torch.no_grad():
                new_weight = old_weight.repeat(1, 3, 1, 1) / 3.0
                new_conv.weight.data = new_weight
                
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data.clone()
            
            # 替換第一個卷積層
            self.backbone.first_conv = new_conv
            print("   已將第一層卷積從 1 通道轉換為 3 通道")
        else:
            print("   第一層卷積已經是 3 通道，無需轉換")
        
        print("✅ NYU 預訓練權重載入完成")

    # ---------------- feature extractor ----------------
    def forward_one_view(self, x):
        """處理單張影像提取特徵"""
        # x shape: (Batch_Size * 4, 3, H, W)
        
        if self.backbone_name == 'resnet22_nyu':
            # NYU ResNet22 直接前向傳播
            x = self.backbone(x)
            
        elif 'resnet' in self.backbone_name:
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

        x = x.reshape(B * V, C, H, W)
        fmap = self.forward_one_view(x)                    # (B*4,C,Hf,Wf)
        _, C2, Hf, Wf = fmap.shape
        fmap = fmap.reshape(B, 4, C2, Hf, Wf)                 # (B,4,C,Hf,Wf)

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
        fmap_flat = fmap.reshape(B * V, C2, Hf, Wf)        # (B*4, C, Hf, Wf)

        pooled = self.global_pool(fmap_flat)           # (B*4, C, 1, 1)
        pooled = pooled.reshape(B, V, self.feature_dim)   # (B, 4, C)

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
