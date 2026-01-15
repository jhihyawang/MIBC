import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from IpsilateralFusion import IpsiCrossViewFusion
from BilateralFusion import BilateralFusion
from nyu_layers import resnet22_nyu, load_nyu_pretrained_weights, kaiming_init_resnet22
from SideInvariantFusion import SideInvariantCrossViewFusion  # 你放哪裡就改哪裡 import

# ✅ 你自己的 backbone
from MNet import ViewResNetV3   # ← 假設你存成 view_resnetv3.py，自己改檔名/路徑


class SiameseResNetRuleModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True,
                 num_classes=3, architecture='baseline',
                 concate_method='2fc', decision_rule='avg',
                 # --- 只給 ViewResNetV3 用的超參（不影響其他 backbone） ---
                 vr_input_channels=3,
                 vr_num_filters=16,
                 vr_growth_factor=2,
                 vr_blocks_per_layer_list=[2, 2, 2, 2, 2],
                 vr_block_strides_list=[1, 2, 2, 2, 2],
                 ):
        super().__init__()

        self.backbone_name = backbone_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.concate_method = concate_method
        self.decision_rule = decision_rule

        # ---------------- Backbone ----------------
        if backbone_name == 'view_resnetv3':
            # ViewResNetV3 最後 out_ch = num_filters * (growth_factor ** 4)  (因為 stage_idx=0..4 共 5 個 stage)
            # 預設 16 * 2^4 = 256
            self.backbone = ViewResNetV3(
                input_channels=vr_input_channels,
                num_filters=vr_num_filters,
                growth_factor=vr_growth_factor,
                blocks_per_layer_list=vr_blocks_per_layer_list,
                block_strides_list=vr_block_strides_list,
                kaiming_init=True,
            )
            self.feature_dim = vr_num_filters * (vr_growth_factor ** 4)

        elif backbone_name == 'resnet22_nyu':
            self.backbone = resnet22_nyu(input_channels=3, kaiming_init=True)
            self.feature_dim = 256
            self._nyu_weights_loaded = False

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

        if architecture == "side_invariant":
            self.sideinv_fusion = SideInvariantCrossViewFusion(dim=self.feature_dim, heads=4, use_gates=True)
        else:
            self.sideinv_fusion = None

        # ---------------- classifiers ----------------
        if self.concate_method == 'concat':
            self.breast_classifier = nn.Linear(self.feature_dim * 2, self.num_classes)

        elif self.concate_method == 'concat_linear':
            self.breast_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 2, self.num_classes)
            )

        elif self.concate_method == 'concat_mlp':
            hidden_dim = self.feature_dim
            self.breast_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, self.num_classes)
            )

        elif self.concate_method == '2fc':
            hidden_dim = self.feature_dim
            self.breast_classifier = nn.Sequential(
                nn.Linear(self.feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_classes)
            )
        else:
            raise ValueError(f"不支援的 concate_method: {self.concate_method}")

        # NYU 權重載入旗標
        self.nyu_weights_loaded = False

    # ---------------- NYU weights ----------------
    def load_nyu_pretrained(self, weights_path):
        if self.backbone_name != 'resnet22_nyu':
            raise ValueError("只有 resnet22_nyu backbone 支援載入 NYU 權重")

        print(f'載入 NYU ResNet22 預訓練權重: {weights_path}')
        print(f'使用 CC 視角權重作為 shared backbone')

        load_nyu_pretrained_weights(self.backbone, weights_path, view='CC')
        self._adapt_first_conv_for_rgb(self.backbone)

        self._nyu_weights_loaded = True
        print('✅ NYU 預訓練權重載入完成')

    def _adapt_first_conv_for_rgb(self, backbone):
        first_conv = backbone.first_conv
        if first_conv.in_channels == 1:
            old_weight = first_conv.weight.data
            new_conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            with torch.no_grad():
                new_weight = old_weight.repeat(1, 3, 1, 1) / 3.0
                new_conv.weight.data = new_weight
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data.clone()

            backbone.first_conv = new_conv
            print("   已將第一層卷積從 1 通道轉換為 3 通道")
        else:
            print("   第一層卷積已經是 3 通道，無需轉換")

    # ---------------- feature extractor ----------------
    def forward_one_view(self, x):
        """處理單張影像提取特徵 (給 torchvision backbone 用)"""
        if self.backbone_name == 'resnet22_nyu':
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

        else:
            raise ValueError(f"forward_one_view 不支援 backbone: {self.backbone_name}")

        return x  # (B, C, Hf, Wf)

    # ---------------- forward ----------------
    def forward(self, x):
        # x: (B,4,3,H,W) - 順序: L-CC, R-CC, L-MLO, R-MLO
        B, V, C, H, W = x.shape
        assert V == 4, f"expect 4 views, got V={V}"

        # ---------------- backbone forward ----------------
        if self.backbone_name == 'view_resnetv3':
            # reorder 成 ViewResNetV3 期待的 (lc, lm, rc, rm)
            lc = x[:, 0]  # L-CC
            rc = x[:, 1]  # R-CC
            lm = x[:, 2]  # L-MLO
            rm = x[:, 3]  # R-MLO

            # ViewResNetV3: (lc,lm,rc,rm)->(lc,lm,rc,rm)
            lc_f, lm_f, rc_f, rm_f = self.backbone(lc, lm, rc, rm)

            # 再 stack 回你後段一致的順序: L-CC, R-CC, L-MLO, R-MLO
            fmap = torch.stack([lc_f, rc_f, lm_f, rm_f], dim=1)  # (B,4,C2,Hf,Wf)

        else:
            # 其他 backbone: shared backbone 跑 B*4
            x_ = x.reshape(B * V, C, H, W)
            fmap = self.forward_one_view(x_)          # (B*4, C2, Hf, Wf)
            _, C2, Hf, Wf = fmap.shape
            fmap = fmap.reshape(B, 4, C2, Hf, Wf)

        # ---------------- optional fusion blocks (你原本的) ----------------
        if self.architecture == "side_invariant":
            feats = {"L-CC": fmap[:, 0], "R-CC": fmap[:, 1], "L-MLO": fmap[:, 2], "R-MLO": fmap[:, 3]}
            feats = self.sideinv_fusion(feats)
            fmap = torch.stack([feats["L-CC"], feats["R-CC"], feats["L-MLO"], feats["R-MLO"]], dim=1)

        elif self.architecture == "cross_view":
            feats = {"L-CC": fmap[:, 0], "R-CC": fmap[:, 1], "L-MLO": fmap[:, 2], "R-MLO": fmap[:, 3]}
            feats = self.cross_ipsi(feats)
            h_cm_left, h_cm_right = self.bilateral_fusion(feats["L-CC"], feats["R-CC"])
            h_mc_left, h_mc_right = self.bilateral_fusion(feats["L-MLO"], feats["R-MLO"])
            fmap = torch.stack([h_cm_left, h_cm_right, h_mc_left, h_mc_right], dim=1)

        elif self.architecture == "ipsi":
            feats = {"L-CC": fmap[:, 0], "R-CC": fmap[:, 1], "L-MLO": fmap[:, 2], "R-MLO": fmap[:, 3]}
            feats = self.cross_ipsi(feats)
            fmap = torch.stack([feats["L-CC"], feats["R-CC"], feats["L-MLO"], feats["R-MLO"]], dim=1)

        elif self.architecture == "bi":
            feats = {"L-CC": fmap[:, 0], "R-CC": fmap[:, 1], "L-MLO": fmap[:, 2], "R-MLO": fmap[:, 3]}
            h_cm_left, h_cm_right = self.bilateral_fusion(feats["L-CC"], feats["R-CC"])
            h_mc_left, h_mc_right = self.bilateral_fusion(feats["L-MLO"], feats["R-MLO"])
            fmap = torch.stack([h_cm_left, h_cm_right, h_mc_left, h_mc_right], dim=1)

        # ---------------- global pooling (你原本的) ----------------
        B, V, C2, Hf, Wf = fmap.shape
        fmap_flat = fmap.reshape(B * V, C2, Hf, Wf)     # (B*4, C2, Hf, Wf)

        pooled = self.global_pool(fmap_flat)            # (B*4, C2, 1, 1)
        pooled = pooled.reshape(B, V, C2)               # ✅ 用 C2，不要強行用 self.feature_dim

        # (可選) sanity check：避免你 feature_dim 設錯造成 silent bug
        if C2 != self.feature_dim:
            # 若你希望嚴格一點，可以直接 raise
            # raise RuntimeError(f"Backbone out channels C2={C2} != self.feature_dim={self.feature_dim}")
            # 這裡我選擇自動對齊，避免你改了 vr_num_filters 卻忘了改 feature_dim
            self.feature_dim = C2

        # 左右乳各自 concat（L-CC + L-MLO；R-CC + R-MLO）
        l_feat = torch.cat([pooled[:, 0], pooled[:, 2]], dim=1)  # (B, 2*C2)
        r_feat = torch.cat([pooled[:, 1], pooled[:, 3]], dim=1)  # (B, 2*C2)

        L_logits = self.breast_classifier(l_feat)
        R_logits = self.breast_classifier(r_feat)

        L_prob = F.softmax(L_logits, dim=1)
        R_prob = F.softmax(R_logits, dim=1)

        if self.decision_rule == 'max':
            m = torch.max(L_prob, R_prob)
            exam_prob = m / (m.sum(dim=1, keepdim=True) + 1e-8)

        elif self.decision_rule == 'avg':
            m = (L_prob + R_prob) / 2.0
            exam_prob = m / (m.sum(dim=1, keepdim=True) + 1e-8)

        elif self.decision_rule == 'rule':
            pL0, pL1, pL2 = L_prob[:, 0], L_prob[:, 1], L_prob[:, 2]
            pR0, pR1, pR2 = R_prob[:, 0], R_prob[:, 1], R_prob[:, 2]

            exam_p2 = 1.0 - (1.0 - pL2) * (1.0 - pR2)
            exam_p0 = pL0 * pR0 + pL0 * pR1 + pL1 * pR0
            exam_p1 = 1.0 - exam_p0 - exam_p2

            exam_prob = torch.stack([exam_p0, exam_p1, exam_p2], dim=1)
            exam_prob = torch.clamp(exam_prob, min=1e-8)
            exam_prob = exam_prob / exam_prob.sum(dim=1, keepdim=True)

        else:
            raise ValueError(f"不支援的決策規則: {self.decision_rule}")

        exam_log_prob = torch.log(exam_prob + 1e-8)
        return exam_log_prob, L_prob, R_prob, L_logits, R_logits
