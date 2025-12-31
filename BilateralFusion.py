import torch
import torch.nn as nn

class BilateralFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        """
        Args:
            dim: 輸入特徵維度 (例如 ResNet50 為 2048)
            reduction: 壓縮倍率，建議設為 8 或 4
        """
        super(BilateralFusion, self).__init__()
        
        # 我們希望在低維度空間做融合，減少參數
        hidden_dim = max(dim // reduction, 64) 
        
        # 下路徑: Context Stream (保持不變，或也加上 bottleneck)
        # 這裡為了簡單，我們只對最肥的 Fusion 層做優化
        self.context_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # 聚合層 (Fusion Stream) - 瓶頸設計
        # 1. 降維 (2*dim -> hidden_dim)
        # 2. 空間融合 (3x3 Conv)
        # 3. 升維 (hidden_dim -> 2*dim)
        self.fusion_conv = nn.Sequential(
            # Step 1: 降維 (1x1)
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Step 2: 空間融合 (3x3) - 現在通數少，參數量很低
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Step 3: 升維還原 (1x1)
            nn.Conv2d(hidden_dim, dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * 2),
            # 這裡最後不一定要 ReLU，留給殘差相加後再做也可以
        )
        
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, left_feat, right_feat):
        # 1. 差異特徵 (Difference) - 記得加 Abs
        diff_feat = torch.abs(left_feat - right_feat) 

        # 2. 共性特徵 (Context)
        cat_feat = torch.cat([left_feat, right_feat], dim=1) 
        context_feat = self.context_conv(cat_feat)

        # 3. 聚合
        combined = torch.cat([diff_feat, context_feat], dim=1)
        
        # 經過瓶頸層融合 (Bottleneck Fusion)
        H = self.fusion_conv(combined) 

        # 4. 輸出分裂
        h_left, h_right = torch.chunk(H, 2, dim=1)
        
        # Residual Connection
        out_left = self.final_relu(h_left + left_feat)
        out_right = self.final_relu(h_right + right_feat)
        
        return out_left, out_right