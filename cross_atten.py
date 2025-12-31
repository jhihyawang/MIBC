import math
from torch import nn
import torch
class PositionalEncoding2D(nn.Module):
    """
    2D sine-cosine positional encoding
    產生 (1, C, H, W) 的位置編碼，加在 feature map 上
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("PositionalEncoding2D 要求 dim 能被 4 整除")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        回傳同 shape 的位置編碼 tensor
        """
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        pe = torch.zeros(1, c, h, w, device=device, dtype=dtype)

        c_half = c // 2
        n_y = c_half // 2      # y 方向的頻率數量
        n_x = (c - c_half) // 2  # x 方向的頻率數量（其實也是 c_half//2）

        # --- y 方向 ---
        y = torch.arange(h, device=device, dtype=dtype)         # (H,)
        yy = y[:, None].repeat(1, w).unsqueeze(0)               # (1, H, W)
        div_y = torch.exp(
            torch.arange(0, n_y, device=device, dtype=dtype)
            * -(math.log(10000.0) / max(n_y, 1))
        ).view(n_y, 1, 1)                                       # (n_y,1,1)

        pe[0, 0:c_half:2, :, :] = torch.sin(yy / div_y)         # (n_y,H,W)
        pe[0, 1:c_half:2, :, :] = torch.cos(yy / div_y)

        # --- x 方向 ---
        x_pos = torch.arange(w, device=device, dtype=dtype)     # (W,)
        xx = x_pos[None, :].repeat(h, 1).unsqueeze(0)           # (1,H,W)
        div_x = torch.exp(
            torch.arange(0, n_x, device=device, dtype=dtype)
            * -(math.log(10000.0) / max(n_x, 1))
        ).view(n_x, 1, 1)                                       # (n_x,1,1)

        pe[0, c_half::2, :, :] = torch.sin(xx / div_x)
        pe[0, c_half+1::2, :, :] = torch.cos(xx / div_x)

        return pe

class CrossAttention2D(nn.Module):
    """
    加入 2D 位置編碼的 cross-attn
    - 保留你原本的 MultiheadAttention
    - 在 flatten 之前先加 PositionalEncoding2D
    """
    def __init__(self, dim, num_heads=4, use_pos_encoding: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding2D(dim)

        # 可選的 LayerNorm，讓跨視角 fusion 比較穩
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q_map, kv_map):
            B, C, H, W = q_map.size()
            
            # 1. 計算位置編碼 (只算一次)
            q_pe = self.pos_encoding(q_map) if self.use_pos_encoding else 0
            kv_pe = self.pos_encoding(kv_map) if self.use_pos_encoding else 0

            # 2. 準備 Q 和 K (包含位置資訊)
            q = q_map + q_pe
            k = kv_map + kv_pe
            
            # 3. 準備 V (保持純淨，不加位置資訊)
            v = kv_map  # <--- 關鍵差異在這裡

            # 4. Flatten & Permute
            q = q.flatten(2).permute(0, 2, 1) # (B, HW, C)
            k = k.flatten(2).permute(0, 2, 1)
            v = v.flatten(2).permute(0, 2, 1) # V 也是純淨的

            # LayerNorm (通常 V 也要 Norm)
            q = self.norm_q(q)
            k = self.norm_kv(k) # K 的 Norm
            v = self.norm_kv(v) # V 使用相同的 Norm 參數 (或獨立一個 norm_v)

            # 5. Attention
            # nn.MultiheadAttention(query, key, value)
            out, _ = self.attn(q, k, v) 
            
            out = out.permute(0, 2, 1).view(B, C, H, W)
            return out
class IpsiCrossViewFusion(nn.Module):
    """
    同側 CC<->MLO cross-view + learnable gates
    - 每個方向都有一個 gate（sigmoid 後介於 0~1）
    - 模型可以學到「什麼時候應該多相信 cross-view、什麼時候少一點」
    """
    def __init__(self, dim, heads=4, use_pos_encoding: bool = True):
        super().__init__()
        self.l_cc2mlo = CrossAttention2D(dim, heads, use_pos_encoding)
        self.l_mlo2cc = CrossAttention2D(dim, heads, use_pos_encoding)
        self.r_cc2mlo = CrossAttention2D(dim, heads, use_pos_encoding)
        self.r_mlo2cc = CrossAttention2D(dim, heads, use_pos_encoding)

        # 每個方向一個 gate 參數（初始化為 0 -> 一開始幾乎不改動原始 feature）
        self.alpha_l_cc2mlo = nn.Parameter(torch.zeros(1))
        self.alpha_l_mlo2cc = nn.Parameter(torch.zeros(1))
        self.alpha_r_cc2mlo = nn.Parameter(torch.zeros(1))
        self.alpha_r_mlo2cc = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()

    def _bidir(self, a, b, ab, ba, alpha_ab, alpha_ba):
            # 1. 計算 Cross-Attention
            # delta_ab: Q=a, KV=b -> 形狀同 a, 意義是 "b 中對 a 有用的資訊"
            delta_ab = ab(a, b) 
            
            # delta_ba: Q=b, KV=a -> 形狀同 b, 意義是 "a 中對 b 有用的資訊"
            delta_ba = ba(b, a)

            # 2. 計算 Gating
            g_ab = self.sigmoid(alpha_ab)
            g_ba = self.sigmoid(alpha_ba)

            # 3. 殘差更新 (Residual Update)
            # 修正點：將 delta_ab 加給 a，將 delta_ba 加給 b
            a_out = a + g_ab * delta_ab
            b_out = b + g_ba * delta_ba
            
            return a_out, b_out

    def forward(self, feats):
        lcc, lmlo = feats["L-CC"], feats["L-MLO"]
        rcc, rmlo = feats["R-CC"], feats["R-MLO"]

        lcc, lmlo = self._bidir(
            lcc, lmlo,
            self.l_cc2mlo, self.l_mlo2cc,
            self.alpha_l_cc2mlo, self.alpha_l_mlo2cc
        )
        rcc, rmlo = self._bidir(
            rcc, rmlo,
            self.r_cc2mlo, self.r_mlo2cc,
            self.alpha_r_cc2mlo, self.alpha_r_mlo2cc
        )

        feats["L-CC"], feats["L-MLO"] = lcc, lmlo
        feats["R-CC"], feats["R-MLO"] = rcc, rmlo
        return feats
