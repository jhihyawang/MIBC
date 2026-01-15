
import torch
import torch.nn as nn

class CrossAttn2D(nn.Module):
    """
    src provides Q, tgt provides K/V, output is a delta for tgt (same shape as tgt).
    Works on feature maps: (B,C,H,W) by flattening to tokens (B,HW,C).
    """
    def __init__(self, dim, heads=4, norm=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim) if norm else nn.Identity()
        self.norm_kv = nn.LayerNorm(dim) if norm else nn.Identity()

    def forward_delta(self, src_map, tgt_map):
        B, C, H, W = tgt_map.shape

        src = src_map.flatten(2).transpose(1, 2)  # (B, HW, C)
        tgt = tgt_map.flatten(2).transpose(1, 2)  # (B, HW, C)

        q = self.norm_q(src)
        kv = self.norm_kv(tgt)
        out, _ = self.attn(q, kv, kv)             # (B, HW, C)

        delta = out.transpose(1, 2).view(B, C, H, W)
        return delta


class SideInvariantCrossViewFusion(nn.Module):
    """
    Side-invariant version:
    - pairs: ipsi (LCC,LMLO), (RCC,RMLO) and contra (LCC,RCC), (LMLO,RMLO)
    - bidirectional deltas for each pair
    - synchronous update: all deltas computed from ORIGINAL features, then added once
    """
    def __init__(self, dim, heads=4, use_gates=True, gate_init=-5.0):
        super().__init__()
        self.xattn = CrossAttn2D(dim, heads=heads, norm=True)

        self.use_gates = use_gates
        if use_gates:
            # 8 directed edges
            self.alpha = nn.ParameterDict({
                "LCC<-LMLO":  nn.Parameter(torch.tensor([gate_init])),
                "LMLO<-LCC":  nn.Parameter(torch.tensor([gate_init])),
                "RCC<-RMLO":  nn.Parameter(torch.tensor([gate_init])),
                "RMLO<-RCC":  nn.Parameter(torch.tensor([gate_init])),
                "LCC<-RCC":   nn.Parameter(torch.tensor([gate_init])),
                "RCC<-LCC":   nn.Parameter(torch.tensor([gate_init])),
                "LMLO<-RMLO": nn.Parameter(torch.tensor([gate_init])),
                "RMLO<-LMLO": nn.Parameter(torch.tensor([gate_init])),
            })
            self.sigmoid = nn.Sigmoid()

    def _g(self, k):
        if not self.use_gates:
            return 1.0
        return self.sigmoid(self.alpha[k])  # scalar in (0,1)

    def forward(self, feats):
        """
        feats: dict keys ["L-CC","R-CC","L-MLO","R-MLO"], each (B,C,H,W)
        returns updated feats dict
        """
        LCC0  = feats["L-CC"]
        RCC0  = feats["R-CC"]
        LMLO0 = feats["L-MLO"]
        RMLO0 = feats["R-MLO"]

        # ---- compute ALL deltas from ORIGINAL features (critical) ----
        # ipsi
        d_LCC_from_LMLO  = self.xattn.forward_delta(LMLO0, LCC0)   # LCC <- LMLO
        d_LMLO_from_LCC  = self.xattn.forward_delta(LCC0, LMLO0)   # LMLO <- LCC
        d_RCC_from_RMLO  = self.xattn.forward_delta(RMLO0, RCC0)   # RCC <- RMLO
        d_RMLO_from_RCC  = self.xattn.forward_delta(RCC0, RMLO0)   # RMLO <- RCC

        # contra (same view-type across sides)
        d_LCC_from_RCC   = self.xattn.forward_delta(RCC0, LCC0)    # LCC <- RCC
        d_RCC_from_LCC   = self.xattn.forward_delta(LCC0, RCC0)    # RCC <- LCC
        d_LMLO_from_RMLO = self.xattn.forward_delta(RMLO0, LMLO0)  # LMLO <- RMLO
        d_RMLO_from_LMLO = self.xattn.forward_delta(LMLO0, RMLO0)  # RMLO <- LMLO

        # ---- synchronous add (no sequential dependency) ----
        LCC  = LCC0  + self._g("LCC<-LMLO")  * d_LCC_from_LMLO  + self._g("LCC<-RCC")   * d_LCC_from_RCC
        LMLO = LMLO0 + self._g("LMLO<-LCC")  * d_LMLO_from_LCC  + self._g("LMLO<-RMLO") * d_LMLO_from_RMLO
        RCC  = RCC0  + self._g("RCC<-RMLO")  * d_RCC_from_RMLO  + self._g("RCC<-LCC")   * d_RCC_from_LCC
        RMLO = RMLO0 + self._g("RMLO<-RCC")  * d_RMLO_from_RCC  + self._g("RMLO<-LMLO") * d_RMLO_from_LMLO

        return {"L-CC": LCC, "R-CC": RCC, "L-MLO": LMLO, "R-MLO": RMLO}



