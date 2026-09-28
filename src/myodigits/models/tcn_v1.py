import torch.nn as nn

from myodigits.models.posenet_seq import _ConvBlock, _ResidBlock


class PoseNetTCN(nn.Module):
    """
    TCN stem only — no temporal context across windows.

    Ablation baseline for the BiGRU/Transformer temporal stage in PoseNetSeq.
    Each window is encoded independently; the sequence dimension is preserved
    only for interface compatibility with LosoSeqDataset / the training loop.

    Input:  (B, T, C, 400)
    Output: (B, T, 20)
    """
    def __init__(self, in_ch=16, out_dim=20, base=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBlock(in_ch, base,   k=9, stride=1, dropout=dropout),
            _ConvBlock(base,  base,   k=7, stride=2, dropout=dropout),
            _ConvBlock(base,  base*2, k=7, stride=2, dropout=dropout),
        )
        c = base * 2
        self.blocks = nn.Sequential(
            _ResidBlock(c, k=5, dropout=dropout),
            _ResidBlock(c, k=5, dropout=dropout),
            _ResidBlock(c, k=5, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self._enc_dim = c

    def _encode(self, x):
        """(N, C, 400) -> (N, enc_dim)"""
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        """(B, T, C, 400) -> (B, T, 20)"""
        B, T, C, W = x.shape
        feats = self._encode(x.reshape(B * T, C, W))   # (B*T, enc_dim)
        feats = feats.reshape(B, T, self._enc_dim)
        return self.head(feats)                         # (B, T, 20)
