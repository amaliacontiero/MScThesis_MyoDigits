import torch.nn as nn

from myodigits.models.blocks import _ConvBlock


class PoseNetSeqV2(nn.Module):
    """
    Per-window TCN encoder (no BN, no residual blocks) + bidirectional GRU.

    Stem: k=11 first conv, then two stride-2 convs.
    Residual blocks replaced by two dilated ConvBlocks (dilation=2).

    Input:  (B, T, C, 400)
    Output: (B, T, 20)
    """
    def __init__(self, in_ch=16, out_dim=20, base=64, dropout=0.1,
                 gru_hidden=128, gru_layers=2):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBlock(in_ch, base,   k=11, stride=1, dropout=dropout),  # (B, 64,  400)
            _ConvBlock(base,  base,   k=7,  stride=2, dropout=dropout),  # (B, 64,  200)
            _ConvBlock(base,  base*2, k=7,  stride=2, dropout=dropout),  # (B, 128, 100)
        )
        c = base * 2
        self.dilated = nn.Sequential(
            _ConvBlock(c, c, k=5, dilation=2, dropout=dropout),
            _ConvBlock(c, c, k=5, dilation=2, dropout=dropout),
        )
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self._enc_dim = c

        self.gru = nn.GRU(
            input_size=c,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def _encode(self, x):
        """(N, C, 400) -> (N, enc_dim)"""
        x = self.stem(x)
        x = self.dilated(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        """(B, T, C, 400) -> (B, T, 20)"""
        B, T, C, W = x.shape
        feats = self._encode(x.reshape(B * T, C, W))   # (B*T, enc_dim)
        feats = feats.reshape(B, T, self._enc_dim)
        ctx, _ = self.gru(feats)                        # (B, T, 2*gru_hidden)
        return self.head(ctx)                           # (B, T, 20)
