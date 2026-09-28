import torch.nn as nn

from myodigits.models.blocks import _ConvBlock


class PoseNetTxV2(nn.Module):
    """
    TCN stem (no BN, no residual blocks) + Transformer encoder, no positional embeddings.

    Stem: k=11 first conv, then two stride-2 convs.
    Residual blocks replaced by two dilated ConvBlocks (dilation=2).
    Pre-LN Transformer (norm_first=True); no learnable positional embedding.

    Input:  (B, T, C, 400)
    Output: (B, T, 20)
    """
    def __init__(self, in_ch=16, out_dim=20, base=64, dropout=0.1,
                 tx_heads=4, tx_layers=2, tx_dim_ff=512):
        super().__init__()

        self.stem = nn.Sequential(
            _ConvBlock(in_ch, base,   k=11, stride=1, dropout=dropout),
            _ConvBlock(base,  base,   k=7,  stride=2, dropout=dropout),
            _ConvBlock(base,  base*2, k=7,  stride=2, dropout=dropout),
        )
        c = base * 2   # 128
        self.dilated = nn.Sequential(
            _ConvBlock(c, c, k=5, dilation=2, dropout=dropout),
            _ConvBlock(c, c, k=5, dilation=2, dropout=dropout),
        )
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self._enc_dim = c

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=tx_heads,
            dim_feedforward=tx_dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tx_layers,
                                                 enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(c, 128),
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
        ctx   = self.transformer(feats)                 # (B, T, 128)
        return self.head(ctx)                           # (B, T, 20)
