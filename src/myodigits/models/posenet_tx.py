import torch
import torch.nn as nn

from myodigits.models.posenet_seq import _ConvBlock, _ResidBlock


class PoseNetTx(nn.Module):
    """
    TCN stem + Transformer encoder for cross-window temporal context.

    Drop-in replacement for PoseNetSeq's BiGRU stage.
    Pre-LN Transformer (norm_first=True) for stable training.

    Input:  (B, T, C, 400)
    Output: (B, T, 20)
    """
    def __init__(self, in_ch=16, out_dim=20, base=64, dropout=0.1,
                 tx_heads=4, tx_layers=2, tx_dim_ff=512, max_seq_len=128):
        super().__init__()

        # ── TCN stem (identical to PoseNetSeq) ───────────────────────────────
        self.stem = nn.Sequential(
            _ConvBlock(in_ch, base,   k=9, stride=1, dropout=dropout),
            _ConvBlock(base,  base,   k=7, stride=2, dropout=dropout),
            _ConvBlock(base,  base*2, k=7, stride=2, dropout=dropout),
        )
        c = base * 2   # 128
        self.blocks = nn.Sequential(
            _ResidBlock(c, k=5, dropout=dropout),
            _ResidBlock(c, k=5, dropout=dropout),
            _ResidBlock(c, k=5, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self._enc_dim = c

        # ── learnable positional embedding ────────────────────────────────────
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, c))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ── Transformer encoder (pre-LN) ──────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=tx_heads,
            dim_feedforward=tx_dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN: more stable than post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tx_layers,
                                                 enable_nested_tensor=False)

        # ── head (same dimensions as PoseNetSeq) ─────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(c, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def _encode(self, x):
        """(N, C, 400) -> (N, enc_dim)"""
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        """(B, T, C, 400) -> (B, T, 20)"""
        B, T, C, W = x.shape
        feats = self._encode(x.reshape(B * T, C, W))   # (B*T, enc_dim)
        feats = feats.reshape(B, T, self._enc_dim)      # (B, T, 128)
        feats = feats + self.pos_emb[:, :T, :]          # add positional encoding
        ctx   = self.transformer(feats)                 # (B, T, 128)
        return self.head(ctx)                           # (B, T, 20)
