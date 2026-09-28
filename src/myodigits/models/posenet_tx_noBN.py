import torch
import torch.nn as nn

from myodigits.models.blocks import _ConvBlock, _ResidBlock


class PoseNetTxNoBN(nn.Module):
    """
    Ablation of PoseNetTx: BatchNorm removed from all conv layers.
    Residual connections, positional embedding, and Transformer unchanged.

    Input:  (B, T, C, 400)
    Output: (B, T, 20)
    """
    def __init__(self, in_ch=16, out_dim=20, base=64, dropout=0.1,
                 tx_heads=4, tx_layers=2, tx_dim_ff=512, max_seq_len=128):
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
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self._enc_dim = c

        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, c))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c, nhead=tx_heads, dim_feedforward=tx_dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tx_layers,
                                                 enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(c, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, out_dim),
        )

    def _encode(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        """(B, T, C, 400) -> (B, T, 20)"""
        B, T, C, W = x.shape
        feats = self._encode(x.reshape(B * T, C, W))
        feats = feats.reshape(B, T, self._enc_dim)
        feats = feats + self.pos_emb[:, :T, :]
        ctx   = self.transformer(feats)
        return self.head(ctx)
