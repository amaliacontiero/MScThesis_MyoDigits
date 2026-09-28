import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=5, stride=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _ResidBlock(nn.Module):
    def __init__(self, c, k=5, dropout=0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(c)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return F.gelu(x + r)


class PoseNetSeq(nn.Module):
    """
    Per-window TCN encoder + bidirectional GRU for temporal context.

    Stage 1: shared TCN stem encodes each (8, 400) window -> 128-d vector.
    Stage 2: BiGRU across the T-window sequence for cross-window context.
    Stage 3: per-timestep linear head -> 20 joint angles.

    Input:  (B, T, 8, 400)
    Output: (B, T, 20)
    """
    def __init__(self, in_ch=8, out_dim=20, base=64, dropout=0.1,
                 gru_hidden=128, gru_layers=2):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBlock(in_ch, base,   k=9, stride=1, dropout=dropout),  # (B, 64,  400)
            _ConvBlock(base,  base,   k=7, stride=2, dropout=dropout),  # (B, 64,  200)
            _ConvBlock(base,  base*2, k=7, stride=2, dropout=dropout),  # (B, 128, 100)
        )
        c = base * 2
        self.blocks = nn.Sequential(
            _ResidBlock(c, k=5, dropout=dropout),
            _ResidBlock(c, k=5, dropout=dropout),
            _ResidBlock(c, k=5, dropout=dropout),
        )
        self.pool    = nn.AdaptiveAvgPool1d(1)
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
        """(N, 8, 400) -> (N, enc_dim)"""
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        """(B, T, 8, 400) -> (B, T, 20)"""
        B, T, C, W = x.shape
        feats = self._encode(x.reshape(B * T, C, W))  # (B*T, enc_dim)
        feats = feats.reshape(B, T, self._enc_dim)
        ctx, _ = self.gru(feats)                       # (B, T, 2*gru_hidden)
        return self.head(ctx)                          # (B, T, 20)
