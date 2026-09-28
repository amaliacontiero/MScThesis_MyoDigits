"""
Subject-Conditional PoseNet (PoseNetSC).

Architectural contribution for cross-subject sEMG-to-joint-angle regression.

Three components:
  1. Subject encoder: K unlabeled EMG windows -> subject embedding z.
     Mean-pooled over K so output is invariant to calibration set size.
  2. FiLM-modulated TCN stem: residual blocks are conditioned on z via
     per-channel scale/shift, allowing subject-specific feature adaptation.
  3. Causal GRU + head: streaming-capable temporal model (no future context).

Inference modes:
  - forward(query, support=Xs)        compute z fresh from support windows
  - forward(query, z=z_precomputed)   use a cached embedding
  - forward(query) after calibrate()  use the most recent calibration

Calibration is unsupervised: ~30 s of unlabeled EMG (K=600 windows at 50 ms hop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from myodigits.models.posenet_seq import _ConvBlock


class SubjectEncoder(nn.Module):
    """
    Lightweight TCN over K unlabeled windows -> subject embedding z.
    Mean-pools across K so the embedding does not depend on calibration set size.

    Input:  (B, K, C, 400)
    Output: (B, z_dim)
    """
    def __init__(self, in_ch=16, base=32, z_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock(in_ch,  base,    k=9, stride=2, dropout=dropout),
            _ConvBlock(base,   base*2,  k=7, stride=2, dropout=dropout),
            _ConvBlock(base*2, base*4,  k=5, stride=2, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(base*4, base*4),
            nn.GELU(),
            nn.Linear(base*4, z_dim),
        )
        self.z_dim = z_dim

    def forward(self, x):
        B, K, C, W = x.shape
        x = x.reshape(B * K, C, W)
        x = self.encoder(x)               # (B*K, base*4, W')
        x = self.pool(x).squeeze(-1)      # (B*K, base*4)
        x = x.reshape(B, K, -1).mean(dim=1)
        return self.proj(x)               # (B, z_dim)


class _FilmResBlock(nn.Module):
    """
    Residual block with per-channel FiLM modulation conditioned on z.

    h = (1 + γ(z)) * BN(W2 GELU(BN(W1 x))) + β(z)
    out = GELU(h + x)

    The (1 + γ) form keeps initialisation near identity (the FiLM linear is
    zero-init), so an untrained subject encoder reduces to a plain residual
    block. This stabilises early training.
    """
    def __init__(self, c, z_dim, k=5, dropout=0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(c)
        self.drop  = nn.Dropout(dropout)
        self.film  = nn.Linear(z_dim, 2 * c)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, z):
        gb = self.film(z)                  # (N, 2c)
        gamma, beta = gb.chunk(2, dim=-1)  # (N, c) each
        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)

        r = x
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        h = (1.0 + gamma) * h + beta
        return F.gelu(h + r)


class PoseNetSC(nn.Module):
    """
    Subject-Conditional PoseNet. Streaming-capable (causal GRU).

    Input:  query   (B, T, C, 400)         streaming EMG sequence
            support (B, K, C, 400)         unlabeled calibration windows  (training)
            z       (B, z_dim)             precomputed embedding (eval / batched)
    Output: (B, T, 20)                     20 joint angles per window
    """
    def __init__(self, in_ch=16, out_dim=20, base=64, dropout=0.1,
                 z_dim=64, gru_hidden=128, gru_layers=1):
        super().__init__()
        self.subject_encoder = SubjectEncoder(in_ch=in_ch, base=32,
                                              z_dim=z_dim, dropout=dropout)

        self.stem = nn.Sequential(
            _ConvBlock(in_ch, base,   k=9, stride=1, dropout=dropout),
            _ConvBlock(base,  base,   k=7, stride=2, dropout=dropout),
            _ConvBlock(base,  base*2, k=7, stride=2, dropout=dropout),
        )
        c = base * 2
        self._enc_dim = c

        self.block1 = _FilmResBlock(c, z_dim, k=5, dropout=dropout)
        self.block2 = _FilmResBlock(c, z_dim, k=5, dropout=dropout)
        self.block3 = _FilmResBlock(c, z_dim, k=5, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.gru = nn.GRU(
            input_size=c,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,                     # CAUSAL — streaming
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

        # Non-persistent cache for the currently calibrated subject embedding.
        self.register_buffer("_z_cache", torch.zeros(1, z_dim), persistent=False)
        self._has_z = False
        self.z_dim  = z_dim

    @torch.no_grad()
    def calibrate(self, support):
        """
        Compute and cache subject embedding from K unlabeled windows.
        support: (K, C, 400) or (1, K, C, 400)
        """
        if support.dim() == 3:
            support = support.unsqueeze(0)
        z = self.subject_encoder(support)
        self._z_cache = z.detach()
        self._has_z   = True

    def _encode(self, query, z):
        """
        query: (B, T, C, W); z: (B, z_dim)
        returns features: (B, T, c)
        """
        B, T, C, W = query.shape
        x = query.reshape(B * T, C, W)
        x = self.stem(x)                                              # (B*T, c, W')
        z_bt = z.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)     # (B*T, z_dim)
        x = self.block1(x, z_bt)
        x = self.block2(x, z_bt)
        x = self.block3(x, z_bt)
        x = self.pool(x).squeeze(-1)                                  # (B*T, c)
        return x.reshape(B, T, self._enc_dim)

    def forward(self, query, support=None, z=None):
        if z is None:
            if support is not None:
                z = self.subject_encoder(support)
            elif self._has_z:
                z = self._z_cache.expand(query.shape[0], -1)
            else:
                raise ValueError(
                    "PoseNetSC requires either `support`, `z`, or a prior "
                    "calibrate() call before forward()."
                )

        feats  = self._encode(query, z)        # (B, T, c)
        ctx, _ = self.gru(feats)               # (B, T, gru_hidden) — causal
        return self.head(ctx)                  # (B, T, 20)
