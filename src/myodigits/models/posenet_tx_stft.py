import torch
import torch.nn as nn

from myodigits.models.posenet_tx import PoseNetTx


class _StftStem(nn.Module):
    """
    Per-window STFT 2D-CNN encoder — the EMGNet front-end, adapted to encode a
    single (C, 400) window into a (C_out)-d embedding.

    Mirrors EMGNetAngles._spectrogram (same Hann STFT, magnitude, onesided,
    center=False, first `num_frequencies` bins) so the *only* thing that differs
    from PoseNetTx's TCN stem is the front-end inductive bias — everything
    downstream (Transformer + head) is identical.

    (N, C, 400) -> STFT magnitude (N, C, frames, freq) -> 2D-CNN -> (N, C_out)
    """
    def __init__(self, in_ch, out_c, dropout=0.1,
                 window_length=256, hop_length=32, num_frequencies=64,
                 cnn_channels=(32, 64, 128)):
        super().__init__()
        self.in_ch           = in_ch
        self.window_length   = window_length
        self.hop_length      = hop_length
        self.num_frequencies = num_frequencies
        self.register_buffer("hann_window", torch.hann_window(window_length))

        # 2D CNN over (C, frames, freq). Pool the freq axis only — the
        # within-window frame axis is short (~5), so we keep it intact.
        # `cnn_channels` controls capacity (used to param-match the TCN stem).
        c1, c2, c3 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(c1), nn.GELU(), nn.MaxPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(c1, c2, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(c2), nn.GELU(), nn.MaxPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(c2, c3, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(c3), nn.GELU(),
        )
        self.proj = nn.Conv2d(c3, out_c, kernel_size=1, bias=False)  # 1x1 -> enc_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _spectrogram(self, x):
        """(N, C, L) raw window -> (N, C, frames, num_frequencies) magnitude."""
        N = x.size(0)
        x = x.reshape(N * self.in_ch, x.size(2))
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(
                x.float(),
                n_fft=self.window_length,
                hop_length=self.hop_length,
                win_length=self.window_length,
                window=self.hann_window,
                center=False, pad_mode="reflect",
                normalized=False, onesided=True,
                return_complex=True,
            ).abs()
        spec = spec.view(N, self.in_ch, spec.size(1), spec.size(2))
        spec = spec.transpose(2, 3)[..., :self.num_frequencies]   # (N, C, frames, freq)
        return spec

    def forward(self, x):
        """(N, C, 400) -> (N, out_c)"""
        spec = self._spectrogram(x)
        h = self.proj(self.cnn(spec))
        return self.pool(h).flatten(1)


class PoseNetTxStft(PoseNetTx):
    """
    PoseNetTx with the per-window TCN stem replaced by an STFT 2D-CNN stem.

    Single-variable ablation against PoseNetTx: identical positional embedding,
    Transformer encoder, head, sequence length, and I/O contract
    (B, T, C, 400) -> (B, T, 20). The ONLY difference is the per-window
    front-end (STFT 2D-CNN vs TCN), so any metric gap is attributable to the
    front-end alone.
    """
    def __init__(self, in_ch=8, out_dim=20, base=64, dropout=0.1,
                 tx_heads=4, tx_layers=2, tx_dim_ff=512, max_seq_len=128,
                 window_length=256, hop_length=32, num_frequencies=64,
                 cnn_channels=(50, 116, 240)):   # param-matched to PoseNetTx TCN stem
        super().__init__(in_ch=in_ch, out_dim=out_dim, base=base, dropout=dropout,
                         tx_heads=tx_heads, tx_layers=tx_layers,
                         tx_dim_ff=tx_dim_ff, max_seq_len=max_seq_len)
        c = base * 2   # == self._enc_dim, matches the Transformer d_model
        # swap the TCN front-end for the STFT 2D-CNN front-end
        self.stem   = _StftStem(in_ch, c, dropout,
                                window_length, hop_length, num_frequencies,
                                cnn_channels=cnn_channels)
        self.blocks = nn.Identity()
        self.pool   = nn.Identity()

    def _encode(self, x):
        """(N, C, 400) -> (N, enc_dim)"""
        return self.stem(x)
