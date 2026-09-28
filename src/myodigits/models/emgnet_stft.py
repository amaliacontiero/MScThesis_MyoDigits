import torch
import torch.nn as nn


class EMGNetAngles(nn.Module):
    """
    Port of PiMForce's EMGNet_angles (Seo et al., NeurIPS 2024), adapted for
    20-joint-angle regression instead of hand-pressure estimation.

    Faithful to the original EMG branch:
      raw EMG (B, C, L) -> STFT magnitude spectrogram (B, C, frames, freq)
      -> 2D CNN encoder (3x Conv2d+BN+ReLU+MaxPool((2,4)))
      -> 2D CNN decoder (3x Conv2d+BN+ReLU+bilinear Upsample) collapsing freq
      -> linear head -> per-frame angles

    Output is (B, num_frames, out_dim) to match the seq convention (B, T, 20),
    so the existing train.py / eval.py work unchanged.

    Input:  (B, C, L)   L = (num_frames + window_length//hop_length - 1) * hop_length
    Output: (B, num_frames, out_dim)
    """
    def __init__(self, in_ch=8, out_dim=20, num_frames=32, num_frequencies=64,
                 window_length=256, hop_length=32):
        super().__init__()
        self.in_ch          = in_ch
        self.out_dim        = out_dim
        self.num_frames     = num_frames
        self.num_frequencies = num_frequencies
        self.window_length  = window_length
        self.hop_length     = hop_length

        self.register_buffer("hann_window", torch.hann_window(window_length))

        self.encoder = nn.Sequential(
            self._enc_block(in_ch, 32,  (2, 4)),
            self._enc_block(32,   128, (2, 4)),
            self._enc_block(128,  256, (2, 4)),
        )
        self.decoder = nn.Sequential(
            self._dec_block(256, 256, (num_frames // 4, 1)),
            self._dec_block(256, 128, (num_frames // 2, 1)),
            self._dec_block(128, 64,  (num_frames,      1)),
        )
        self.linear = nn.Linear(64, out_dim)

    @staticmethod
    def _enc_block(cin, cout, downsample):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=(3, 3), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=downsample, stride=downsample),
        )

    @staticmethod
    def _dec_block(cin, cout, upsample):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=(3, 3), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=False),
            nn.Upsample(size=upsample, mode="bilinear", align_corners=False),
        )

    def _spectrogram(self, x):
        """(B, C, L) raw EMG -> (B, C, num_frames, num_frequencies) magnitude."""
        B = x.size(0)
        x = x.reshape(B * self.in_ch, x.size(2))
        # STFT in float32 (complex ops are not autocast-safe)
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
        spec = spec.view(B, self.in_ch, spec.size(1), spec.size(2))
        spec = spec.transpose(2, 3)[..., :self.num_frequencies]   # (B, C, frames, freq)
        return spec

    def forward(self, x):
        """(B, C, L) -> (B, num_frames, out_dim)"""
        spec = self._spectrogram(x)
        h = self.encoder(spec)               # (B, 256, frames/8, 1)
        h = self.decoder(h)                  # (B, 64, num_frames, 1)
        h = h.view(h.size(0), h.size(1), h.size(2))  # (B, 64, num_frames)
        h = h.transpose(1, 2)                # (B, num_frames, 64)
        return self.linear(h)                # (B, num_frames, out_dim)
