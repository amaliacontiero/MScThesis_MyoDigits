"""Experiment A dataset wrappers.

These mirror LosoSeqDataset / SeqDataset exactly, inserting the strict
re-standardisation per participant BEFORE concatenation. The original classes
are left untouched so the definitive pipeline stays reproducible bit-for-bit.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from myodigits.data.dataset import _build_seq_index, _augment_emg
from strict_norm import restandardise_user


class StrictLosoSeqDataset(Dataset):
    """Multi-participant sequence dataset under fold-fitted statistics."""

    def __init__(self, user_dirs, meta, stats, seq_len=64, augment=False, tag=""):
        all_X, all_y, valid_starts = [], [], []
        offset = 0
        for user_dir in user_dirs:
            ukey = Path(user_dir).name
            files = sorted(Path(user_dir).glob("seg_*.npz"))
            uX, uy, ustarts = _build_seq_index(files, seq_len)
            # re-standardise this participant's arrays once, before they are pooled
            pairs = [restandardise_user(x, y, ukey, meta, stats) for x, y in zip(uX, uy)]
            uX = [p[0] for p in pairs]
            uy = [p[1] for p in pairs]
            all_X.extend(uX); all_y.extend(uy)
            for s in ustarts:
                valid_starts.append(offset + s)
            offset += sum(x.shape[0] for x in uX)

        self.X = torch.from_numpy(np.concatenate(all_X, axis=0))
        self.y = torch.from_numpy(np.concatenate(all_y, axis=0))
        self.seq_len = seq_len
        self.valid_starts = valid_starts
        self.augment = augment
        print(f"[strict{tag}] {len(self.X):,} windows, {len(valid_starts):,} sequences "
              f"from {len(user_dirs)} participants" + (" [augment=ON]" if augment else ""))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        x = self.X[s:s + self.seq_len]
        y = self.y[s:s + self.seq_len]
        if self.augment:
            x = _augment_emg(x)
        return x, y




class StrictSeqDataset(Dataset):
    """Single-participant (test) sequence dataset under fold-fitted statistics."""

    def __init__(self, user_dir, meta, stats, seq_len=64):
        ukey = Path(user_dir).name
        files = sorted(Path(user_dir).glob("seg_*.npz"))
        uX, uy, starts = _build_seq_index(files, seq_len)
        pairs = [restandardise_user(x, y, ukey, meta, stats) for x, y in zip(uX, uy)]
        self.X = torch.from_numpy(np.concatenate([p[0] for p in pairs], axis=0))
        self.y = torch.from_numpy(np.concatenate([p[1] for p in pairs], axis=0))
        self.seq_len = seq_len
        self.valid_starts = starts
        print(f"[strict/test] {len(self.X):,} windows, {len(starts):,} sequences from {ukey}")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        return self.X[s:s + self.seq_len], self.y[s:s + self.seq_len]
