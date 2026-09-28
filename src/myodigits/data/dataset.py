import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


def _augment_emg(x):
    """
    Strong sEMG augmentation for cross-subject robustness.
    x: (T, C, W) sequence tensor  [float32]
    Returns augmented copy (same shape).

    1. Per-channel gain jitter    U(0.5, 1.5)  — electrode contact variation
    2. Additive Gaussian noise    N(0, 0.10)   — measurement noise
    3. Channel dropout            p=0.5        — electrode misplacement
    4. Time masking               p=0.5        — motion artifact / signal loss
    5. Channel shuffle            p=0.3        — electrode position shift
    """
    x = x.clone()
    T, C, W = x.shape

    # 1. gain jitter
    gain = torch.empty(C, device=x.device).uniform_(0.5, 1.5)
    x = x * gain.view(1, C, 1)

    # 2. additive noise
    x = x + torch.randn_like(x) * 0.10

    # 3. channel dropout: zero one random channel
    if torch.rand(1).item() < 0.5:
        ch = torch.randint(0, C, (1,)).item()
        x[:, ch, :] = 0.0

    # 4. time masking: zero a random contiguous time segment (up to 20% of W)
    if torch.rand(1).item() < 0.5:
        mask_len = torch.randint(1, max(2, W // 5), (1,)).item()
        start    = torch.randint(0, W - mask_len, (1,)).item()
        x[:, :, start:start + mask_len] = 0.0

    # 5. channel shuffle: swap two random adjacent channels
    if torch.rand(1).item() < 0.3:
        i = torch.randint(0, C - 1, (1,)).item()
        x[:, [i, i + 1], :] = x[:, [i + 1, i], :]

    return x


class MyoDigitsDataset(Dataset):
    """Eager dataset: loads all windows from one split directory into RAM."""

    def __init__(self, split_dir, verbose=True):
        split_dir = Path(split_dir)
        files = sorted(split_dir.glob("seg_*.npz"))

        all_X, all_y = [], []
        for f in files:
            d = np.load(f)
            all_X.append(d["X"])
            all_y.append(d["y"])

        self.X = torch.from_numpy(np.concatenate(all_X, axis=0))  # (N, 8, 400)
        self.y = torch.from_numpy(np.concatenate(all_y, axis=0))  # (N, 20)
        if verbose:
            print(f"Loaded {len(self.X):,} windows from {split_dir.name}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LosoDataset(Dataset):
    """Eager dataset spanning multiple user directories. Loads all data into RAM."""

    def __init__(self, user_dirs):
        all_X, all_y = [], []
        total = 0
        for d in user_dirs:
            for f in sorted(Path(d).glob("seg_*.npz")):
                data = np.load(f)
                all_X.append(data["X"])
                all_y.append(data["y"])
                total += data["X"].shape[0]

        self.X = torch.from_numpy(np.concatenate(all_X, axis=0))
        self.y = torch.from_numpy(np.concatenate(all_y, axis=0))
        print(f"Loaded {len(self.X):,} windows from {len(user_dirs)} users")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _build_seq_index(files, seq_len):
    """Return (all_X list, all_y list, valid_starts list) respecting seg boundaries."""
    all_X, all_y, valid_starts = [], [], []
    offset = 0
    for f in files:
        d = np.load(f)
        n = d["X"].shape[0]
        all_X.append(d["X"])
        all_y.append(d["y"])
        for s in range(n - seq_len + 1):
            valid_starts.append(offset + s)
        offset += n
    return all_X, all_y, valid_starts


class SeqDataset(Dataset):
    """
    Sequence dataset for a single split directory.
    Each item is (seq_len, 8, 400) X and (seq_len, 20) y.
    Sequences never cross trial (seg file) boundaries.
    """

    def __init__(self, split_dir, seq_len=32, verbose=True):
        files = sorted(Path(split_dir).glob("seg_*.npz"))
        all_X, all_y, valid_starts = _build_seq_index(files, seq_len)

        self.X = torch.from_numpy(np.concatenate(all_X, axis=0))
        self.y = torch.from_numpy(np.concatenate(all_y, axis=0))
        self.seq_len = seq_len
        self.valid_starts = valid_starts
        if verbose:
            print(f"Loaded {len(self.X):,} windows, {len(valid_starts):,} sequences "
                  f"from {Path(split_dir).name}")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        return self.X[s:s + self.seq_len], self.y[s:s + self.seq_len]


def _build_stft_window_index(files, input_len, win_hop):
    """
    Load continuous per-trial (T,8) EMG and (T,20) angles, return
    (all_X, all_y, valid_starts) where each start indexes a full 1248-sample
    window that lies entirely within one trial (never crosses seg boundary).
    """
    all_X, all_y, valid_starts = [], [], []
    offset = 0
    for f in files:
        d = np.load(f)
        n = d["X"].shape[0]
        all_X.append(d["X"])
        all_y.append(d["y"])
        last_start = n - input_len
        for s in range(0, last_start + 1, win_hop):
            valid_starts.append(offset + s)
        offset += n
    return all_X, all_y, valid_starts


class _StftBase(Dataset):
    """
    Shared logic for STFT-model datasets. Each item:
      X: (8, input_len) raw z-scored EMG window
      y: (num_frames, 20) joint angles at STFT frame centres
    so output matches the (B, T, 20) seq convention used by train/eval.
    """
    def _setup(self, all_X, all_y, valid_starts, input_len,
               num_frames, hop_length, window_length):
        self.X = torch.from_numpy(np.concatenate(all_X, axis=0))   # (totalT, 8)
        self.y = torch.from_numpy(np.concatenate(all_y, axis=0))   # (totalT, 20)
        self.valid_starts = valid_starts
        self.input_len    = input_len
        # frame-centre target indices within a window: i*hop + window//2
        self.frame_off = (window_length // 2
                          + torch.arange(num_frames) * hop_length)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        x = self.X[s:s + self.input_len].T.contiguous()   # (8, input_len)
        y = self.y[s + self.frame_off]                    # (num_frames, 20)
        return x, y


class StftDataset(_StftBase):
    """STFT dataset for a single split/user directory (test folds)."""
    def __init__(self, split_dir, input_len=1248, win_hop=256,
                 num_frames=32, hop_length=32, window_length=256, verbose=True):
        files = sorted(Path(split_dir).glob("seg_*.npz"))
        aX, aY, starts = _build_stft_window_index(files, input_len, win_hop)
        self._setup(aX, aY, starts, input_len, num_frames, hop_length, window_length)
        if verbose:
            print(f"Loaded {len(starts):,} STFT windows from {Path(split_dir).name}")


class LosoStftDataset(_StftBase):
    """STFT dataset spanning multiple user directories for LOSO CV (train/val)."""
    def __init__(self, user_dirs, input_len=1248, win_hop=256,
                 num_frames=32, hop_length=32, window_length=256):
        all_X, all_y, valid_starts = [], [], []
        offset = 0
        for user_dir in user_dirs:
            files = sorted(Path(user_dir).glob("seg_*.npz"))
            uX, uY, ustarts = _build_stft_window_index(files, input_len, win_hop)
            all_X.extend(uX)
            all_y.extend(uY)
            for s in ustarts:
                valid_starts.append(offset + s)
            offset += sum(x.shape[0] for x in uX)
        self._setup(all_X, all_y, valid_starts, input_len,
                    num_frames, hop_length, window_length)
        print(f"Loaded {len(valid_starts):,} STFT windows from {len(user_dirs)} users")


class LosoSeqDataset(Dataset):
    """
    Sequence dataset spanning multiple user directories for LOSO CV.
    Each item is (seq_len, C, 400) X and (seq_len, 20) y.
    Sequences never cross trial (seg file) boundaries.
    Set augment=True for training to apply sEMG data augmentation.
    """

    def __init__(self, user_dirs, seq_len=32, augment=False):
        all_X, all_y, valid_starts = [], [], []
        offset = 0
        for user_dir in user_dirs:
            files = sorted(Path(user_dir).glob("seg_*.npz"))
            uX, uy, ustarts = _build_seq_index(files, seq_len)
            all_X.extend(uX)
            all_y.extend(uy)
            for s in ustarts:
                valid_starts.append(offset + s)
            offset += sum(x.shape[0] for x in uX)

        self.X = torch.from_numpy(np.concatenate(all_X, axis=0))
        self.y = torch.from_numpy(np.concatenate(all_y, axis=0))
        self.seq_len = seq_len
        self.valid_starts = valid_starts
        self.augment = augment
        print(f"Loaded {len(self.X):,} windows, {len(valid_starts):,} sequences "
              f"from {len(user_dirs)} users" + (" [augment=ON]" if augment else ""))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        x = self.X[s:s + self.seq_len]
        y = self.y[s:s + self.seq_len]
        if self.augment:
            x = _augment_emg(x)
        return x, y
