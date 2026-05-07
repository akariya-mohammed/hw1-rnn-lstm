import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FREQUENCIES = [1.0, 2.0, 5.0, 10.0]  # Hz — chosen to span slow-to-fast, all below Nyquist (50 Hz)
SAMPLE_RATE = 100   # Hz
DURATION = 10       # seconds
WINDOW_SIZE = 10    # samples per context window
N_FREQS = len(FREQUENCIES)


def generate_signal(freq, amplitude=1.0, noise_pct=0.1, sample_rate=SAMPLE_RATE, duration=DURATION, rng=None):
    """Return (clean, noisy) arrays for a single-frequency sine wave.

    Signal model: y(t) = A * sin(2*pi*f*t) + noise
    where noise ~ N(0, noise_pct * A).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    clean = amplitude * np.sin(2 * np.pi * freq * t)
    noise = rng.normal(0, noise_pct * amplitude, n)
    return clean.astype(np.float32), (clean + noise).astype(np.float32)


def freq_to_onehot(freq_idx):
    """One-hot encode a frequency index into a float32 vector of length N_FREQS."""
    c = np.zeros(N_FREQS, dtype=np.float32)
    c[freq_idx] = 1.0
    return c


class SignalDataset(Dataset):
    """Sliding-window dataset over noisy sine waves at four known frequencies.

    Each sample contains:
        freq_vec  : 1-hot frequency encoding, shape (N_FREQS,)
        noisy     : 10 consecutive noisy signal samples, shape (WINDOW_SIZE,)
        clean     : corresponding noise-free samples, shape (WINDOW_SIZE,)

    The network task is to denoise: predict `clean` from (`freq_vec`, `noisy`).
    """

    def __init__(self, window_size=WINDOW_SIZE, noise_pct=0.1,
                 sample_rate=SAMPLE_RATE, duration=DURATION, seed=42):
        rng = np.random.default_rng(seed)
        self.window_size = window_size
        self._items = []

        for freq_idx, freq in enumerate(FREQUENCIES):
            clean, noisy = generate_signal(
                freq, noise_pct=noise_pct,
                sample_rate=sample_rate, duration=duration, rng=rng
            )
            c = freq_to_onehot(freq_idx)
            n_windows = len(clean) - window_size + 1
            for j in range(n_windows):
                self._items.append((
                    c,
                    noisy[j: j + window_size],
                    clean[j: j + window_size],
                ))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        c, noisy, clean = self._items[idx]
        return (
            torch.from_numpy(c),
            torch.from_numpy(noisy.copy()),
            torch.from_numpy(clean.copy()),
        )


def get_dataloaders(batch_size=64, train_ratio=0.8, noise_pct=0.1, seed=42):
    """Return (train_loader, test_loader) with an 80/20 split."""
    dataset = SignalDataset(noise_pct=noise_pct, seed=seed)
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_test], generator=generator
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
