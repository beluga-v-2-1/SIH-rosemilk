"""
dataset.py
----------
PyTorch Dataset for real-time ANC / Speech Enhancement.

Responsibilities
----------------
1. Load clean speech and noise files from two separate directories.
2. Dynamically mix them at a random SNR in [-5 dB, +15 dB] on every __getitem__
   call so every epoch sees a fresh augmentation.
3. Apply optional augmentation:
   - Random clipping  : simulate mic overload / hard transients (defense env).
   - Slight reverberation : convolve with a small synthetic RIR via torchaudio.
4. Return (noisy_waveform, clean_waveform) as mono tensors of a fixed length
   (clip_duration seconds @ sample_rate).

Directory layout expected
-------------------------
data/
  clean/   *.wav  (speech recordings)
  noise/   *.wav  (gunshots, helicopters, sirens, background noise …)

Usage
-----
>>> from dataset import SpeechNoiseDataset
>>> ds = SpeechNoiseDataset("data/clean", "data/noise", sample_rate=16000,
...                          clip_duration=4.0)
>>> noisy, clean = ds[0]          # tensors of shape (1, 64000)
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load an audio file, resample if needed, and convert to mono.

    Returns
    -------
    Tensor of shape (1, num_samples)
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform                             # (1, T)


def _fix_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad (wrap-around) or truncate a waveform to exactly *target_len* samples."""
    T_len = waveform.shape[-1]
    if T_len >= target_len:
        # Random crop for variety
        start = random.randint(0, T_len - target_len)
        return waveform[..., start: start + target_len]
    else:
        # Wrap-around padding
        repeats = (target_len // T_len) + 1
        waveform = waveform.repeat(1, repeats)
        return waveform[..., :target_len]


def _rms(waveform: torch.Tensor) -> torch.Tensor:
    """Root-mean-square energy of a waveform."""
    return waveform.pow(2).mean().sqrt().clamp(min=1e-9)


def _mix_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale *noise* so that the mixture has the requested SNR w.r.t. *clean*.

    Returns
    -------
    (noisy, clean_normalized)  — both normalised to unit RMS for stable training.
    """
    # Normalise clean to unit RMS
    clean_rms = _rms(clean)
    clean_norm = clean / clean_rms

    # Scale noise to desired SNR
    noise_rms = _rms(noise)
    target_noise_rms = 10 ** (-snr_db / 20.0)        # noise RMS that gives snr_db
    noise_scaled = noise * (target_noise_rms / noise_rms)

    noisy = clean_norm + noise_scaled

    # Prevent clipping while preserving relative levels
    peak = noisy.abs().max().clamp(min=1e-9)
    if peak > 1.0:
        noisy = noisy / peak
        clean_norm = clean_norm / peak

    return noisy, clean_norm


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment_random_clip(
    waveform: torch.Tensor,
    clip_prob: float = 0.3,
    clip_threshold: float = 0.7,
) -> torch.Tensor:
    """Simulate microphone overload / hard transients by hard-clipping a random
    fraction of the signal.

    Parameters
    ----------
    clip_prob       : probability of applying clipping at all.
    clip_threshold  : absolute threshold in [-1, 1] at which clipping occurs.
    """
    if random.random() < clip_prob:
        threshold = random.uniform(clip_threshold, 0.95)
        waveform = waveform.clamp(-threshold, threshold)
    return waveform


def _augment_reverb(
    waveform: torch.Tensor,
    sample_rate: int,
    reverb_prob: float = 0.4,
    rt60_range: Tuple[float, float] = (0.05, 0.3),
) -> torch.Tensor:
    """Add slight reverberation using a synthetic exponentially-decaying RIR.

    The RIR is generated entirely in PyTorch (no external dependency) so it
    runs on CPU/GPU uniformly.  RT60 is kept short (50–300 ms) to mimic
    small, acoustically-treated tactical spaces rather than large rooms.

    Parameters
    ----------
    rt60_range : (min_rt60_s, max_rt60_s) in seconds.
    """
    if random.random() >= reverb_prob:
        return waveform

    rt60 = random.uniform(*rt60_range)

    # Build a simple exponentially decaying noise RIR
    rir_len = int(rt60 * sample_rate)
    if rir_len < 2:
        return waveform

    # Exponential decay envelope
    t = torch.linspace(0, rt60, rir_len)
    decay = torch.exp(-6.9 * t / rt60)           # -60 dB at t = rt60

    # Excite with white noise * decay
    rir = torch.randn(rir_len) * decay
    rir = rir / rir.abs().max().clamp(min=1e-9)   # normalise impulse

    # Convolve via FFT (linear convolution, then trim)
    sig_len = waveform.shape[-1]
    fft_len = sig_len + rir_len - 1

    # Pad to next power of 2 for speed
    fft_size = 1
    while fft_size < fft_len:
        fft_size <<= 1

    sig_f = torch.fft.rfft(waveform, n=fft_size)
    rir_f = torch.fft.rfft(rir.unsqueeze(0), n=fft_size)
    reverbed = torch.fft.irfft(sig_f * rir_f, n=fft_size)[..., :sig_len]

    # Mix dry + wet (keep speech intelligible)
    wet_mix = random.uniform(0.1, 0.35)
    output = (1.0 - wet_mix) * waveform + wet_mix * reverbed

    # Re-normalise to prevent gain build-up
    peak = output.abs().max().clamp(min=1e-9)
    if peak > 1.0:
        output = output / peak

    return output


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpeechNoiseDataset(Dataset):
    """Dynamic speech-noise mixing dataset for ANC / speech enhancement.

    Parameters
    ----------
    clean_dir       : path to directory containing clean speech .wav files.
    noise_dir       : path to directory containing noise .wav files.
    sample_rate     : target sample rate (Hz).  Default: 16 000.
    clip_duration   : length of each training clip (seconds).  Default: 4.0.
    snr_range       : (min_snr_dB, max_snr_dB).  Default: (-5, 15).
    augment_clip    : whether to apply random hard-clipping.  Default: True.
    augment_reverb  : whether to apply synthetic reverberation.  Default: True.
    max_files       : cap total files per split (useful for debugging).
    """

    SUPPORTED_EXT: Tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str,
        sample_rate: int = 16_000,
        clip_duration: float = 4.0,
        snr_range: Tuple[float, float] = (-5.0, 15.0),
        augment_clip: bool = True,
        augment_reverb: bool = True,
        max_files: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.clip_len = int(clip_duration * sample_rate)
        self.snr_range = snr_range
        self.augment_clip = augment_clip
        self.augment_reverb = augment_reverb

        self.clean_files = self._gather_files(clean_dir, max_files)
        self.noise_files = self._gather_files(noise_dir, max_files)

        if len(self.clean_files) == 0:
            raise ValueError(f"No audio files found in clean_dir: {clean_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"No audio files found in noise_dir: {noise_dir}")

        print(
            f"[SpeechNoiseDataset] clean={len(self.clean_files)} files | "
            f"noise={len(self.noise_files)} files | "
            f"sr={sample_rate} Hz | clip={clip_duration}s"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_files(self, directory: str, max_files: Optional[int]) -> List[str]:
        """Recursively collect supported audio files from *directory*."""
        root = Path(directory)
        files = [
            str(p)
            for p in sorted(root.rglob("*"))
            if p.suffix.lower() in self.SUPPORTED_EXT
        ]
        if max_files is not None:
            files = files[:max_files]
        return files

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Dataset length == number of clean utterances.
        # Noise files are randomly sampled, so the effective dataset
        # size is len(clean_files) * len(noise_files) combinations,
        # but we expose len(clean_files) as the epoch length.
        return len(self.clean_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (noisy_waveform, clean_waveform) both of shape (1, clip_len)."""

        # ---- 1. Load clean speech ----------------------------------------
        clean_raw = _load_audio(self.clean_files[idx], self.sample_rate)
        clean_raw = _fix_length(clean_raw, self.clip_len)

        # ---- 2. Load a random noise clip ---------------------------------
        noise_path = random.choice(self.noise_files)
        noise_raw = _load_audio(noise_path, self.sample_rate)
        noise_raw = _fix_length(noise_raw, self.clip_len)

        # ---- 3. Random SNR mix -------------------------------------------
        snr_db = random.uniform(*self.snr_range)
        noisy, clean_norm = _mix_at_snr(clean_raw, noise_raw, snr_db)

        # ---- 4. Augmentation (applied to the noisy mixture only) ----------
        if self.augment_reverb:
            noisy = _augment_reverb(noisy, self.sample_rate)

        if self.augment_clip:
            noisy = _augment_random_clip(noisy)

        return noisy.float(), clean_norm.float()


# ---------------------------------------------------------------------------
# Convenience split helper
# ---------------------------------------------------------------------------

def build_datasets(
    clean_dir: str,
    noise_dir: str,
    val_split: float = 0.1,
    sample_rate: int = 16_000,
    clip_duration: float = 4.0,
    snr_range: Tuple[float, float] = (-5.0, 15.0),
    seed: int = 42,
) -> Tuple["SpeechNoiseDataset", "SpeechNoiseDataset"]:
    """Create train and validation SpeechNoiseDataset splits.

    The split is performed on the *clean* file list; the noise pool is shared.
    Validation set has augmentation disabled for reproducible metric reporting.

    Parameters
    ----------
    val_split : fraction of clean files to hold out for validation.
    seed      : random seed for reproducible split.

    Returns
    -------
    (train_dataset, val_dataset)
    """
    rng = random.Random(seed)

    # Gather all clean files and shuffle
    all_clean = SpeechNoiseDataset(
        clean_dir, noise_dir, sample_rate, clip_duration, snr_range
    ).clean_files
    rng.shuffle(all_clean)

    split_idx = max(1, int(len(all_clean) * (1 - val_split)))
    train_clean = all_clean[:split_idx]
    val_clean = all_clean[split_idx:]

    # Write temp sub-lists to temp dirs is complex; instead we subclass
    # and override the file list directly.
    def _make_split(files: List[str], augment: bool) -> SpeechNoiseDataset:
        ds = SpeechNoiseDataset.__new__(SpeechNoiseDataset)
        ds.sample_rate = sample_rate
        ds.clip_len = int(clip_duration * sample_rate)
        ds.snr_range = snr_range
        ds.augment_clip = augment
        ds.augment_reverb = augment
        ds.clean_files = files
        ds.noise_files = SpeechNoiseDataset(
            clean_dir, noise_dir, sample_rate, clip_duration, snr_range
        ).noise_files
        return ds

    train_ds = _make_split(train_clean, augment=True)
    val_ds = _make_split(val_clean, augment=False)

    print(f"[build_datasets] train={len(train_ds)} | val={len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset.py <clean_dir> <noise_dir>")
        sys.exit(1)

    clean_d, noise_d = sys.argv[1], sys.argv[2]
    train_ds, val_ds = build_datasets(clean_d, noise_d)

    noisy, clean = train_ds[0]
    print(f"noisy shape : {noisy.shape}  dtype={noisy.dtype}")
    print(f"clean shape : {clean.shape}  dtype={clean.dtype}")
    print(f"noisy range : [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"clean range : [{clean.min():.4f}, {clean.max():.4f}]")
    print("Smoke-test passed ✓")
