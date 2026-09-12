"""
model.py
--------
Lightweight Audio UNet for real-time speech enhancement / ANC.

Architecture
------------
                    ┌─────────────────────────────────────────┐
  Noisy STFT  ───►  │  STFT  │  Encoder  │  Bottleneck  │     │
  (magnitude +       │        │  (4 down  │  (LSTM or    │     │
   phase)            │        │   blocks) │   GRU)       │     │  ──►  Enhanced waveform
                    │        │           │  Decoder     │     │
                    │        │           │  (4 up-skip  │     │
                    └─────────────────────────────────────────┘

Signal flow
-----------
1. Input waveform  →  STFT  →  complex spectrogram (B, 2, F, T)
                               [ch0 = real, ch1 = imag]
2. Encoder: 4 × (Conv2d + BN + PReLU) with stride=(1,2) – compress time
3. Bottleneck: Bidirectional GRU over the time axis for temporal context
4. Decoder: 4 × (ConvTranspose2d + BN + PReLU) + skip connections (UNet style)
5. Mask head: 1×1 Conv → tanh-bounded complex mask (cIRM-style)
6. Masking: enhanced_stft = mask * noisy_stft
7. iSTFT  →  enhanced waveform

Parameter budget
----------------
With base_channels=32 the model has ≈ 1.8M parameters – well under the 5M cap,
leaving headroom for the bottleneck GRU and feature growth on the encoder path.

Usage
-----
>>> from model import AudioUNet, count_parameters
>>> model = AudioUNet()
>>> print(f"Parameters: {count_parameters(model):,}")
>>> noisy = torch.randn(2, 1, 16000)  # batch=2, mono, 1 second @ 16 kHz
>>> enhanced = model(noisy)           # (2, 1, 16000)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# STFT / iSTFT wrapper  (kept inside the model graph for end-to-end training)
# ---------------------------------------------------------------------------

class STFT(nn.Module):
    """Differentiable STFT/iSTFT wrapper using torch.stft / torch.istft.

    Parameters
    ----------
    n_fft       : FFT size.                 Default: 512
    hop_length  : Hop size in samples.      Default: 128  (75 % overlap)
    win_length  : Window length in samples. Default: 512
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Register as buffer so it moves with .to(device)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Waveform → complex spectrogram stacked as (B, 2, F, T).

        Parameters
        ----------
        waveform : (B, 1, L)  mono waveform

        Returns
        -------
        spec : (B, 2, F, T)  — ch0=real, ch1=imag
        """
        B = waveform.shape[0]
        x = waveform.squeeze(1)  # (B, L)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )  # (B, F, T)  complex

        # Stack real/imag on channel dim → (B, 2, F, T)
        return torch.stack([spec.real, spec.imag], dim=1)

    def inverse(
        self,
        spec: torch.Tensor,
        length: int | None = None,
    ) -> torch.Tensor:
        """Complex spectrogram (B, 2, F, T) → waveform (B, 1, L).

        Parameters
        ----------
        spec   : (B, 2, F, T)
        length : original waveform length for trimming
        """
        # Reconstruct complex tensor
        complex_spec = torch.complex(spec[:, 0], spec[:, 1])  # (B, F, T)

        wav = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
        )  # (B, L)

        return wav.unsqueeze(1)  # (B, 1, L)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → PReLU encoder block.

    stride=(1,2) halves the time dimension to gradually compress context.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TransposeBlock(nn.Module):
    """ConvTranspose2d → BatchNorm2d → PReLU decoder block.

    Mirrors ConvBlock; receives skip-connection concatenated input.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (1, 1),
        output_padding: Tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Concatenate skip connection on channel dim
        x = torch.cat([x, skip], dim=1)
        return self.act(self.bn(self.conv(x)))


class GRUBottleneck(nn.Module):
    """Bidirectional GRU over the time axis for sequential context.

    Reshapes (B, C, F, T) → (B*F, T, C) → GRU → back to (B, C_out, F, T).
    This captures long-range temporal dependencies without full attention.
    """

    def __init__(self, channels: int, hidden: int, num_layers: int = 2) -> None:
        super().__init__()
        # bidirectional → output dim = 2 * hidden
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * hidden, channels)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        # (B, C, F, T) → (B*F, T, C)
        x_perm = x.permute(0, 2, 3, 1).reshape(B * F, T, C)
        out, _ = self.gru(x_perm)                  # (B*F, T, 2H)
        out = self.proj(out)                        # (B*F, T, C)
        out = self.ln(out)
        # (B*F, T, C) → (B, C, F, T)
        out = out.reshape(B, F, T, C).permute(0, 3, 1, 2)
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AudioUNet(nn.Module):
    """Lightweight Audio UNet for speech enhancement / ANC.

    Parameters
    ----------
    n_fft        : STFT FFT size.
    hop_length   : STFT hop size.
    win_length   : STFT window length.
    base_ch      : Base channel count for encoder (doubles each block).
                   Default 32 gives ≈ 1.8 M parameters.
    gru_hidden   : Hidden size of each GRU direction in the bottleneck.
    mask_act     : Activation on the mask output.
                   'tanh' → bounded cIRM in [-1, 1] (recommended).
                   'sigmoid' → magnitude-only mask in [0, 1].
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        base_ch: int = 32,
        gru_hidden: int = 64,
        mask_act: str = "tanh",
    ) -> None:
        super().__init__()

        self.stft = STFT(n_fft, hop_length, win_length)
        self.n_fft = n_fft
        self.hop_length = hop_length

        # ---- Encoder (4 blocks) ----------------------------------------
        # Input: (B, 2, F, T)  — F = n_fft//2 + 1 = 257
        ch = [2, base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        # ch = [2, 32, 64, 128, 256]

        self.enc = nn.ModuleList([
            ConvBlock(ch[i], ch[i + 1]) for i in range(len(ch) - 1)
        ])

        # ---- Bottleneck (GRU) -------------------------------------------
        self.bottleneck = GRUBottleneck(
            channels=ch[-1],
            hidden=gru_hidden,
            num_layers=2,
        )

        # ---- Decoder (4 blocks) ----------------------------------------
        # Each block receives (current + skip) channels on input
        # dec[0]: in = ch[4] + ch[4] = 512, out = ch[3] = 128
        # dec[1]: in = ch[3] + ch[3] = 256, out = ch[2] = 64
        # dec[2]: in = ch[2] + ch[2] = 128, out = ch[1] = 32
        # dec[3]: in = ch[1] + ch[1] =  64, out = ch[0] = 2
        dec_in  = [ch[4] + ch[4], ch[3] + ch[3], ch[2] + ch[2], ch[1] + ch[1]]
        dec_out = [ch[3],         ch[2],          ch[1],         ch[0]]

        self.dec = nn.ModuleList([
            TransposeBlock(dec_in[i], dec_out[i]) for i in range(len(dec_out))
        ])

        # ---- Mask head --------------------------------------------------
        # 1×1 conv on the 2-channel decoder output → 2-channel mask
        self.mask_head = nn.Conv2d(2, 2, kernel_size=1)

        assert mask_act in ("tanh", "sigmoid"), \
            "mask_act must be 'tanh' or 'sigmoid'"
        self.mask_act = nn.Tanh() if mask_act == "tanh" else nn.Sigmoid()

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : (B, 1, L)  noisy mono waveform

        Returns
        -------
        enhanced : (B, 1, L)  enhanced waveform, same length as input
        """
        L = waveform.shape[-1]

        # 1. STFT  →  (B, 2, F, T)
        noisy_spec = self.stft(waveform)

        # 2. Encoder  → collect skips
        x = noisy_spec
        skips: List[torch.Tensor] = []
        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)

        # 3. Bottleneck
        x = self.bottleneck(x)

        # 4. Decoder  → skip connections in reverse
        for i, dec_block in enumerate(self.dec):
            skip = skips[-(i + 1)]
            # Align spatial dimensions before concatenation (T may differ by 1)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = dec_block(x, skip)

        # 5. Mask
        mask = self.mask_act(self.mask_head(x))

        # 6. Apply mask to noisy spectrogram (element-wise)
        #    Align F, T dims in case of rounding from stride/deconv
        if mask.shape != noisy_spec.shape:
            mask = F.interpolate(mask, size=noisy_spec.shape[2:], mode="nearest")

        enhanced_spec = mask * noisy_spec

        # 7. iSTFT  →  waveform
        enhanced = self.stft.inverse(enhanced_spec, length=L)

        return enhanced

    # ------------------------------------------------------------------
    def enhance_spectrogram(
        self, noisy_spec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for inference when the STFT is pre-computed.

        Returns (enhanced_spec, mask) both of shape (B, 2, F, T).
        Useful for streaming inference where STFT chunks are pre-batched.
        """
        x = noisy_spec
        skips: List[torch.Tensor] = []
        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)

        x = self.bottleneck(x)

        for i, dec_block in enumerate(self.dec):
            skip = skips[-(i + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = dec_block(x, skip)

        mask = self.mask_act(self.mask_head(x))
        if mask.shape != noisy_spec.shape:
            mask = F.interpolate(mask, size=noisy_spec.shape[2:], mode="nearest")

        return mask * noisy_spec, mask


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    base_ch: int = 32,
    gru_hidden: int = 64,
    mask_act: str = "tanh",
    device: str | None = None,
) -> AudioUNet:
    """Factory function that builds and optionally moves the model to a device.

    Parameters
    ----------
    device : 'cuda', 'cpu', or None (auto-detect).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AudioUNet(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        base_ch=base_ch,
        gru_hidden=gru_hidden,
        mask_act=mask_act,
    ).to(device)

    n_params = count_parameters(model)
    print(
        f"[AudioUNet] Parameters: {n_params:,}  "
        f"({n_params / 1e6:.2f}M)  |  device={device}"
    )
    assert n_params < 5_000_000, (
        f"Model has {n_params:,} params — exceeds 5M edge budget. "
        "Reduce base_ch or gru_hidden."
    )
    return model


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = build_model(device=device)

    # Simulate a 4-second batch of 2 clips at 16 kHz
    batch = torch.randn(2, 1, 64_000).to(device)

    model.eval()
    with torch.no_grad():
        out = model(batch)

    print(f"Input  shape : {batch.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == batch.shape, "Shape mismatch — check iSTFT length!"
    print("\nSmoke-test passed ✓")

    # Latency estimate (single clip, no batching)
    import time
    single = torch.randn(1, 1, 1024).to(device)   # 64 ms chunk @ 16 kHz
    model.eval()
    # Warm-up
    for _ in range(10):
        _ = model(single)
    torch.cuda.synchronize() if device == "cuda" else None

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model(single)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = (time.perf_counter() - t0) / N * 1000

    print(f"Avg inference latency (1024-sample chunk): {elapsed:.2f} ms")
