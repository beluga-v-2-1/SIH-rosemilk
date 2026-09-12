"""
train.py
--------
Training loop for the AudioUNet speech enhancement / ANC model.

Features
--------
- SI-SNR loss  +  L1 spectral loss (combined)
- Validation each epoch: STOI and PESQ via torchmetrics
- Checkpoint saved whenever validation STOI improves
- Gradient clipping, learning-rate scheduling (ReduceLROnPlateau)
- TensorBoard logging (optional, falls back gracefully)
- Configurable via a single TrainConfig dataclass — no argparse juggling

Quick-start
-----------
1.  Prepare data:
        data/
          clean/   *.wav   (LibriSpeech / VCTK)
          noise/   *.wav   (ESC-50 / FreeSound gunshots, helicopters, sirens)

2.  Run:
        python train.py

    Or with custom paths:
        python train.py --clean_dir /path/to/clean --noise_dir /path/to/noise

Usage as a library
------------------
>>> from train import TrainConfig, Trainer
>>> cfg = TrainConfig(clean_dir="data/clean", noise_dir="data/noise", epochs=50)
>>> trainer = Trainer(cfg)
>>> trainer.fit()
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project modules
from dataset import build_datasets
from model import AudioUNet, build_model, count_parameters

# Metrics
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    SignalNoiseRatio,
)

# PESQ requires a compiled C extension that may not be available on all platforms.
# We fall back to NISQA (Non-Intrusive Speech Quality Assessment via ONNX) which
# is always available through torchmetrics[audio] + onnxruntime.
try:
    from torchmetrics.audio import PerceptualEvaluationSpeechQuality as _PESQ
    _PESQ_METRIC = _PESQ
    _PESQ_MODE   = "pesq"
    _PESQ_KWARGS: dict = {"fs": None, "mode": "wb"}   # filled at runtime
except ImportError:
    _PESQ_METRIC = None
    _PESQ_MODE   = "nisqa"

try:
    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment as _NISQA
    _NISQA_AVAILABLE = True
except ImportError:
    _NISQA_AVAILABLE = False

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ---- Data ---------------------------------------------------------------
    clean_dir: str = "data/clean"
    noise_dir: str = "data/noise"
    sample_rate: int = 16_000
    clip_duration: float = 4.0          # seconds per training clip
    snr_range: Tuple[float, float] = (-5.0, 15.0)
    val_split: float = 0.1              # fraction of clean files for validation
    num_workers: int = 4

    # ---- Model --------------------------------------------------------------
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    base_ch: int = 32
    gru_hidden: int = 64
    mask_act: str = "tanh"

    # ---- Training -----------------------------------------------------------
    epochs: int = 100
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0              # max grad norm
    si_snr_weight: float = 0.8          # weight on SI-SNR loss
    l1_weight: float = 0.2              # weight on L1 spectral loss

    # ---- Scheduler ----------------------------------------------------------
    lr_patience: int = 5                # epochs without improvement before LR drop
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    # ---- Checkpointing ------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None        # path to checkpoint to resume from

    # ---- Logging ------------------------------------------------------------
    log_dir: str = "runs"
    log_interval: int = 50              # batches between train loss prints
    val_max_samples: int = 32           # clips to evaluate PESQ/STOI on (slow)

    # ---- Device -------------------------------------------------------------
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def si_snr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-Invariant Signal-to-Noise Ratio loss.

    Both tensors are shape (B, 1, L) or (B, L).
    Returns a scalar loss (negative SI-SNR, so minimising = maximising SI-SNR).

    Formula
    -------
    s_target = (<ŝ, s> / ||s||²) · s
    e_noise  = ŝ - s_target
    SI-SNR   = 10 log₁₀( ||s_target||² / ||e_noise||² )
    """
    est    = est.flatten(1)    # (B, L)
    target = target.flatten(1)

    # Zero-mean
    est    = est    - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # s_target
    dot   = (est * target).sum(dim=-1, keepdim=True)
    norm2 = (target * target).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    s_tgt = dot / norm2 * target

    # Noise
    e_noise = est - s_tgt

    # SI-SNR per sample
    si_snr = 10 * torch.log10(
        (s_tgt * s_tgt).sum(dim=-1).clamp(min=1e-8) /
        (e_noise * e_noise).sum(dim=-1).clamp(min=1e-8)
    )

    return -si_snr.mean()   # negate: lower loss = better SNR


def l1_spectral_loss(
    est_wav: torch.Tensor,
    target_wav: torch.Tensor,
    stft_module: nn.Module,
) -> torch.Tensor:
    """L1 loss in the magnitude spectrogram domain.

    Penalises spectral differences that time-domain SI-SNR may miss
    (e.g., harmonic distortions, phase errors).
    """
    est_spec    = stft_module(est_wav)     # (B, 2, F, T)
    target_spec = stft_module(target_wav)

    # Magnitude
    est_mag    = torch.sqrt(est_spec[:, 0]**2    + est_spec[:, 1]**2    + 1e-8)
    target_mag = torch.sqrt(target_spec[:, 0]**2 + target_spec[:, 1]**2 + 1e-8)

    return F.l1_loss(est_mag, target_mag)


def combined_loss(
    est: torch.Tensor,
    target: torch.Tensor,
    stft_module: nn.Module,
    si_snr_w: float = 0.8,
    l1_w: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined SI-SNR + L1 spectral loss.

    Returns
    -------
    (total_loss, si_snr_loss_val, l1_loss_val)
    """
    loss_si  = si_snr_loss(est, target)
    loss_l1  = l1_spectral_loss(est, target, stft_module)
    total    = si_snr_w * loss_si + l1_w * loss_l1
    return total, loss_si, loss_l1


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

class MetricsEvaluator:
    """Wraps torchmetrics STOI, PESQ/NISQA, and SNR for validation.

    PESQ requires the compiled `pesq` C extension.  When unavailable (e.g.
    Python 3.12 on Windows without MSVC), NISQA (ONNX-based) is used instead
    and its score is reported under the 'pesq' key for consistency.

    All metrics run on CPU to avoid device management complexity with
    torchmetrics internals.
    """

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.stoi = ShortTimeObjectiveIntelligibility(
            fs=sample_rate, extended=False
        )
        self.snr = SignalNoiseRatio()

        # Quality metric: PESQ preferred, NISQA fallback
        self._quality_metric = None
        self._quality_label  = "pesq"

        if _PESQ_METRIC is not None:
            try:
                self._quality_metric = _PESQ_METRIC(fs=sample_rate, mode="wb")
                self._quality_label  = "pesq"
            except Exception:
                pass

        if self._quality_metric is None and _NISQA_AVAILABLE:
            try:
                self._quality_metric = _NISQA(fs=sample_rate)
                self._quality_label  = "nisqa"
                print("[MetricsEvaluator] PESQ unavailable — using NISQA (ONNX) instead.")
            except Exception:
                pass

        if self._quality_metric is None:
            print("[MetricsEvaluator] Neither PESQ nor NISQA available — quality metric skipped.")

    @torch.no_grad()
    def compute(
        self,
        est_wav: torch.Tensor,
        clean_wav: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute STOI, PESQ/NISQA, SNR for a batch.

        Parameters
        ----------
        est_wav   : (B, 1, L)  enhanced waveforms
        clean_wav : (B, 1, L)  reference clean waveforms

        Returns
        -------
        dict with keys 'stoi', 'pesq' (or 'nisqa'), 'snr'
        """
        est   = est_wav.squeeze(1).cpu().float()    # (B, L)
        clean = clean_wav.squeeze(1).cpu().float()  # (B, L)

        results: Dict[str, float] = {}

        # STOI
        try:
            self.stoi.reset()
            self.stoi.update(est, clean)
            results["stoi"] = float(self.stoi.compute().item())
        except Exception as e:
            results["stoi"] = float("nan")
            print(f"  [STOI error] {e}")

        # Quality metric (PESQ or NISQA)
        if self._quality_metric is not None:
            try:
                if self._quality_label == "nisqa":
                    # NISQA processes one clip at a time; average across batch
                    scores = []
                    for i in range(est.shape[0]):
                        try:
                            self._quality_metric.reset()
                            self._quality_metric.update(est[i].unsqueeze(0))
                            scores.append(float(self._quality_metric.compute().item()))
                        except Exception:
                            pass
                    results["pesq"] = float(sum(scores) / len(scores)) if scores else float("nan")
                else:
                    self._quality_metric.reset()
                    self._quality_metric.update(est, clean)
                    results["pesq"] = float(self._quality_metric.compute().item())
            except Exception as e:
                results["pesq"] = float("nan")
                print(f"  [{self._quality_label.upper()} error] {e}")
        else:
            results["pesq"] = float("nan")

        # SNR
        try:
            self.snr.reset()
            self.snr.update(est, clean)
            results["snr"] = float(self.snr.compute().item())
        except Exception as e:
            results["snr"] = float("nan")
            print(f"  [SNR error] {e}")

        return results


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end training manager for AudioUNet.

    Parameters
    ----------
    cfg : TrainConfig dataclass with all hyperparameters.
    """

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self._set_seed(cfg.seed)
        self.device = torch.device(cfg.device)

        # ---- Model ----------------------------------------------------------
        self.model = build_model(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            base_ch=cfg.base_ch,
            gru_hidden=cfg.gru_hidden,
            mask_act=cfg.mask_act,
            device=cfg.device,
        )
        print(f"[Trainer] Trainable params: {count_parameters(self.model):,}")

        # ---- Data -----------------------------------------------------------
        print("[Trainer] Building datasets …")
        self.train_ds, self.val_ds = build_datasets(
            clean_dir=cfg.clean_dir,
            noise_dir=cfg.noise_dir,
            val_split=cfg.val_split,
            sample_rate=cfg.sample_rate,
            clip_duration=cfg.clip_duration,
            snr_range=cfg.snr_range,
            seed=cfg.seed,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
            drop_last=True,
            persistent_workers=(cfg.num_workers > 0),
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
            drop_last=False,
            persistent_workers=(cfg.num_workers > 0),
        )

        # ---- Optimiser & Scheduler -----------------------------------------
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="max",             # maximise STOI
            patience=cfg.lr_patience,
            factor=cfg.lr_factor,
            min_lr=cfg.min_lr,
        )

        # ---- Metrics --------------------------------------------------------
        self.evaluator = MetricsEvaluator(sample_rate=cfg.sample_rate)

        # ---- Checkpointing --------------------------------------------------
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_stoi: float = -1.0
        self.start_epoch: int = 0

        # ---- TensorBoard ----------------------------------------------------
        self.writer = None
        if _TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=cfg.log_dir)

        # ---- Resume ---------------------------------------------------------
        if cfg.resume:
            self._load_checkpoint(cfg.resume)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = si_snr_total = l1_total = 0.0
        n_batches = len(self.train_loader)
        t0 = time.time()

        for batch_idx, (noisy, clean) in enumerate(self.train_loader):
            noisy = noisy.to(self.device, non_blocking=True)   # (B, 1, L)
            clean = clean.to(self.device, non_blocking=True)

            self.optimiser.zero_grad(set_to_none=True)

            # Forward
            enhanced = self.model(noisy)

            # Loss
            loss, loss_si, loss_l1 = combined_loss(
                enhanced, clean,
                stft_module=self.model.stft,
                si_snr_w=self.cfg.si_snr_weight,
                l1_w=self.cfg.l1_weight,
            )

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimiser.step()

            # Accumulate
            total_loss   += loss.item()
            si_snr_total += loss_si.item()
            l1_total     += loss_l1.item()

            # Logging
            if (batch_idx + 1) % self.cfg.log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:3d} | "
                    f"Batch {batch_idx+1:4d}/{n_batches} | "
                    f"Loss {avg:.4f} | "
                    f"SI-SNR {si_snr_total/(batch_idx+1):.4f} | "
                    f"L1 {l1_total/(batch_idx+1):.4f} | "
                    f"{elapsed:.1f}s"
                )

        metrics = {
            "train/loss":    total_loss   / n_batches,
            "train/si_snr":  si_snr_total / n_batches,
            "train/l1":      l1_total     / n_batches,
        }
        return metrics

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        val_loss = val_si_snr = val_l1 = 0.0
        n_batches = len(self.val_loader)

        # Collect waveforms for PESQ/STOI (capped to val_max_samples)
        all_est:   list[torch.Tensor] = []
        all_clean: list[torch.Tensor] = []
        collected = 0

        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            enhanced = self.model(noisy)

            loss, loss_si, loss_l1 = combined_loss(
                enhanced, clean,
                stft_module=self.model.stft,
                si_snr_w=self.cfg.si_snr_weight,
                l1_w=self.cfg.l1_weight,
            )
            val_loss   += loss.item()
            val_si_snr += loss_si.item()
            val_l1     += loss_l1.item()

            # Collect for perceptual metrics
            remaining = self.cfg.val_max_samples - collected
            if remaining > 0:
                n = min(enhanced.shape[0], remaining)
                all_est.append(enhanced[:n].cpu())
                all_clean.append(clean[:n].cpu())
                collected += n

        # Compute STOI / PESQ / SNR
        est_cat   = torch.cat(all_est,   dim=0)
        clean_cat = torch.cat(all_clean, dim=0)
        perceptual = self.evaluator.compute(est_cat, clean_cat)

        metrics = {
            "val/loss":   val_loss   / n_batches,
            "val/si_snr": val_si_snr / n_batches,
            "val/l1":     val_l1     / n_batches,
            "val/stoi":   perceptual.get("stoi", float("nan")),
            "val/pesq":   perceptual.get("pesq", float("nan")),
            "val/snr":    perceptual.get("snr",  float("nan")),
        }
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], tag: str = "best") -> None:
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch":      epoch,
                "model":      self.model.state_dict(),
                "optimiser":  self.optimiser.state_dict(),
                "scheduler":  self.scheduler.state_dict(),
                "best_stoi":  self.best_stoi,
                "metrics":    metrics,
                "config":     self.cfg.__dict__,
            },
            path,
        )
        print(f"  [✓] Checkpoint saved → {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_stoi  = ckpt.get("best_stoi", -1.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        print(
            f"[Trainer] Resumed from {path} "
            f"(epoch {self.start_epoch}, best STOI={self.best_stoi:.4f})"
        )

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        cfg = self.cfg
        print(
            f"\n{'='*60}\n"
            f"  AudioUNet Training — {cfg.epochs} epochs\n"
            f"  device={cfg.device}  bs={cfg.batch_size}  lr={cfg.lr}\n"
            f"  train={len(self.train_ds)} | val={len(self.val_ds)}\n"
            f"{'='*60}\n"
        )

        for epoch in range(self.start_epoch, cfg.epochs):
            epoch_start = time.time()

            # --- Train -------------------------------------------------------
            train_metrics = self._train_epoch(epoch)

            # --- Validate ----------------------------------------------------
            val_metrics = self._val_epoch(epoch)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimiser.param_groups[0]["lr"]

            # --- Print summary -----------------------------------------------
            stoi = val_metrics["val/stoi"]
            pesq = val_metrics["val/pesq"]
            snr  = val_metrics["val/snr"]
            improved = stoi > self.best_stoi

            print(
                f"\nEpoch {epoch:3d}/{cfg.epochs-1} ({epoch_time:.1f}s) | "
                f"lr={current_lr:.2e}\n"
                f"  Train → loss={train_metrics['train/loss']:.4f}  "
                f"SI-SNR={train_metrics['train/si_snr']:.4f}\n"
                f"  Val   → loss={val_metrics['val/loss']:.4f}  "
                f"STOI={stoi:.4f}  PESQ={pesq:.4f}  SNR={snr:.2f} dB"
                f"{'  ← best ✓' if improved else ''}\n"
            )

            # --- TensorBoard -------------------------------------------------
            if self.writer:
                all_metrics = {**train_metrics, **val_metrics}
                for k, v in all_metrics.items():
                    if not (v != v):  # skip NaN
                        self.writer.add_scalar(k, v, epoch)
                self.writer.add_scalar("lr", current_lr, epoch)

            # --- Checkpoint on best STOI ------------------------------------
            if improved:
                self.best_stoi = stoi
                self._save_checkpoint(epoch, val_metrics, tag="best")

            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, tag=f"epoch_{epoch:04d}")

            # --- LR scheduler step ------------------------------------------
            self.scheduler.step(stoi if not (stoi != stoi) else 0.0)

        # --- Final save ------------------------------------------------------
        self._save_checkpoint(cfg.epochs - 1, val_metrics, tag="final")

        if self.writer:
            self.writer.close()

        print(
            f"\n{'='*60}\n"
            f"  Training complete.\n"
            f"  Best STOI : {self.best_stoi:.4f}\n"
            f"  Target    : STOI > 0.85 | PESQ > 2.5 | SNR > 15 dB\n"
            f"  Checkpoint: {self.ckpt_dir / 'checkpoint_best.pt'}\n"
            f"{'='*60}\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AudioUNet ANC model")
    p.add_argument("--clean_dir",     default="data/clean",   help="Clean speech directory")
    p.add_argument("--noise_dir",     default="data/noise",   help="Noise directory")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--base_ch",       type=int,   default=32)
    p.add_argument("--gru_hidden",    type=int,   default=64)
    p.add_argument("--clip_duration", type=float, default=4.0)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--checkpoint_dir",default="checkpoints")
    p.add_argument("--resume",        default=None, help="Path to checkpoint to resume")
    p.add_argument("--device",        default=None, help="cuda or cpu (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_ch=args.base_ch,
        gru_hidden=args.gru_hidden,
        clip_duration=args.clip_duration,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    trainer = Trainer(cfg)
    trainer.fit()
