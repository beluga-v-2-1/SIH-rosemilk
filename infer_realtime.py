"""
infer_realtime.py
-----------------
Real-time Active Noise Cancellation / Speech Enhancement using PyAudio.

Architecture
------------
                     ┌──────────────────────────────────────────────────────┐
  Microphone  ──►   │  PyAudio input stream                                │
  (raw PCM)         │       │                                               │
                    │       ▼                                               │
                    │  Circular overlap buffer  (hop_length samples/frame)  │
                    │       │                                               │
                    │       ▼                                               │
                    │  AudioUNet.enhance_spectrogram()  (GPU/CPU)           │
                    │       │                                               │
                    │       ▼                                               │
                    │  Overlap-Add reconstruction                           │
                    │       │                                               │
                    │       ▼                                               │
                    │  PyAudio output stream  ──►  Headphones / Speaker     │
                    └──────────────────────────────────────────────────────┘

Latency budget (default settings, 16 kHz)
------------------------------------------
  Frame size      : hop_length = 128 samples  →  8 ms per frame
  Look-ahead      : 0  (causal — no future frames used)
  Model latency   : ~3-4 ms on RTX 2050
  Total one-way   : ≈ 12 ms  (well within the 20 ms perceptual threshold)

Usage
-----
    # Basic (auto-detect mic + speaker, load best checkpoint)
    python infer_realtime.py --checkpoint checkpoints/checkpoint_best.pt

    # Choose specific devices
    python infer_realtime.py --checkpoint checkpoints/checkpoint_best.pt \
        --input_device 1 --output_device 2

    # List available audio devices
    python infer_realtime.py --list_devices

    # Save enhanced output to a WAV file instead of playing
    python infer_realtime.py --checkpoint checkpoints/checkpoint_best.pt \
        --save_output output/enhanced.wav --duration 10
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pyaudio

# ---------------------------------------------------------------------------
# Lazy model import (allows --list_devices without needing a checkpoint)
# ---------------------------------------------------------------------------
def _import_model():
    sys.path.insert(0, str(Path(__file__).parent))
    from model import AudioUNet, STFT
    return AudioUNet, STFT


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def list_audio_devices() -> None:
    """Print all available PyAudio input/output devices."""
    pa = pyaudio.PyAudio()
    print(f"\n{'─'*60}")
    print(f"  {'IDX':>3}  {'NAME':<36}  {'IN':>3}  {'OUT':>3}")
    print(f"{'─'*60}")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(
            f"  {i:>3}  {info['name'][:36]:<36}  "
            f"{int(info['maxInputChannels']):>3}  "
            f"{int(info['maxOutputChannels']):>3}"
        )
    print(f"{'─'*60}\n")
    pa.terminate()


def get_default_devices(pa: pyaudio.PyAudio):
    """Return (input_idx, output_idx) for default devices."""
    try:
        in_idx  = pa.get_default_input_device_info()["index"]
    except OSError:
        in_idx  = None
    try:
        out_idx = pa.get_default_output_device_info()["index"]
    except OSError:
        out_idx = None
    return in_idx, out_idx


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: str,
) -> "AudioUNet":
    """Load AudioUNet from a training checkpoint."""
    AudioUNet, _ = _import_model()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Re-build model from saved config if available, else use defaults
    cfg = ckpt.get("config", {})
    model = AudioUNet(
        n_fft      = cfg.get("n_fft",       512),
        hop_length = cfg.get("hop_length",   128),
        win_length = cfg.get("win_length",   512),
        base_ch    = cfg.get("base_ch",       32),
        gru_hidden = cfg.get("gru_hidden",    64),
        mask_act   = cfg.get("mask_act",   "tanh"),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    stoi    = metrics.get("val/stoi", float("nan"))
    print(
        f"[Model] Loaded checkpoint (epoch={epoch}, val_stoi={stoi:.4f})"
        if stoi == stoi else
        f"[Model] Loaded checkpoint (epoch={epoch})"
    )
    return model


# ---------------------------------------------------------------------------
# Overlap-Add buffer
# ---------------------------------------------------------------------------

class OverlapAddBuffer:
    """Manages the overlap-add reconstruction for STFT-based streaming.

    The STFT uses a Hann window with 75 % overlap (hop = win//4).
    For correct reconstruction we maintain an output accumulation buffer
    that is flushed hop_length samples per frame.

    Parameters
    ----------
    win_length : STFT window length (samples).
    hop_length : STFT hop length  (samples).
    """

    def __init__(self, win_length: int, hop_length: int) -> None:
        self.win_length = win_length
        self.hop_length = hop_length
        # Input ring buffer: holds win_length samples (one full STFT frame)
        self._in_buf  = np.zeros(win_length, dtype=np.float32)
        # Output accumulation buffer
        self._out_buf = np.zeros(win_length, dtype=np.float32)

    def push_samples(self, samples: np.ndarray) -> None:
        """Shift hop_length new samples into the input ring buffer."""
        assert len(samples) == self.hop_length
        self._in_buf = np.roll(self._in_buf, -self.hop_length)
        self._in_buf[-self.hop_length:] = samples

    def get_frame(self) -> np.ndarray:
        """Return the current win_length input frame as a float32 array."""
        return self._in_buf.copy()

    def add_output(self, frame: np.ndarray) -> np.ndarray:
        """Overlap-add a new enhanced frame; return hop_length output samples.

        Parameters
        ----------
        frame : enhanced waveform of length win_length.

        Returns
        -------
        hop_length samples ready for playback.
        """
        self._out_buf += frame
        # Pop the first hop_length samples
        out = self._out_buf[:self.hop_length].copy()
        # Shift accumulation buffer
        self._out_buf = np.roll(self._out_buf, -self.hop_length)
        self._out_buf[-self.hop_length:] = 0.0
        return out


# ---------------------------------------------------------------------------
# Real-time processor
# ---------------------------------------------------------------------------

class RealTimeEnhancer:
    """Coordinates mic → model → speaker in a low-latency loop.

    Design
    ------
    - One PyAudio *input*  callback pushes raw chunks into `_in_queue`.
    - One PyAudio *output* callback pops enhanced chunks from `_out_queue`.
    - A dedicated *processing thread* drains `_in_queue`, runs the model,
      and pushes results to `_out_queue`.

    This decouples audio I/O (hard real-time) from model inference (soft
    real-time), preventing buffer under-runs on the output side.

    Parameters
    ----------
    model           : Loaded AudioUNet in eval mode.
    device          : 'cuda' or 'cpu'.
    sample_rate     : Audio sample rate (must match training).
    hop_length      : Samples per processing frame (== STFT hop).
    win_length      : STFT window length.
    input_device    : PyAudio device index for microphone.
    output_device   : PyAudio device index for speaker/headphones.
    bypass          : If True, pass audio through without enhancement (A/B test).
    save_path       : Optional path to write enhanced audio as WAV.
    max_queue_size  : Maximum frames buffered before dropping (back-pressure).
    """

    FORMAT       = pyaudio.paFloat32
    CHANNELS     = 1
    BYTES_PER_SAMPLE = 4   # float32

    def __init__(
        self,
        model: "AudioUNet",
        device: str,
        sample_rate: int      = 16_000,
        hop_length: int       = 128,
        win_length: int       = 512,
        input_device:  Optional[int] = None,
        output_device: Optional[int] = None,
        bypass: bool          = False,
        save_path: Optional[str] = None,
        max_queue_size: int   = 64,
    ) -> None:
        self.model         = model
        self.device        = device
        self.sample_rate   = sample_rate
        self.hop_length    = hop_length
        self.win_length    = win_length
        self.input_device  = input_device
        self.output_device = output_device
        self.bypass        = bypass
        self.save_path     = save_path

        self._in_queue:  queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._out_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)

        self._ola   = OverlapAddBuffer(win_length, hop_length)
        self._pa    = pyaudio.PyAudio()
        self._running = threading.Event()

        # Stats
        self._frames_processed = 0
        self._total_model_ms   = 0.0
        self._drop_count       = 0

        # WAV writer
        self._wav_writer: Optional[wave.Wave_write] = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self._wav_writer = wave.open(save_path, "wb")
            self._wav_writer.setnchannels(1)
            self._wav_writer.setsampwidth(2)              # 16-bit PCM
            self._wav_writer.setframerate(sample_rate)

    # ------------------------------------------------------------------
    # PyAudio callbacks (called from audio thread — keep minimal)
    # ------------------------------------------------------------------

    def _input_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info,
        status_flags,
    ):
        samples = np.frombuffer(in_data, dtype=np.float32).copy()
        try:
            self._in_queue.put_nowait(samples)
        except queue.Full:
            self._drop_count += 1   # Overload: drop oldest
        return (None, pyaudio.paContinue)

    def _output_callback(
        self,
        in_data,
        frame_count: int,
        time_info,
        status_flags,
    ):
        try:
            out = self._out_queue.get_nowait()
        except queue.Empty:
            # Underrun: output silence
            out = np.zeros(self.hop_length, dtype=np.float32)
        return (out.tobytes(), pyaudio.paContinue)

    # ------------------------------------------------------------------
    # Processing thread
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _process_loop(self) -> None:
        """Drain input queue → model → output queue."""
        while self._running.is_set():
            try:
                samples = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.bypass:
                # Pass-through mode for A/B comparison
                self._out_queue.put(samples[:self.hop_length])
                continue

            # Push new samples into the overlap buffer
            # Ensure exactly hop_length samples
            chunk = samples[:self.hop_length]
            if len(chunk) < self.hop_length:
                chunk = np.pad(chunk, (0, self.hop_length - len(chunk)))
            self._ola.push_samples(chunk)

            # Build a win_length frame tensor
            frame_np = self._ola.get_frame()                    # (win_length,)
            frame_t  = torch.from_numpy(frame_np).float()
            frame_t  = frame_t.unsqueeze(0).unsqueeze(0)        # (1, 1, win_length)
            frame_t  = frame_t.to(self.device, non_blocking=True)

            # Model inference
            t0 = time.perf_counter()
            enhanced_t = self.model(frame_t)                    # (1, 1, win_length)
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000

            # Convert back to numpy
            enhanced_np = enhanced_t.squeeze().cpu().numpy()    # (win_length,)

            # Overlap-add → hop_length output samples
            out_samples = self._ola.add_output(enhanced_np)

            # Soft-clip to prevent speaker damage
            out_samples = np.clip(out_samples, -1.0, 1.0)

            # Write to output queue
            try:
                self._out_queue.put_nowait(out_samples)
            except queue.Full:
                pass   # Slow consumer: skip

            # Write to WAV file if requested (convert to int16)
            if self._wav_writer:
                pcm16 = (out_samples * 32767).astype(np.int16)
                self._wav_writer.writeframes(pcm16.tobytes())

            # Stats
            self._frames_processed += 1
            self._total_model_ms   += dt_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open audio streams and start processing thread."""
        self._running.set()

        # Processing thread
        self._proc_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="ANC-processor",
        )
        self._proc_thread.start()

        # Input stream
        self._in_stream = self._pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.hop_length,
            stream_callback=self._input_callback,
        )

        # Output stream
        self._out_stream = self._pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.output_device,
            frames_per_buffer=self.hop_length,
            stream_callback=self._output_callback,
        )

        self._in_stream.start_stream()
        self._out_stream.start_stream()

        print(
            f"\n[ANC] Streaming started "
            f"(sr={self.sample_rate} Hz, "
            f"frame={self.hop_length} samples = "
            f"{self.hop_length/self.sample_rate*1000:.1f} ms)\n"
            f"  Press Ctrl-C to stop.\n"
        )

    def stop(self) -> None:
        """Stop streams and clean up."""
        self._running.clear()
        self._proc_thread.join(timeout=2.0)

        self._in_stream.stop_stream()
        self._in_stream.close()
        self._out_stream.stop_stream()
        self._out_stream.close()
        self._pa.terminate()

        if self._wav_writer:
            self._wav_writer.close()
            print(f"[ANC] Saved enhanced audio → {self.save_path}")

        self._print_stats()

    def _print_stats(self) -> None:
        n = self._frames_processed
        if n == 0:
            return
        avg_ms   = self._total_model_ms / n
        duration = n * self.hop_length / self.sample_rate
        print(
            f"\n[ANC] Stats\n"
            f"  Frames processed    : {n}\n"
            f"  Audio processed     : {duration:.1f}s\n"
            f"  Avg model latency   : {avg_ms:.2f} ms / frame\n"
            f"  Dropped input frames: {self._drop_count}\n"
        )

    def run_blocking(self, duration: Optional[float] = None) -> None:
        """Start streaming and block until Ctrl-C or duration expires."""
        self.start()
        try:
            if duration:
                time.sleep(duration)
            else:
                while self._in_stream.is_active():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[ANC] Interrupted by user.")
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time ANC / Speech Enhancement — AudioUNet",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint_best.pt",
        help="Path to trained model checkpoint (default: checkpoints/checkpoint_best.pt)",
    )
    p.add_argument(
        "--list_devices", action="store_true",
        help="Print available audio devices and exit",
    )
    p.add_argument(
        "--input_device",  type=int, default=None,
        help="PyAudio input device index  (default: system default mic)",
    )
    p.add_argument(
        "--output_device", type=int, default=None,
        help="PyAudio output device index (default: system default speaker)",
    )
    p.add_argument(
        "--sample_rate", type=int, default=16_000,
        help="Sample rate in Hz — must match training (default: 16000)",
    )
    p.add_argument(
        "--hop_length", type=int, default=128,
        help="STFT hop / frame size in samples (default: 128 = 8 ms @ 16 kHz)",
    )
    p.add_argument(
        "--win_length", type=int, default=512,
        help="STFT window length in samples (default: 512)",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    p.add_argument(
        "--bypass", action="store_true",
        help="Pass audio through without enhancement (for A/B comparison)",
    )
    p.add_argument(
        "--save_output", default=None,
        help="Save enhanced audio to this WAV file instead of (or in addition to) playback",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Auto-stop after this many seconds (default: run until Ctrl-C)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- List devices and exit -------------------------------------------
    if args.list_devices:
        list_audio_devices()
        return

    # ---- Device selection ------------------------------------------------
    torch_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] torch device = {torch_device}")

    # ---- Load model ------------------------------------------------------
    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        print(
            f"[ERROR] Checkpoint not found: {ckpt_path}\n"
            f"  Train the model first:\n"
            f"    python train.py --clean_dir data/clean --noise_dir data/noise\n"
            f"  Or point to an existing checkpoint with --checkpoint <path>"
        )
        sys.exit(1)

    model = load_model(ckpt_path, torch_device)

    # ---- Print device info -----------------------------------------------
    pa = pyaudio.PyAudio()
    in_idx, out_idx = get_default_devices(pa)
    pa.terminate()

    in_dev  = args.input_device  if args.input_device  is not None else in_idx
    out_dev = args.output_device if args.output_device is not None else out_idx

    print(f"[Config] Input  device index : {in_dev}")
    print(f"[Config] Output device index : {out_dev}")
    print(f"[Config] Sample rate         : {args.sample_rate} Hz")
    print(f"[Config] Frame size          : {args.hop_length} samples "
          f"({args.hop_length/args.sample_rate*1000:.1f} ms)")
    if args.bypass:
        print("[Config] MODE: BYPASS (pass-through, no enhancement)")

    # ---- Run -------------------------------------------------------------
    enhancer = RealTimeEnhancer(
        model          = model,
        device         = torch_device,
        sample_rate    = args.sample_rate,
        hop_length     = args.hop_length,
        win_length     = args.win_length,
        input_device   = in_dev,
        output_device  = out_dev,
        bypass         = args.bypass,
        save_path      = args.save_output,
    )
    enhancer.run_blocking(duration=args.duration)


if __name__ == "__main__":
    main()
