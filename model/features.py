"""48-kHz mono Log-Mel and MFCC features, shared by offline and streaming paths.

The specification is intentionally explicit to make an equivalent ESP32-S3
implementation possible. No librosa-specific implicit centering or padding.

One 6-second signal (288,000 samples) produces 299 frames:
    frame_length=1200 (25 ms), hop_length=960 (20 ms), n_fft=2048.

Spectral pipeline:
    periodic Hann -> right-zero-padded real FFT -> power / n_fft
    -> 40 unit-height HTK Mel triangles (60 to 20,000 Hz)
    -> natural logarithm of max(Mel energy, 1e-10)
    -> orthonormal DCT-II, coefficients 0..12 (MFCC).

All features are raw (not standardized). Training-only normalization is a
separate stage; never estimate its statistics on validation/test samples.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 48_000
    segment_seconds: int = 6
    frame_length: int = 1_200
    hop_length: int = 960
    n_fft: int = 2_048
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 60.0
    fmax: float = 20_000.0
    log_floor: float = 1e-10

    @property
    def samples_per_segment(self) -> int:
        return self.sample_rate * self.segment_seconds

    @property
    def frames_per_segment(self) -> int:
        return 1 + (self.samples_per_segment - self.frame_length) // self.hop_length

    def validate(self) -> None:
        if not (0 < self.frame_length <= self.n_fft):
            raise ValueError('frame_length must be in (0, n_fft]')
        if not (0 < self.hop_length <= self.frame_length):
            raise ValueError('hop_length must be in (0, frame_length]')
        if not (0 < self.fmin < self.fmax < self.sample_rate / 2):
            raise ValueError('Mel frequency bounds are invalid')
        if not (0 < self.n_mfcc <= self.n_mels):
            raise ValueError('n_mfcc must be in (0, n_mels]')
        if not (self.samples_per_segment >= self.frame_length):
            raise ValueError('segment shorter than analysis frame')

    def describe(self) -> dict:
        return {**asdict(self), 'frames_per_segment': self.frames_per_segment,
                'window': 'periodic_hann', 'fft_padding': 'right',
                'spectrum': 'abs(rfft)^2 / n_fft',
                'mel_scale': 'htk', 'mel_filter_normalization': 'unit_height',
                'log': 'natural_log', 'mfcc_dct': 'type_II_orthonormal',
                'mfcc_includes_c0': True, 'center_frames': False}


CONFIG = FeatureConfig()


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class AcousticFeatures:
    """Stateless extractor for full segments and individual analysis frames."""

    def __init__(self, config: FeatureConfig = CONFIG):
        config.validate()
        self.config = config
        # Periodic Hann, as used by typical FFT analysis; not np.hanning(n).
        t = np.arange(config.frame_length, dtype=np.float64)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * t / config.frame_length)).astype(np.float32)

        mel_points = np.linspace(
            _hz_to_mel(np.asarray(config.fmin)),
            _hz_to_mel(np.asarray(config.fmax)), config.n_mels + 2,
        )
        edges = _mel_to_hz(mel_points)
        hz_bins = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
        filters = np.empty((config.n_mels, hz_bins.size), dtype=np.float32)
        for k in range(config.n_mels):
            left, mid, right = edges[k:k+3]
            ascending = (hz_bins - left) / (mid - left)
            descending = (right - hz_bins) / (right - mid)
            filters[k] = np.maximum(0.0, np.minimum(ascending, descending))
        if np.any(filters.sum(axis=1) <= 0):
            raise ValueError('At least one Mel filter has no FFT bin support')
        self.mel_filters = filters

        k = np.arange(config.n_mfcc, dtype=np.float64)[:, None]
        m = np.arange(config.n_mels, dtype=np.float64)[None, :]
        dct = np.cos(np.pi * k * (m + 0.5) / config.n_mels)
        dct *= np.sqrt(2.0 / config.n_mels)
        dct[0] /= np.sqrt(2.0)
        self.dct = dct.T.astype(np.float32)  # [mel, mfcc]

    def _from_frames(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Frames: [N, 1200]. Returns [N, 40] Log-Mel and [N, 13] MFCC."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != self.config.frame_length:
            raise ValueError(f'Expected frames with shape [N,{self.config.frame_length}]')
        if not np.isfinite(frames).all():
            raise ValueError('Input audio contains NaN or infinity')
        spectrum = np.fft.rfft(frames * self.window, n=self.config.n_fft, axis=1)
        power = (np.abs(spectrum) ** 2 / self.config.n_fft).astype(np.float32)
        mel_energy = power @ self.mel_filters.T
        logmel = np.log(np.maximum(mel_energy, self.config.log_floor)).astype(np.float32)
        mfcc = (logmel @ self.dct).astype(np.float32)
        return logmel, mfcc

    def extract(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract from exactly one mono 6-second 48-kHz PCM float segment."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size != self.config.samples_per_segment:
            raise ValueError(f'Expected {self.config.samples_per_segment} mono samples')
        frames = np.lib.stride_tricks.sliding_window_view(
            audio, self.config.frame_length
        )[::self.config.hop_length]
        return self._from_frames(frames)


class StreamingFeatures:
    """Simulate a DMA consumer by accepting arbitrary successive PCM chunks.

    The stream is intentionally reset at each 6-second boundary, matching the
    dataset's non-overlapping 6-second labels. Each push returns newly available
    frames. Only the unconsumed audio tail is retained (<= frame_length samples).
    """

    def __init__(self, extractor: AcousticFeatures):
        self.extractor = extractor
        self._tail = np.empty(0, dtype=np.float32)
        self._received = 0
        self._frames = 0

    def push(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError('Streaming PCM chunks must be mono 1D arrays')
        if self._received + chunk.size > self.extractor.config.samples_per_segment:
            raise ValueError('Stream exceeded the 6-second segment boundary')
        self._received += chunk.size
        data = np.concatenate((self._tail, chunk))
        cfg = self.extractor.config
        if data.size < cfg.frame_length:
            self._tail = data
            return (np.empty((0, cfg.n_mels), dtype=np.float32),
                    np.empty((0, cfg.n_mfcc), dtype=np.float32))
        n_frames = 1 + (data.size - cfg.frame_length) // cfg.hop_length
        frames = np.lib.stride_tricks.sliding_window_view(data, cfg.frame_length)[
            :n_frames * cfg.hop_length:cfg.hop_length
        ]
        # Use an exactly sized frame batch even if future chunk sizes differ.
        frames = frames[:n_frames]
        logmel, mfcc = self.extractor._from_frames(frames)
        self._frames += n_frames
        self._tail = data[n_frames * cfg.hop_length:].copy()
        return logmel, mfcc

    def finish(self) -> None:
        """Confirm that exactly one 6-second segment was processed."""
        cfg = self.extractor.config
        if self._received != cfg.samples_per_segment:
            raise ValueError(f'Expected {cfg.samples_per_segment} samples; got {self._received}')
        if self._frames != cfg.frames_per_segment:
            raise ValueError(f'Expected {cfg.frames_per_segment} frames; got {self._frames}')
