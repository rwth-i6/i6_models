__all__ = [
    "LogMelFeatureExtractionV1",
    "LogMelFeatureExtractionV1Config",
    "RasrCompatibleLogMelFeatureExtractionV1",
    "RasrCompatibleLogMelFeatureExtractionV1Config",
]

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from librosa import filters
import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class LogMelFeatureExtractionV1Config(ModelConfiguration):
    """
    Attributes:
        sample_rate: audio sample rate in Hz
        win_size: window size in seconds
        hop_size: window shift in seconds
        f_min: minimum filter frequency in Hz
        f_max: maximum filter frequency in Hz
        min_amp: minimum amplitude for safe log
        num_filters: number of mel windows
        center: centered STFT with automatic padding
    """

    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool
    n_fft: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.f_max <= self.sample_rate // 2, "f_max can not be larger than half of the sample rate"
        assert self.f_min >= 0 and self.f_max > 0 and self.sample_rate > 0, "frequencies need to be positive"
        assert self.win_size > 0 and self.hop_size > 0, "window settings need to be positive"
        assert self.num_filters > 0, "number of filters needs to be positive"
        assert self.hop_size <= self.win_size, "using a larger hop size than window size does not make sense"
        if self.n_fft is None:
            # if n_fft is not given, set n_fft to the window size (in samples)
            self.n_fft = int(self.win_size * self.sample_rate)
        else:
            assert self.n_fft >= self.win_size * self.sample_rate, "n_fft cannot to be smaller than the window size"


class LogMelFeatureExtractionV1(nn.Module):
    """
    Librosa-compatible log-mel feature extraction using log10. Does not use torchaudio.

    Using it wrapped with torch.no_grad() is recommended if no gradient is needed
    """

    def __init__(self, cfg: LogMelFeatureExtractionV1Config):
        super().__init__()
        self.center = cfg.center
        self.hop_length = int(cfg.hop_size * cfg.sample_rate)
        self.min_amp = cfg.min_amp
        self.n_fft = cfg.n_fft
        self.win_length = int(cfg.win_size * cfg.sample_rate)

        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=cfg.n_fft,
                    n_mels=cfg.num_filters,
                    fmin=cfg.f_min,
                    fmax=cfg.f_max,
                )
            ),
        )
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """
        power_spectrum = (
            torch.abs(
                torch.stft(
                    raw_audio,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode="constant",
                    return_complex=True,
                )
            )
            ** 2
        )
        if len(power_spectrum.size()) == 2:
            # For some reason torch.stft removes the batch axis for batch sizes of 1, so we need to add it again
            power_spectrum = torch.unsqueeze(power_spectrum, 0)
        melspec = torch.einsum("...ft,mf->...mt", power_spectrum, self.mel_basis)
        log_melspec = torch.log10(torch.clamp(melspec, min=self.min_amp))
        feature_data = torch.transpose(log_melspec, 1, 2)

        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1

        return feature_data, length.int()


@dataclass
class RasrCompatibleLogMelFeatureExtractionV1Config(ModelConfiguration):
    """
    Attributes:
        sample_rate: audio sample rate in Hz
        win_size: window size in seconds
        hop_size: window shift in seconds
        min_amp: minimum amplitude for safe log
        num_filters: number of mel windows
        alpha: preemphasis weight
    """

    sample_rate: int
    win_size: float
    hop_size: float
    min_amp: float
    num_filters: int
    alpha: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.win_size > 0 and self.hop_size > 0, "window settings need to be positive"
        assert self.num_filters > 0, "number of filters needs to be positive"
        assert self.hop_size <= self.win_size, "using a larger hop size than window size does not make sense"


class RasrCompatibleLogMelFeatureExtractionV1(nn.Module):
    """
    Rasr-compatible log-mel feature extraction using log10. Does not use torchaudio.
    """

    def __init__(self, cfg: RasrCompatibleLogMelFeatureExtractionV1Config):
        super().__init__()

        self.sample_rate = int(cfg.sample_rate)
        self.hop_length = int(cfg.hop_size * cfg.sample_rate)
        self.min_amp = cfg.min_amp
        self.win_length = int(cfg.win_size * cfg.sample_rate)
        self.n_fft = 2 ** math.ceil(
            math.log2(self.win_length)
        )  # smallest power if two which is greater than or equal to win_length
        self.alpha = cfg.alpha

        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=self.n_fft,
                    n_mels=cfg.num_filters,
                    fmin=0,
                    fmax=cfg.sample_rate // 2,
                    htk=True,
                    norm=None,
                ),
            ),
        )
        self.register_buffer(
            "window", torch.hann_window(self.win_length, periodic=False, dtype=torch.float64).to(torch.float32)
        )

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length: in samples [B]
        :return features as [B,T,F] and length in frames [B]
        """
        assert raw_audio.shape[1] > 0  # also same for length
        res_size = max(raw_audio.shape[1] - self.win_length + self.hop_length - 1, 0) // self.hop_length + 1
        res_length = (
            torch.maximum(length - self.win_length + self.hop_length - 1, torch.zeros_like(length)) // self.hop_length
            + 1
        )

        # preemphasize
        preemphasized = raw_audio.clone()
        preemphasized[..., 1:] -= self.alpha * preemphasized[..., :-1]
        preemphasized[..., 0] = 0.0

        # zero pad for the last frame of each sequence in the batch
        last_win_size = length - (res_length - 1) * self.hop_length  # [B]
        last_pad = self.win_length - last_win_size  # [B]

        # zero pad for the whole batch
        last_pad_batch = self.win_length - (preemphasized.shape[1] - (res_size - 1) * self.hop_length)
        padded = torch.nn.functional.pad(preemphasized, (0, last_pad_batch))

        windowed = padded.unfold(1, size=self.win_length, step=self.hop_length)  # [B, T', W=self.win_length]

        smoothed = windowed * self.window[None, None, :]  # [B, T', W]

        # The last window might be shorter. Will use a shorter Hanning window then. Need to fix that.
        for i, (last_w_size, last_p, res_l) in enumerate(zip(last_win_size, last_pad, res_length)):
            last_win = torch.hann_window(last_w_size, periodic=False, dtype=torch.float64).to(
                self.window.device, torch.float32
            )
            last_win = torch.nn.functional.pad(last_win, (0, last_p))  # [W]
            smoothed[i, res_l - 1] = windowed[i, res_l - 1] * last_win[None, :]

        # compute amplitude spectrum using torch.fft.rfftn with Rasr specific scaling
        fft = torch.fft.rfftn(smoothed, s=self.n_fft) / self.sample_rate  # [B, T', F=n_fft//2+1]
        amplitude_spectrum = torch.abs(fft)  # [B, T', F=n_fft//2+1]

        melspec = torch.einsum("...tf,mf->...tm", amplitude_spectrum, self.mel_basis)  # [B, T', F'=num_filters]
        log_melspec = torch.log10(melspec + self.min_amp)

        return log_melspec, res_length
