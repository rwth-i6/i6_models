__all__ = ["LogMelFeatureExtractionV1", "LogMelFeatureExtractionV1Config"]

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal
from enum import Enum

from librosa import filters
import torch
from torch import nn
import numpy as np
from numpy.typing import DTypeLike

from i6_models.config import ModelConfiguration


class SpectrumType(Enum):
    STFT = 1
    RFFTN = 2


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
        periodic: whether the window is assumed to be periodic
        htk: whether use HTK formula instead of Slaney
        norm: how to normalize the filters, cf. https://librosa.org/doc/main/generated/librosa.filters.mel.html
        spectrum_type: apply torch.stft on raw audio input (default) or torch.fft.rfftn on windowed audio to make features compatible to RASR's
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
    periodic: bool = True
    htk: bool = False
    norm: Optional[Union[Literal["slaney"], float]] = "slaney"
    dtype: DTypeLike = np.float32
    spectrum_type: SpectrumType = SpectrumType.STFT

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
        self.spectrum_type = cfg.spectrum_type

        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=cfg.n_fft,
                    n_mels=cfg.num_filters,
                    fmin=cfg.f_min,
                    fmax=cfg.f_max,
                    htk=cfg.htk,
                    norm=cfg.norm,
                    dtype=cfg.dtype,
                ),
            ),
        )
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=cfg.periodic))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """
        if self.spectrum_type == SpectrumType.STFT:
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
        elif self.spectrum_type == SpectrumType.RFFTN:
            windowed = raw_audio.unfold(1, size=self.win_length, step=self.hop_length)  # [B, T', W=win_length]
            smoothed = windowed * self.window.unsqueeze(0)  # [B, T', W]

            # Compute power spectrum using torch.fft.rfftn
            power_spectrum = torch.abs(torch.fft.rfftn(smoothed, s=self.n_fft)) ** 2  # [B, T', F=n_fft//2+1]
            power_spectrum = power_spectrum.transpose(1, 2)  # [B, F, T']
        else:
            raise ValueError(f"Invalid spectrum type {self.spectrum_type!r}.")

        if len(power_spectrum.size()) == 2:
            # For some reason torch.stft removes the batch axis for batch sizes of 1, so we need to add it again
            power_spectrum = torch.unsqueeze(power_spectrum, 0)  # [B, F, T']
        melspec = torch.einsum("...ft,mf->...mt", power_spectrum, self.mel_basis)  # [B, F'=num_filters, T']
        log_melspec = torch.log10(torch.clamp(melspec, min=self.min_amp))
        feature_data = torch.transpose(log_melspec, 1, 2)  # [B, T', F']

        if self.spectrum_type == SpectrumType.STFT:
            if self.center:
                length = (length // self.hop_length) + 1
            else:
                length = ((length - self.n_fft) // self.hop_length) + 1
        elif self.spectrum_type == SpectrumType.RFFTN:
            length = ((length - self.win_length) // self.hop_length) + 1
        else:
            raise ValueError(f"Invalid spectrum type {self.spectrum_type!r}.")
        return feature_data, length.int()
