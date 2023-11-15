__all__ = ["LogMelFeatureExtractionV1", "LogMelFeatureExtractionV1Config"]

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

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
        periodic: whether the window is assumed to be periodic
        mel_options: extra options for mel filters
        rasr_compatible: apply FFT to make features compatible to RASR's, otherwise (default) apply STFT
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
    mel_options: Optional[Dict[str, Any]] = None
    rasr_compatible: bool = False

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
        self.mel_options = cfg.mel_options or {}
        self.rasr_compatible = cfg.rasr_compatible

        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=cfg.n_fft,
                    n_mels=cfg.num_filters,
                    fmin=cfg.f_min,
                    fmax=cfg.f_max,
                    **self.mel_options,
                )
            ),
        )
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=cfg.periodic))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """
        if self.rasr_compatible:
            windowed = raw_audio.unfold(1, size=self.win_length, step=self.hop_length)  # [B, T', W=win_length]
            smoothed = windowed * self.window.unsqueeze(0)  # [B, T', W]

            # Compute power spectrum using torch.fft.rfftn
            power_spectrum = torch.abs(torch.fft.rfftn(smoothed, s=self.n_fft)) ** 2  # [B, T', F=n_fft//2+1]
            power_spectrum = power_spectrum.transpose(1, 2)  # [B, F, T']
        else:
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
            power_spectrum = torch.unsqueeze(power_spectrum, 0)  # [B, F, T']
        melspec = torch.einsum("...ft,mf->...mt", power_spectrum, self.mel_basis)  # [B, F'=num_filters, T']
        log_melspec = torch.log10(torch.clamp(melspec, min=self.min_amp))
        feature_data = torch.transpose(log_melspec, 1, 2)  # [B, T', F']

        if self.center and not self.rasr_compatible:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.win_length) // self.hop_length) + 1

        return feature_data, length.int()
