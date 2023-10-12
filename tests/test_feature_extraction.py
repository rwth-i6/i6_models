import copy
import numpy as np
import torch

from librosa.feature import melspectrogram

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config


def test_logmel_librosa_compatibility():

    audio = np.asarray(np.random.random((50000)), dtype=np.float32)
    librosa_mel = melspectrogram(
        y=audio,
        sr=16000,
        n_fft=int(0.05 * 16000),
        hop_length=int(0.0125 * 16000),
        win_length=int(0.05 * 16000),
        fmin=60,
        fmax=7600,
        n_mels=80,
    )
    librosa_log_mel = np.log10(np.maximum(librosa_mel, 1e-10))

    fe_cfg = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.05,
        hop_size=0.0125,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,
    )
    fe = LogMelFeatureExtractionV1(cfg=fe_cfg)
    audio_tensor = torch.unsqueeze(torch.Tensor(audio), 0)  # [B, T]
    audio_length = torch.tensor([50000])  # [B]
    pytorch_log_mel, frame_length = fe(audio_tensor, audio_length)
    librosa_log_mel = torch.tensor(librosa_log_mel).transpose(0, 1)
    assert torch.allclose(librosa_log_mel, pytorch_log_mel, atol=1e-06)


def test_logmel_length():
    fe_center_cfg = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.05,
        hop_size=0.0125,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,
    )
    fe_center = LogMelFeatureExtractionV1(cfg=fe_center_cfg)
    fe_no_center_cfg = copy.deepcopy(fe_center_cfg)
    fe_no_center_cfg.center = False
    fe_no_center = LogMelFeatureExtractionV1(cfg=fe_no_center_cfg)
    for i in range(10):
        audio_length = int(np.random.randint(10000, 50000))
        audio = np.asarray(np.random.random(audio_length), dtype=np.float32)
        audio_length = torch.tensor(int(audio_length))
        audio_length = torch.unsqueeze(audio_length, 0)
        audio = torch.unsqueeze(torch.tensor(audio), 0)
        mel_center, length_center = fe_center(audio, audio_length)
        assert torch.all(mel_center.size()[1] == length_center)
        mel_no_center, length_no_center = fe_no_center(audio, audio_length)
        assert torch.all(mel_no_center.size()[1] == length_no_center)
