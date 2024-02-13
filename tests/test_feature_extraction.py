import os
import copy
import numpy as np
import torch
import unittest

from librosa.feature import melspectrogram

from i6_models.primitives.feature_extraction import *


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


def test_rasr_compatible():
    try:
        from i6_core.lib.rasr_cache import FileArchive
    except ImportError:
        raise unittest.SkipTest("i6_core not available")
    try:
        import soundfile
    except ImportError:
        raise unittest.SkipTest("soundfile not available")
    if not os.path.exists("test_data/features.cache") or not os.path.exists("test_data/103-1240-0000.wav"):
        raise unittest.SkipTest("test data not available")

    def _torch_repr(x: torch.Tensor) -> str:
        try:
            from lovely_tensors import lovely
        except ImportError:
            mean, std = x.mean(), x.std()
            min_, max_ = x.min(), x.max()
            return f"{x.shape} x∈[{min_}, {max_}] μ={mean} σ={std} {x.dtype}"
        else:
            return lovely(x)

    rasr_cache = FileArchive("test_data/features.cache", must_exists=True)
    print(rasr_cache.file_list())
    print(rasr_cache.read("corpus/103-1240-0000/1.attribs", "str"))
    time_, rasr_feat = rasr_cache.read("corpus/103-1240-0000/1", "feat")
    assert len(time_) == len(rasr_feat)
    print("RASR feature len:", len(rasr_feat), "frame 0 times:", time_[0], "frame 0 shape:", rasr_feat[0].shape)
    rasr_feat = torch.tensor(np.stack(rasr_feat, axis=0), dtype=torch.float32)
    print("RASR feature shape:", rasr_feat.shape)

    cfg = RasrCompatibleLogMelFeatureExtractionV1Config(
        sample_rate=16_000,
        win_size=0.025,
        hop_size=0.01,
        min_amp=1.175494e-38,
        num_filters=80,
    )
    feat_extractor = RasrCompatibleLogMelFeatureExtractionV1(cfg)

    # int16 audio is in [2**15, 2**15-1].
    # This is how BlissToPcmHDFJob does it by default:
    # https://github.com/rwth-i6/i6_core/blob/add09a8b640a2ba5928b815fa65f7504242be038/returnn/hdf.py#L207
    # This is also how our standard RASR flow handles it:
    # https://github.com/rwth-i6/i6_models/pull/44#issuecomment-1938264642
    audio, sample_rate = soundfile.read(open("test_data/103-1240-0000.wav", "rb"), dtype="int16")
    assert sample_rate == cfg.sample_rate
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]
    print("raw audio", _torch_repr(audio))

    i6m_feat, _ = feat_extractor(audio.unsqueeze(0), torch.tensor([len(audio)]))
    i6m_feat = i6m_feat.squeeze(0)

    print("i6_models:", _torch_repr(i6m_feat))
    print("RASR:", _torch_repr(rasr_feat))

    torch.testing.assert_allclose(i6m_feat, rasr_feat, rtol=1e-5, atol=1e-5)
