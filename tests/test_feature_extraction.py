import os
import sys
import copy
import math
import numpy as np
import torch
import unittest
import tempfile
import atexit
import textwrap

from typing import Optional
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
    rasr_feature_extractor_bin_path = (
        "/work/tools22/asr/rasr/rasr_onnx_haswell_0623/arch/linux-x86_64-standard/"
        "feature-extraction.linux-x86_64-standard"
    )
    if not os.path.exists(rasr_feature_extractor_bin_path):
        raise unittest.SkipTest("RASR feature-extraction binary not found")

    torch.manual_seed(42)
    wav_file_path = tempfile.mktemp(suffix=".wav", prefix="tmp-i6models-random-audio")
    atexit.register(os.remove, wav_file_path)
    generate_random_speech_like_audio_wav(wav_file_path)
    rasr_feature_cache_path = generate_rasr_feature_cache_from_wav_and_flow(
        rasr_feature_extractor_bin_path,
        wav_file_path,
        textwrap.dedent(
            f"""\
            <node filter="generic-vector-s16-demultiplex" name="demultiplex" track="$(track)"/>
            <link from="samples" to="demultiplex"/>
            <node filter="generic-convert-vector-s16-to-vector-f32" name="convert"/>
            <link from="demultiplex" to="convert"/>
            <node alpha="1.0" filter="signal-preemphasis" name="preemphasis"/>
            <link from="convert" to="preemphasis"/>
            <node filter="signal-window" length="0.025" name="window" shift="0.01" type="hanning"/>
            <link from="preemphasis" to="window"/>
            <node filter="signal-real-fast-fourier-transform" maximum-input-size="0.025" name="fft"/>
            <link from="window" to="fft"/>
            <node filter="signal-vector-alternating-complex-f32-amplitude" name="amplitude-spectrum"/>
            <link from="fft" to="amplitude-spectrum"/>
            <node filter="signal-filterbank" filter-width="70.12402584464985" name="filterbank"
             warp-differential-unit="false" warping-function="mel"/>
            <link from="amplitude-spectrum" to="filterbank"/>
            <node filter="generic-vector-f32-log-plus" name="nonlinear" value="1.175494e-38"/>
            <link from="filterbank" to="nonlinear"/>
            """
        ),
        flow_output_name="nonlinear",
    )

    rasr_cache = FileArchive(rasr_feature_cache_path, must_exists=True)
    print(rasr_cache.file_list())
    print(rasr_cache.read("corpus/recording/1.attribs", "str"))
    time_, rasr_feat = rasr_cache.read("corpus/recording/1", "feat")
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
    audio, sample_rate = soundfile.read(open(wav_file_path, "rb"), dtype="int16")
    assert sample_rate == cfg.sample_rate
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]
    print("raw audio", _torch_repr(audio))

    i6m_feat, _ = feat_extractor(audio.unsqueeze(0), torch.tensor([len(audio)]))
    i6m_feat = i6m_feat.squeeze(0)

    print("i6_models:", _torch_repr(i6m_feat))
    print("RASR:", _torch_repr(rasr_feat))

    torch.testing.assert_close(i6m_feat, rasr_feat, rtol=1e-5, atol=1e-5)


def test_rasr_compatible_raw_audio_samples():
    try:
        from i6_core.lib.rasr_cache import FileArchive
    except ImportError:
        raise unittest.SkipTest("i6_core not available")
    try:
        import soundfile
    except ImportError:
        raise unittest.SkipTest("soundfile not available")
    rasr_feature_extractor_bin_path = (
        "/work/tools22/asr/rasr/rasr_onnx_haswell_0623/arch/linux-x86_64-standard/"
        "feature-extraction.linux-x86_64-standard"
    )
    if not os.path.exists(rasr_feature_extractor_bin_path):
        raise unittest.SkipTest("RASR feature-extraction binary not found")

    torch.manual_seed(42)
    wav_file_path = tempfile.mktemp(suffix=".wav", prefix="tmp-i6models-random-audio")
    atexit.register(os.remove, wav_file_path)
    generate_random_speech_like_audio_wav(wav_file_path)
    rasr_feature_cache_path = generate_rasr_feature_cache_from_wav_and_flow(
        rasr_feature_extractor_bin_path,
        wav_file_path,
        textwrap.dedent(
            f"""\
            <node filter="generic-vector-s16-demultiplex" name="demultiplex" track="$(track)"/>
            <link from="samples" to="demultiplex"/>
            <node filter="generic-convert-vector-s16-to-vector-f32" name="convert"/>
            <link from="demultiplex" to="convert"/>
            """
        ),
        flow_output_name="convert",
    )

    rasr_cache = FileArchive(rasr_feature_cache_path, must_exists=True)
    time_, rasr_feat = rasr_cache.read("corpus/recording/1", "feat")
    assert len(time_) == len(rasr_feat)
    rasr_feat = torch.tensor(np.concatenate(rasr_feat, axis=0), dtype=torch.float32)
    print("RASR:", _torch_repr(rasr_feat))

    audio, sample_rate = soundfile.read(open(wav_file_path, "rb"), dtype="int16")
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]
    print("raw audio", _torch_repr(audio))

    torch.testing.assert_close(audio, rasr_feat, rtol=1e-30, atol=1e-30)


def test_rasr_compatible_preemphasis():
    try:
        from i6_core.lib.rasr_cache import FileArchive
    except ImportError:
        raise unittest.SkipTest("i6_core not available")
    try:
        import soundfile
    except ImportError:
        raise unittest.SkipTest("soundfile not available")
    rasr_feature_extractor_bin_path = (
        "/work/tools22/asr/rasr/rasr_onnx_haswell_0623/arch/linux-x86_64-standard/"
        "feature-extraction.linux-x86_64-standard"
    )
    if not os.path.exists(rasr_feature_extractor_bin_path):
        raise unittest.SkipTest("RASR feature-extraction binary not found")

    torch.manual_seed(42)
    wav_file_path = tempfile.mktemp(suffix=".wav", prefix="tmp-i6models-random-audio")
    atexit.register(os.remove, wav_file_path)
    generate_random_speech_like_audio_wav(wav_file_path)
    rasr_feature_cache_path = generate_rasr_feature_cache_from_wav_and_flow(
        rasr_feature_extractor_bin_path,
        wav_file_path,
        textwrap.dedent(
            f"""\
            <node filter="generic-vector-s16-demultiplex" name="demultiplex" track="$(track)"/>
            <link from="samples" to="demultiplex"/>
            <node filter="generic-convert-vector-s16-to-vector-f32" name="convert"/>
            <link from="demultiplex" to="convert"/>
            <node alpha="1.0" filter="signal-preemphasis" name="preemphasis"/>
            <link from="convert" to="preemphasis"/>
            """
        ),
        flow_output_name="preemphasis",
    )

    rasr_cache = FileArchive(rasr_feature_cache_path, must_exists=True)
    time_, rasr_feat = rasr_cache.read("corpus/recording/1", "feat")
    assert len(time_) == len(rasr_feat)
    rasr_feat = torch.tensor(np.concatenate(rasr_feat, axis=0), dtype=torch.float32)
    print("RASR:", _torch_repr(rasr_feat))

    audio, sample_rate = soundfile.read(open(wav_file_path, "rb"), dtype="int16")
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]
    audio[..., 1:] -= 1.0 * audio[..., :-1]
    audio[..., 0] = 0.0
    print("i6_models", _torch_repr(audio))

    torch.testing.assert_close(audio, rasr_feat, rtol=1e-30, atol=1e-30)


def test_rasr_compatible_window():
    try:
        from i6_core.lib.rasr_cache import FileArchive
    except ImportError:
        raise unittest.SkipTest("i6_core not available")
    try:
        import soundfile
    except ImportError:
        raise unittest.SkipTest("soundfile not available")
    rasr_feature_extractor_bin_path = (
        "/work/tools22/asr/rasr/rasr_onnx_haswell_0623/arch/linux-x86_64-standard/"
        "feature-extraction.linux-x86_64-standard"
    )
    if not os.path.exists(rasr_feature_extractor_bin_path):
        raise unittest.SkipTest("RASR feature-extraction binary not found")

    torch.manual_seed(42)
    wav_file_path = tempfile.mktemp(suffix=".wav", prefix="tmp-i6models-random-audio")
    atexit.register(os.remove, wav_file_path)
    generate_random_speech_like_audio_wav(wav_file_path)
    rasr_feature_cache_path = generate_rasr_feature_cache_from_wav_and_flow(
        rasr_feature_extractor_bin_path,
        wav_file_path,
        textwrap.dedent(
            f"""\
            <node filter="generic-vector-s16-demultiplex" name="demultiplex" track="$(track)"/>
            <link from="samples" to="demultiplex"/>
            <node filter="generic-convert-vector-s16-to-vector-f32" name="convert"/>
            <link from="demultiplex" to="convert"/>
            <node alpha="1.0" filter="signal-preemphasis" name="preemphasis"/>
            <link from="convert" to="preemphasis"/>
            <node filter="signal-window" length="0.025" name="window" shift="0.01" type="hanning"/>
            <link from="preemphasis" to="window"/>
            """
        ),
        flow_output_name="window",
    )

    rasr_cache = FileArchive(rasr_feature_cache_path, must_exists=True)
    time_, rasr_feat = rasr_cache.read("corpus/recording/1", "feat")
    assert len(time_) == len(rasr_feat)
    rasr_last_pad = rasr_feat[0].shape[0] - rasr_feat[-1].shape[0]
    rasr_feat[-1] = np.pad(rasr_feat[-1], (0, rasr_last_pad))
    rasr_feat = torch.tensor(np.stack(rasr_feat, axis=0), dtype=torch.float32)
    print("RASR:", _torch_repr(rasr_feat))

    audio, sample_rate = soundfile.read(open(wav_file_path, "rb"), dtype="int16")
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]

    # preemphasize
    audio[..., 1:] -= 1.0 * audio[..., :-1]
    audio[..., 0] = 0.0

    # windowing
    win_size = 0.025
    hop_size = 0.01
    hop_length = int(hop_size * sample_rate)
    win_length = int(win_size * sample_rate)

    def _get_length(audio_len: int) -> int:
        if audio_len == 0:
            return 0
        return max(audio_len - win_length + hop_length - 1, 0) // hop_length + 1

    def _get_length_naive(audio_len: int) -> int:
        n = 0
        for t in range(0, audio_len, hop_length):
            n += 1
            if audio_len <= t + win_length:
                break
        return n

    for t in range(10 * win_length):
        assert _get_length(t) == _get_length_naive(t), f"t={t}, {_get_length(t)} != {_get_length_naive(t)}"

    res_len = _get_length(audio.shape[0])
    assert res_len == len(rasr_feat)

    # https://github.com/rwth-i6/rasr/blob/master/src/Signal/WindowFunction.cc
    # https://pytorch.org/docs/stable/generated/torch.hann_window.html

    # manual
    for i, t in enumerate(range(0, audio.shape[0], hop_length)):
        x = audio[t : t + win_length]
        x = x * torch.hann_window(x.shape[0], periodic=False, dtype=torch.float64).to(torch.float32)
        torch.testing.assert_close(x, rasr_feat[i][: x.shape[0]], rtol=1e-30, atol=1e-30)
        # once end was reached, stop
        if audio.shape[0] <= t + win_length:
            assert win_length - x.shape[0] == rasr_last_pad, f"win {win_length}, cur {x.shape[0]}, pad {rasr_last_pad}"
            break

    last_win_size = audio.shape[0] - (res_len - 1) * hop_length
    last_pad = win_length - last_win_size
    assert last_pad == rasr_last_pad, f"last pad {last_pad}, RASR last pad {rasr_last_pad}"
    padded = torch.nn.functional.pad(audio, (0, last_pad))  # zero pad for the last frame

    windowed = padded.unfold(0, size=win_length, step=hop_length)  # [T', W=win_length]
    assert len(windowed) == res_len
    window = torch.hann_window(win_length, periodic=False, dtype=torch.float64).to(torch.float32)
    smoothed = windowed[:-1] * window[None, :]  # [T'-1, W]

    # The last window might be shorter. Will use a shorter Hanning window then. Need to fix that.
    last_win = torch.hann_window(last_win_size, periodic=False, dtype=torch.float64).to(torch.float32)
    last_win = torch.nn.functional.pad(last_win, (0, last_pad))
    smoothed = torch.cat([smoothed, (windowed[-1] * last_win)[None, :]], dim=0)

    print("i6_models", _torch_repr(smoothed))

    torch.testing.assert_close(smoothed, rasr_feat, rtol=1e-30, atol=1e-30)


def test_rasr_compatible_fft():
    try:
        from i6_core.lib.rasr_cache import FileArchive
    except ImportError:
        raise unittest.SkipTest("i6_core not available")
    try:
        import soundfile
    except ImportError:
        raise unittest.SkipTest("soundfile not available")
    rasr_feature_extractor_bin_path = (
        "/work/tools22/asr/rasr/rasr_onnx_haswell_0623/arch/linux-x86_64-standard/"
        "feature-extraction.linux-x86_64-standard"
    )
    if not os.path.exists(rasr_feature_extractor_bin_path):
        raise unittest.SkipTest("RASR feature-extraction binary not found")

    torch.manual_seed(42)
    wav_file_path = tempfile.mktemp(suffix=".wav", prefix="tmp-i6models-random-audio")
    atexit.register(os.remove, wav_file_path)
    generate_random_speech_like_audio_wav(wav_file_path)
    rasr_feature_cache_path = generate_rasr_feature_cache_from_wav_and_flow(
        rasr_feature_extractor_bin_path,
        wav_file_path,
        textwrap.dedent(
            f"""\
            <node filter="generic-vector-s16-demultiplex" name="demultiplex" track="$(track)"/>
            <link from="samples" to="demultiplex"/>
            <node filter="generic-convert-vector-s16-to-vector-f32" name="convert"/>
            <link from="demultiplex" to="convert"/>
            <node alpha="1.0" filter="signal-preemphasis" name="preemphasis"/>
            <link from="convert" to="preemphasis"/>
            <node filter="signal-window" length="0.025" name="window" shift="0.01" type="hanning"/>
            <link from="preemphasis" to="window"/>
            <node filter="signal-real-fast-fourier-transform" maximum-input-size="0.025" name="fft"/>
            <link from="window" to="fft"/>
            <node filter="generic-vector-f32-multiplication" name="scaling" value="16000"/>
            <link from="fft" to="scaling"/>
            """
        ),
        flow_output_name="scaling",
    )

    rasr_cache = FileArchive(rasr_feature_cache_path, must_exists=True)
    time_, rasr_feat = rasr_cache.read("corpus/recording/1", "feat")
    assert len(time_) == len(rasr_feat)
    rasr_feat = torch.tensor(np.stack(rasr_feat, axis=0), dtype=torch.float32)
    print("RASR:", _torch_repr(rasr_feat))

    audio, sample_rate = soundfile.read(open(wav_file_path, "rb"), dtype="int16")
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]

    # preemphasize
    audio[..., 1:] -= 1.0 * audio[..., :-1]
    audio[..., 0] = 0.0

    # windowing
    win_size = 0.025
    hop_size = 0.01
    hop_length = int(hop_size * sample_rate)
    win_length = int(win_size * sample_rate)

    res_len = max(audio.shape[0] - win_length + hop_length - 1, 0) // hop_length + 1
    assert res_len == len(rasr_feat)

    last_win_size = audio.shape[0] - (res_len - 1) * hop_length
    last_pad = win_length - last_win_size
    padded = torch.nn.functional.pad(audio, (0, last_pad))  # zero pad for the last frame

    windowed = padded.unfold(0, size=win_length, step=hop_length)  # [T', W=win_length]
    assert len(windowed) == res_len
    window = torch.hann_window(win_length, periodic=False, dtype=torch.float64).to(torch.float32)
    smoothed = windowed[:-1] * window[None, :]  # [T'-1, W]

    # The last window might be shorter. Will use a shorter Hanning window then. Need to fix that.
    last_win = torch.hann_window(last_win_size, periodic=False, dtype=torch.float64).to(torch.float32)
    last_win = torch.nn.functional.pad(last_win, (0, last_pad))
    smoothed = torch.cat([smoothed, (windowed[-1] * last_win)[None, :]], dim=0)

    n_fft = 2 ** math.ceil(math.log2(win_length))
    print(f"win_length={win_length}, n_fft={n_fft}")

    # fft = torch.fft.rfftn(smoothed, s=n_fft)  # [B, T', F]
    # fft = torch.view_as_real(fft).flatten(-2)  # [B, T', F*2]
    # fft = torch.fft.rfftn(smoothed, s=n_fft)  # [B, T', F=n_fft//2+1]
    # fft = torch.view_as_real(fft).flatten(-2)  # [B, T', F=(n_fft//2+1)*2]
    fft = my_fft(smoothed, n_fft=n_fft)

    print("i6_models", _torch_repr(fft))

    torch.testing.assert_close(fft, rasr_feat, rtol=1e-6, atol=1e-2)  # values are huge, accept some errors...?


def test_rasr_compatible_amplitude_spectrum():
    try:
        from i6_core.lib.rasr_cache import FileArchive
    except ImportError:
        raise unittest.SkipTest("i6_core not available")
    try:
        import soundfile
    except ImportError:
        raise unittest.SkipTest("soundfile not available")
    rasr_feature_extractor_bin_path = (
        "/work/tools22/asr/rasr/rasr_onnx_haswell_0623/arch/linux-x86_64-standard/"
        "feature-extraction.linux-x86_64-standard"
    )
    if not os.path.exists(rasr_feature_extractor_bin_path):
        raise unittest.SkipTest("RASR feature-extraction binary not found")

    torch.manual_seed(42)
    wav_file_path = tempfile.mktemp(suffix=".wav", prefix="tmp-i6models-random-audio")
    atexit.register(os.remove, wav_file_path)
    generate_random_speech_like_audio_wav(wav_file_path)
    rasr_feature_cache_path = generate_rasr_feature_cache_from_wav_and_flow(
        rasr_feature_extractor_bin_path,
        wav_file_path,
        textwrap.dedent(
            f"""\
            <node filter="generic-vector-s16-demultiplex" name="demultiplex" track="$(track)"/>
            <link from="samples" to="demultiplex"/>
            <node filter="generic-convert-vector-s16-to-vector-f32" name="convert"/>
            <link from="demultiplex" to="convert"/>
            <node alpha="1.0" filter="signal-preemphasis" name="preemphasis"/>
            <link from="convert" to="preemphasis"/>
            <node filter="signal-window" length="0.025" name="window" shift="0.01" type="hanning"/>
            <link from="preemphasis" to="window"/>
            <node filter="signal-real-fast-fourier-transform" maximum-input-size="0.025" name="fft"/>
            <link from="window" to="fft"/>
            <node filter="generic-vector-f32-multiplication" name="scaling" value="16000"/>
            <link from="fft" to="scaling"/>
            <node filter="signal-vector-alternating-complex-f32-amplitude" name="amplitude-spectrum"/>
            <link from="scaling" to="amplitude-spectrum"/>
            """
        ),
        flow_output_name="amplitude-spectrum",
    )

    rasr_cache = FileArchive(rasr_feature_cache_path, must_exists=True)
    time_, rasr_feat = rasr_cache.read("corpus/recording/1", "feat")
    assert len(time_) == len(rasr_feat)
    rasr_feat = torch.tensor(np.stack(rasr_feat, axis=0), dtype=torch.float32)
    print("RASR:", _torch_repr(rasr_feat))

    audio, sample_rate = soundfile.read(open(wav_file_path, "rb"), dtype="int16")
    audio = torch.tensor(audio.astype(np.float32))  # [-2**15, 2**15-1]

    # preemphasize
    audio[..., 1:] -= 1.0 * audio[..., :-1]
    audio[..., 0] = 0.0

    # windowing
    win_size = 0.025
    hop_size = 0.01
    hop_length = int(hop_size * sample_rate)
    win_length = int(win_size * sample_rate)

    res_len = max(audio.shape[0] - win_length + hop_length - 1, 0) // hop_length + 1
    assert res_len == len(rasr_feat)

    last_win_size = audio.shape[0] - (res_len - 1) * hop_length
    last_pad = win_length - last_win_size
    padded = torch.nn.functional.pad(audio, (0, last_pad))  # zero pad for the last frame

    windowed = padded.unfold(0, size=win_length, step=hop_length)  # [T', W=win_length]
    assert len(windowed) == res_len
    window = torch.hann_window(win_length, periodic=False, dtype=torch.float64).to(torch.float32)
    smoothed = windowed[:-1] * window[None, :]  # [T'-1, W]

    # The last window might be shorter. Will use a shorter Hanning window then. Need to fix that.
    last_win = torch.hann_window(last_win_size, periodic=False, dtype=torch.float64).to(torch.float32)
    last_win = torch.nn.functional.pad(last_win, (0, last_pad))
    smoothed = torch.cat([smoothed, (windowed[-1] * last_win)[None, :]], dim=0)

    n_fft = 2 ** math.ceil(math.log2(win_length))
    print(f"win_length={win_length}, n_fft={n_fft}")

    # fft = torch.fft.rfftn(smoothed, s=n_fft)  # [B, T', F]
    # fft = torch.view_as_real(fft).flatten(-2)  # [B, T', F*2]
    # fft = torch.fft.rfftn(smoothed, s=n_fft)  # [B, T', F=n_fft//2+1]
    # fft = torch.view_as_real(fft).flatten(-2)  # [B, T', F=(n_fft//2+1)*2]
    fft = my_fft(smoothed, n_fft=n_fft)
    amplitude_spectrum = torch.abs(torch.view_as_complex(fft.unflatten(-1, (-1, 2))))

    print("i6_models", _torch_repr(amplitude_spectrum))

    torch.testing.assert_close(amplitude_spectrum, rasr_feat, rtol=1e-30, atol=1e-30)


def create_bit_reversal_reordering(size):
    """
    Creates a bit reversal reordering tensor for a given size.
    """
    # Initial setup
    length = size // 2
    reordering = torch.arange(size)

    # Bit reversal reordering logic
    j = 1
    for i in range(1, size, 2):
        if j > i:
            reordering[i - 1] = j - 1
            reordering[i] = j
        m = length
        while 2 <= m < j:
            j -= m
            m //= 2
        j += m

    return reordering


def bit_reversal_reordering(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reorders a given tensor according to bit reversal pattern.
    """
    size = tensor.size(-1)
    reordering = create_bit_reversal_reordering(size)
    v = tensor.clone()
    for i in range(size):
        h = v[..., i].clone()
        v[..., i] = v[..., reordering[i]]
        v[..., reordering[i]] = h
    return v


def my_fft(tensor: torch.Tensor, *, n_fft: int) -> torch.Tensor:
    # https://github.com/rwth-i6/rasr/blob/master/src/Math/FastFourierTransform.cc#L95
    size = n_fft
    d_pi = torch.tensor(6.28318530717959, dtype=torch.float64)
    theta_base = d_pi

    tensor = torch.nn.functional.pad(tensor, (0, size - tensor.shape[-1]))
    v = bit_reversal_reordering(tensor)

    cur_length = 2
    # estimate DFFT using Danielson and Lanczos formula
    while cur_length < size:
        # initialization of trigonometric recurrence
        step = cur_length * 2
        theta = theta_base / cur_length
        sin_h_theta = torch.sin(0.5 * theta)
        wp_r = -2.0 * sin_h_theta * sin_h_theta
        wp_i = torch.sin(theta)
        w_r = torch.tensor(1.0, dtype=torch.float64)
        w_i = torch.tensor(0.0, dtype=torch.float64)
        for m in range(1, cur_length, 2):
            for i in range(m, size, step):
                # Danielson & Lanczos formula
                j = i + cur_length
                tempr = w_r * v[..., j - 1].to(torch.float64) - w_i * v[..., j].to(torch.float64)
                tempi = w_r * v[..., j].to(torch.float64) + w_i * v[..., j - 1].to(torch.float64)
                tempr = tempr.to(torch.float32)
                tempi = tempi.to(torch.float32)
                v[..., j - 1] = v[..., i - 1] - tempr
                v[..., j] = v[..., i] - tempi
                v[..., i - 1] += tempr
                v[..., i] += tempi
            w_temp_r, w_temp_i = w_r, w_i
            w_r = w_r * wp_r - w_temp_i * wp_i + w_r
            w_i = w_i * wp_r + w_temp_r * wp_i + w_i
        cur_length = step

    pi = torch.tensor(3.141592653589793238, dtype=torch.float64)
    size_d4 = size >> 2
    theta = pi / (size >> 1)

    sin_h_theta = torch.sin(0.5 * theta)
    wp_r = -2.0 * sin_h_theta * sin_h_theta
    wp_i = torch.sin(theta)
    w_r = wp_r + 1
    w_i = wp_i

    for i in range(1, size_d4):
        i1 = i + i
        i2 = i1 + 1
        i3 = size - i1
        i4 = i3 + 1

        # separate the two transforms
        h1_r = 0.5 * (v[..., i1] + v[..., i3]).to(torch.float64)
        h1_i = 0.5 * (v[..., i2] - v[..., i4]).to(torch.float64)
        h2_r = 0.5 * (v[..., i2] + v[..., i4]).to(torch.float64)
        h2_i = -0.5 * (v[..., i1] - v[..., i3]).to(torch.float64)

        # Calculating the true transform of the original real data
        v[..., i1] = h1_r + w_r * h2_r - w_i * h2_i
        v[..., i2] = h1_i + w_r * h2_i + w_i * h2_r
        v[..., i3] = h1_r - w_r * h2_r + w_i * h2_i
        v[..., i4] = -h1_i + w_r * h2_i + w_i * h2_r

        # Updating the trigonometric recurrences
        old_wr = w_r
        w_r = w_r * wp_r - w_i * wp_i + w_r
        w_i = w_i * wp_r + old_wr * wp_i + w_i

    h = v[..., 0].clone()
    v[..., 0] = h + v[..., 1]
    v[..., 1] = h - v[..., 1]

    # unpack logic
    v = torch.nn.functional.pad(v, (0, 2))
    v[..., -2] = v[..., 1]
    v[..., 1] = 0

    return v


def generate_rasr_feature_cache_from_wav_and_flow(
    rasr_feature_extractor_bin_path: str,
    wav_file_path: str,
    flow_network_str: str,
    *,
    flow_input_name: str = "samples",
    flow_output_name: str = "nonlinear",
) -> str:
    import subprocess

    corpus_xml_path = tempfile.mktemp(suffix=".xml", prefix="tmp-rasr-corpus")
    atexit.register(os.remove, corpus_xml_path)
    with open(corpus_xml_path, "w") as f:
        f.write(
            textwrap.dedent(
                f"""\
                <?xml version="1.0" encoding="utf8"?>
                <corpus name="corpus">
                  <recording name="recording" audio="{wav_file_path}">
                    <segment name="1" start="0.0000" end="{_get_wav_file_duration_sec(wav_file_path)}" track="0">
                    </segment>
                  </recording>
                </corpus>
                """
            )
        )

    rasr_feature_cache_path = tempfile.mktemp(suffix=".cache", prefix="tmp-rasr-features")
    atexit.register(os.remove, rasr_feature_cache_path)
    rasr_flow_xml_path = tempfile.mktemp(suffix=".config", prefix="tmp-rasr-flow")
    atexit.register(os.remove, rasr_flow_xml_path)
    with open(rasr_flow_xml_path, "w") as f:
        f.write(
            textwrap.dedent(
                f"""\
                <?xml version="1.0" ?>
                <network name="network">
                  <out name="features"/>
                  <param name="end-time"/>
                  <param name="input-file"/>
                  <param name="start-time"/>
                  <param name="track"/>
                  <param name="id"/>
                  <node filter="audio-input-file-wav" file="$(input-file)"
                   start-time="$(start-time)" end-time="$(end-time)" name="{flow_input_name}"/>
                """
            )
            + textwrap.indent(flow_network_str, "  ")
            + textwrap.dedent(
                f"""\
                  <node filter="generic-cache" id="$(id)" name="cache" path="{rasr_feature_cache_path}"/>
                  <link from="{flow_output_name}" to="cache"/>
                  <link from="cache" to="network:features"/>
                </network>
                """
            )
        )

    rasr_config_path = tempfile.mktemp(suffix=".config", prefix="tmp-rasr-feature-extract")
    atexit.register(os.remove, rasr_config_path)
    with open(rasr_config_path, "w") as f:
        f.write(
            textwrap.dedent(
                f"""\
                [*.corpus]
                file = {corpus_xml_path}

                [*.feature-extraction]
                file = {rasr_flow_xml_path}
                """
            )
        )

    subprocess.check_call([rasr_feature_extractor_bin_path, "--config", rasr_config_path])
    return rasr_feature_cache_path


def _get_wav_file_duration_sec(wav_file_path: str) -> float:
    import wave
    import contextlib

    with contextlib.closing(wave.open(wav_file_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def generate_random_speech_like_audio_wav(
    output_wav_file_path: str,
    duration_sec: float = 5.0,
    *,
    samples_per_sec: int = 16_000,
    sample_width_bytes: int = 2,  # int16
    frequency: float = 150.0,
    num_random_freqs_per_sec: int = 15,
    amplitude: float = 0.3,
    amplitude_frequency: Optional[float] = None,
):
    import wave

    f = wave.open(output_wav_file_path, "wb")
    f.setframerate(samples_per_sec)
    f.setnchannels(1)
    f.setsampwidth(sample_width_bytes)

    samples = generate_random_speech_like_audio(
        batch_size=1,
        num_frames=int(duration_sec * samples_per_sec),
        samples_per_sec=samples_per_sec,
        frequency=frequency,
        num_random_freqs_per_sec=num_random_freqs_per_sec,
        amplitude=amplitude,
        amplitude_frequency=amplitude_frequency,
    )  # [B,T]
    samples = samples[0]  # [T]
    print("generated raw samples:", _torch_repr(samples))

    samples_int = (samples * (2 ** (8 * sample_width_bytes - 1) - 1)).to(
        {1: torch.int8, 2: torch.int16, 4: torch.int32}[sample_width_bytes]
    )

    f.writeframes(samples_int.numpy().tobytes())
    f.close()


def generate_random_speech_like_audio(
    batch_size: int,
    num_frames: int,
    *,
    samples_per_sec: int = 16_000,
    frequency: float = 150.0,
    num_random_freqs_per_sec: int = 15,
    amplitude: float = 0.3,
    amplitude_frequency: Optional[float] = None,
) -> torch.Tensor:
    """
    generate audio

    Source:
    https://github.com/albertz/playground/blob/master/create-random-speech-like-sound.py

    :return: shape [batch_size,num_frames]
    """
    frame_idxs = torch.arange(num_frames, dtype=torch.int64)  # [T]

    samples = _integrate_rnd_frequencies(
        batch_size,
        frame_idxs,
        base_frequency=frequency,
        samples_per_sec=samples_per_sec,
        num_random_freqs_per_sec=num_random_freqs_per_sec,
    )  # [T,B]

    if amplitude_frequency is None:
        amplitude_frequency = frequency / 75.0
    amplitude_variations = _integrate_rnd_frequencies(
        batch_size,
        frame_idxs,
        base_frequency=amplitude_frequency,
        samples_per_sec=samples_per_sec,
        num_random_freqs_per_sec=amplitude_frequency,
    )  # [T,B]

    samples *= amplitude * (0.666 + 0.333 * amplitude_variations)
    return samples.permute(1, 0)  # [B,T]


def _integrate_rnd_frequencies(
    batch_size: int,
    frame_idxs: torch.Tensor,
    *,
    base_frequency: float,
    samples_per_sec: int,
    num_random_freqs_per_sec: float,
) -> torch.Tensor:
    rnd_freqs = torch.empty(
        size=(int(len(frame_idxs) * num_random_freqs_per_sec / samples_per_sec) + 1, batch_size),
        dtype=torch.float32,
    )  # [T',B]
    torch.nn.init.trunc_normal_(rnd_freqs, a=-1.0, b=1.0)
    rnd_freqs = (rnd_freqs * 0.5 + 1.0) * base_frequency  # [T',B]

    freq_idx_f = (frame_idxs * num_random_freqs_per_sec) / samples_per_sec
    freq_idx = freq_idx_f.to(torch.int64)
    next_freq_idx = torch.clip(freq_idx + 1, 0, len(rnd_freqs) - 1)
    frac = (freq_idx_f % 1)[:, None]  # [T,1]
    freq = rnd_freqs[freq_idx] * (1 - frac) + rnd_freqs[next_freq_idx] * frac  # [T,B]

    ts = torch.cumsum(freq / samples_per_sec, dim=0)  # [T,B]
    return torch.sin(2 * torch.pi * ts)


def _torch_repr(x: torch.Tensor) -> str:
    try:
        from lovely_tensors import lovely
    except ImportError:
        mean, std = x.mean(), x.std()
        min_, max_ = x.min(), x.max()
        return f"{x.shape} x∈[{min_}, {max_}] μ={mean} σ={std} {x.dtype}"
    else:
        return lovely(x)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"*** {arg}()")
        globals()[arg]()
