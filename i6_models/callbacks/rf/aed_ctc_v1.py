__all__ = ["model_def", "train_def", "recog_def_aed"]

import functools
from typing import Optional, Tuple, TYPE_CHECKING

import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.tensor import Dim, Tensor, single_step_dim

if TYPE_CHECKING:
    from i6_models.assemblies.rf.aed_ctc_v1 import ConformerAedCtcModelV1


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def model_def(*, epoch: int, in_dim: Dim, target_dim: Dim):
    """
    Model definition function for the AED + CTC model.

    Function is run within RETURNN within the ModelDef/TrainDef/RecogDef framework
    of callbacks that are implemented outside of i6_models.
    """
    from i6_models.assemblies.rf.aed_ctc_v1 import ConformerAedCtcModelV1, ConformerAedCtcModelV1Config

    from returnn.config import get_global_config
    import returnn.frontend as rf

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers", (4, 8))
    num_enc_layers = config.int("num_enc_layers", 12)
    num_dec_layers = config.int("num_dec_layers", 6)
    enc_model_dim = config.typed_value("enc_model_dim", 512)
    dec_model_dim = config.typed_value("dec_model_dim", 512)
    enc_ff_dim = config.typed_value("enc_ff_dim", 2048)
    enc_att_num_heads = config.typed_value("enc_att_num_heads", 8)
    sampling_rate = config.typed_value("sampling_rate", 16_000)

    # real input is raw audio, internally it does logmel
    mel_feature_dim = config.typed_value("mel_feature_dim", 80)
    in_dim = Dim(name="logmel", dimension=mel_feature_dim, kind=Dim.Types.Feature)

    cfg = ConformerAedCtcModelV1Config(
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        enc_model_dim=Dim(name="enc", dimension=enc_model_dim, kind=Dim.Types.Feature),
        dec_model_dim=Dim(name="dec", dimension=dec_model_dim, kind=Dim.Types.Feature),
        enc_ff_dim=Dim(name="enc-ff", dimension=enc_ff_dim, kind=Dim.Types.Feature),
        enc_att_num_heads=enc_att_num_heads,
        enc_conformer_layer_opts=dict(
            conv_norm_opts=dict(use_mask=True),
            self_att_opts=dict(
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
                pos_emb_dropout=0.1,
            ),
            ff_activation=lambda x: rf.relu(x) ** 2.0,
        ),
        enc_aux_logits=enc_aux_logits or (),
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        sampling_rate=sampling_rate,
    )
    return ConformerAedCtcModelV1(in_dim, target_dim, cfg)


def train_def(
    *,
    model: "ConformerAedCtcModelV1",
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_spatial_dim: Dim,
):
    """
    Training function for AED+CTC model.

    Function is run within RETURNN within the ModelDef/TrainDef/RecogDef framework
    of callbacks that are implemented outside of i6_models.
    """
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_loss = rf.ctc_loss(
                logits=aux_logits,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=model.eos_idx,
        out_dims=[targets_w_eos_spatial_dim],
    )

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
    )

    logits_packed, pack_dim = rf.pack_padded(
        logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    targets_packed, _ = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


def _gather_backrefs(s, *, backrefs: Tensor):
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        assert s.dimension or backrefs not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def recog_def_aed(
    *,
    model: "ConformerAedCtcModelV1",
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Open-vocabulary AED recognition function for AED+CTC model.

    Function is run within RETURNN within the ModelDef/TrainDef/RecogDef framework
    of callbacks that are implemented outside of i6_models.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import tree

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12
    length_normalization_exponent = 1.0
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        logits, decoder_state = model.decoder(
            target,
            spatial_dim=single_step_dim,
            encoder=enc,
            state=decoder_state,
        )
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim
