__all__ = ["ConformerAedCtcModelV1"]

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


from i6_models.config import ModelConfiguration

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder


@dataclass
class ConformerAedCtcModelV1Config(ModelConfiguration):
    """
    Attributes:
        blank_idx: index of the blank label
        eos_idx: index of the end-of-sequence label
        bos_idx: index of the begin-of-sequence label
        num_enc_layers: number of Conformer encoder layers
        num_dec_layers: number of Transformer decoder layers
        enc_model_dim: Conformer encoder model dimension
        dec_model_dim: Transformer decoder model dimension
        enc_ff_dim: Conformer encoder feed-forward dimension
        enc_att_num_heads: number of Conformer encoder attention heads
        enc_conformer_layer_opts: optional Conformer layer options
        enc_dropout: Conformer encoder dropout rate
        enc_att_dropout: Conformer encoder attention dropout rate
        sampling_rate: audio sampling rate in Hz
        aux_target_dim: auxiliary target dimension
        enc_aux_logits: indices of encoder layers for auxiliary logits
        ctc_blank_label: label for CTC blank
    """

    blank_idx: int
    eos_idx: int
    bos_idx: int

    num_enc_layers: int = 12
    num_dec_layers: int = 6
    enc_model_dim: Dim = field(default_factory=lambda: Dim(name="enc", dimension=512))
    dec_model_dim: Dim = field(default_factory=lambda: Dim(name="dec", dimension=512))
    enc_ff_dim: Dim = field(default_factory=lambda: Dim(name="enc-ff", dimension=2048))
    enc_att_num_heads: int = 4
    enc_conformer_layer_opts: Optional[Dict[str, Any]] = None
    enc_dropout: float = 0.1
    enc_att_dropout: float = 0.1

    sampling_rate: int = 16_000

    enc_aux_logits: Sequence[int] = ()  # layer idx
    aux_target_dim: Optional[Dim] = None
    ctc_blank_label: str = "<blank>"

    def __post_init__(self):
        super().__post_init__()

        assert self.num_enc_layers > 0
        assert self.num_dec_layers > 0
        assert self.sampling_rate > 0
        assert self.enc_att_num_heads > 0
        assert self.enc_model_dim.dimension > 0
        assert self.enc_model_dim.dimension % self.enc_att_num_heads == 0


class ConformerAedCtcModelV1(rf.Module):
    """
    RF AED + CTC model using a Conformer encoder and Transformer decoder.
    """

    def __init__(self, in_dim: Dim, target_dim: Dim, cfg: ConformerAedCtcModelV1Config):
        super().__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.in_dim = in_dim
        self.sampling_rate = cfg.sampling_rate

        self.encoder = ConformerEncoder(
            in_dim,
            cfg.enc_model_dim,
            ff_dim=cfg.enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=cfg.enc_conformer_layer_opts,
            num_layers=cfg.num_enc_layers,
            num_heads=cfg.enc_att_num_heads,
            dropout=cfg.enc_dropout,
            att_dropout=cfg.enc_att_dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_dec_layers,
            encoder_dim=cfg.enc_model_dim,
            vocab_dim=target_dim,
            model_dim=cfg.dec_model_dim,
        )

        self.target_dim = target_dim
        self.blank_idx = cfg.blank_idx
        self.eos_idx = cfg.eos_idx
        self.bos_idx = cfg.bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if cfg.enc_aux_logits:
            wb_target_dim = cfg.aux_target_dim
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
            self.wb_target_dim = wb_target_dim

            if target_dim.vocab and not wb_target_dim.vocab:
                from returnn.datasets.util.vocabulary import Vocabulary

                # Add blank label to existing vocabulary
                assert cfg.ctc_blank_label
                assert wb_target_dim.dimension == target_dim.dimension + 1 and cfg.blank_idx == target_dim.dimension
                vocab_labels = list(target_dim.vocab.labels) + [cfg.ctc_blank_label]
                wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                    vocab_labels, user_defined_symbols={cfg.ctc_blank_label: cfg.blank_idx}
                )
        for i in cfg.enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.enc_aux_logits = cfg.enc_aux_logits

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (in_dim.dimension // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

    def encode(
        self, source: Tensor, *, in_spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[rf.State, Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source, in_spatial_dim=in_spatial_dim, out_dim=self.in_dim, sampling_rate=self.sampling_rate
        )
        source = rf.audio.specaugment(
            source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim, **self._specaugment_opts
        )
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        return self.decoder.transform_encoder(enc, axis=enc_spatial_dim), enc_spatial_dim
