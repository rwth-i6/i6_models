from typing import Optional

import torch
from torch import nn

from i6_models.decoder.attention import (
    AdditiveAttention,
    AdditiveAttentionConfig,
    AttentionLSTMDecoderV1,
    AttentionLSTMDecoderV1Config,
    AttentionLSTMDecoderV1RasrEncoder,
    AttentionLSTMDecoderV1RasrScorer,
    AttentionLSTMDecoderV1RasrStateInitializer,
    AttentionLSTMDecoderV1RasrStateUpdater,
)
from i6_models.decoder.zoneout_lstm import ZoneoutLSTMCell


def _make_decoder(
    *,
    encoder_dim: int = 5,
    vocab_size: int = 15,
    target_embed_dim: int = 3,
    lstm_hidden_size: int = 12,
    attention_dim: int = 10,
    output_proj_dim: int = 12,
    dropout: float = 0.0,
    zoneout_drop_h: float = 0.0,
    zoneout_drop_c: float = 0.0,
    target_padding_idx: Optional[int] = None,
) -> AttentionLSTMDecoderV1:
    decoder_cfg = AttentionLSTMDecoderV1Config(
        encoder_dim=encoder_dim,
        vocab_size=vocab_size,
        target_embed_dim=target_embed_dim,
        target_embed_dropout=dropout,
        lstm_hidden_size=lstm_hidden_size,
        attention_cfg=AdditiveAttentionConfig(attention_dim=attention_dim, att_weights_dropout=dropout),
        output_proj_dim=output_proj_dim,
        output_dropout=dropout,
        zoneout_drop_c=zoneout_drop_c,
        zoneout_drop_h=zoneout_drop_h,
        target_padding_idx=target_padding_idx,
    )
    return AttentionLSTMDecoderV1(decoder_cfg)


def test_additive_attention():
    cfg = AdditiveAttentionConfig(attention_dim=5, att_weights_dropout=0.1)
    att = AdditiveAttention(cfg)
    key = torch.rand((10, 20, 5))
    value = torch.rand((10, 20, 5))
    query = torch.rand((10, 5))

    enc_seq_len = torch.arange(start=10, end=20)  # [10, ..., 19]

    # pass key as weight feedback just for testing
    context, weights = att(key=key, value=value, query=query, weight_feedback=key, enc_seq_len=enc_seq_len)
    assert context.shape == (10, 5)
    assert weights.shape == (10, 20, 1)

    # Testing attention weights masking:
    # for first seq, the enc seq length is 10 so half the weights should be 0
    assert torch.eq(weights[0, 10:, 0], torch.tensor(0.0)).all()
    # test for other seqs
    assert torch.eq(weights[5, 15:, 0], torch.tensor(0.0)).all()


def test_encoder_decoder_attention_model():
    encoder = torch.rand((10, 20, 5))
    encoder_seq_len = torch.arange(start=10, end=20)  # [10, ..., 19]
    decoder = _make_decoder(dropout=0.1)
    target_labels = torch.randint(low=0, high=15, size=(10, 7))  # [B,N]

    decoder_logits, _ = decoder(encoder_outputs=encoder, labels=target_labels, enc_seq_len=encoder_seq_len)

    assert decoder_logits.shape == (10, 7, 15)


def test_zoneout_lstm_cell():
    encoder = torch.rand((10, 20, 5))
    encoder_seq_len = torch.arange(start=10, end=20)  # [10, ..., 19]
    target_labels = torch.randint(low=0, high=15, size=(10, 7))  # [B,N]

    def forward_decoder(zoneout_drop_c: float, zoneout_drop_h: float):
        decoder = _make_decoder(
            dropout=0.1,
            zoneout_drop_c=zoneout_drop_c,
            zoneout_drop_h=zoneout_drop_h,
        )
        decoder_logits, _ = decoder(encoder_outputs=encoder, labels=target_labels, enc_seq_len=encoder_seq_len)
        return decoder_logits

    decoder_logits = forward_decoder(zoneout_drop_c=0.15, zoneout_drop_h=0.05)
    assert decoder_logits.shape == (10, 7, 15)

    decoder_logits = forward_decoder(zoneout_drop_c=0.0, zoneout_drop_h=0.0)
    assert decoder_logits.shape == (10, 7, 15)


def test_decoder_step_matches_full_sequence_forward_without_internal_shift():
    torch.manual_seed(1)
    encoder = torch.rand((2, 5, 5))
    encoder_seq_len = torch.tensor([5, 3])
    history_labels = torch.tensor([[0, 1, 2, 3], [0, 4, 3, 2]])
    decoder = _make_decoder()
    decoder.eval()

    full_logits, _ = decoder(
        encoder_outputs=encoder,
        labels=history_labels,
        enc_seq_len=encoder_seq_len,
    )

    state = None
    step_logits = []
    for step in range(history_labels.size(1)):
        logits, state = decoder(
            encoder_outputs=encoder,
            labels=history_labels[:, step : step + 1],
            enc_seq_len=encoder_seq_len,
            state=state,
        )
        step_logits.append(logits)

    assert torch.allclose(full_logits, torch.cat(step_logits, dim=1), atol=1e-6)


def test_rasr_modules_match_decoder_forward_for_single_sequence_search():
    torch.manual_seed(2)
    encoder = torch.rand((1, 5, 5))
    encoder_seq_len = torch.tensor([4])
    history_labels = torch.tensor([[0, 1, 2, 3]])
    decoder = _make_decoder(target_padding_idx=0)
    decoder.eval()

    full_logits, _ = decoder(
        encoder_outputs=encoder,
        labels=history_labels,
        enc_seq_len=encoder_seq_len,
    )
    full_scores = -full_logits.log_softmax(dim=2)

    rasr_encoder = AttentionLSTMDecoderV1RasrEncoder(decoder)
    rasr_initializer = AttentionLSTMDecoderV1RasrStateInitializer(decoder)
    rasr_updater = AttentionLSTMDecoderV1RasrStateUpdater(decoder)
    rasr_scorer = AttentionLSTMDecoderV1RasrScorer(decoder)

    encoder_states = rasr_encoder(encoder)
    assert encoder_states.shape == (1, 5, decoder.encoder_dim + decoder.attention_dim + 1)

    token_embedding, lstm_state_h, lstm_state_c, att_context, accum_att_weights = rasr_initializer(
        encoder_states=encoder_states,
        encoder_states_size=encoder_seq_len,
    )
    assert token_embedding.shape == (1, decoder.target_embed_dim)
    assert torch.count_nonzero(token_embedding) == 0
    assert lstm_state_h.shape == (1, decoder.lstm_hidden_size)
    assert lstm_state_c.shape == (1, decoder.lstm_hidden_size)
    assert att_context.shape == (1, decoder.encoder_dim)
    assert accum_att_weights.shape == (1, encoder.size(1), 1)

    rasr_scores = [
        rasr_scorer(
            token_embedding=token_embedding,
            lstm_state_h=lstm_state_h,
            att_context=att_context,
        )
    ]
    for step in range(1, history_labels.size(1)):
        token_embedding, lstm_state_h, lstm_state_c, att_context, accum_att_weights = rasr_updater(
            encoder_states=encoder_states,
            encoder_states_size=encoder_seq_len,
            token=history_labels[:, step],
            lstm_state_h=lstm_state_h,
            lstm_state_c=lstm_state_c,
            att_context=att_context,
            accum_att_weights=accum_att_weights,
        )
        rasr_scores.append(
            rasr_scorer(
                token_embedding=token_embedding,
                lstm_state_h=lstm_state_h,
                att_context=att_context,
            )
        )

    assert torch.allclose(full_scores, torch.stack(rasr_scores, dim=1), atol=1e-6)


def test_rasr_state_updater_expands_single_encoder_state_for_active_hypotheses():
    torch.manual_seed(3)
    encoder = torch.rand((1, 5, 5))
    encoder_seq_len = torch.tensor([4])
    tokens = torch.tensor([1, 2, 3])
    decoder = _make_decoder()
    decoder.eval()

    encoder_states = AttentionLSTMDecoderV1RasrEncoder(decoder)(encoder)
    (
        token_embedding,
        lstm_state_h,
        lstm_state_c,
        att_context,
        accum_att_weights,
    ) = AttentionLSTMDecoderV1RasrStateInitializer(decoder)(
        encoder_states=encoder_states,
        encoder_states_size=encoder_seq_len,
    )

    (
        token_embedding,
        lstm_state_h,
        lstm_state_c,
        att_context,
        accum_att_weights,
    ) = AttentionLSTMDecoderV1RasrStateUpdater(decoder)(
        encoder_states=encoder_states,
        encoder_states_size=encoder_seq_len,
        token=tokens,
        lstm_state_h=lstm_state_h.expand(tokens.size(0), -1),
        lstm_state_c=lstm_state_c.expand(tokens.size(0), -1),
        att_context=att_context.expand(tokens.size(0), -1),
        accum_att_weights=accum_att_weights.expand(tokens.size(0), -1, -1),
    )

    assert torch.allclose(token_embedding, decoder.target_embed(tokens))
    assert lstm_state_h.shape == (tokens.size(0), decoder.lstm_hidden_size)
    assert lstm_state_c.shape == (tokens.size(0), decoder.lstm_hidden_size)
    assert att_context.shape == (tokens.size(0), decoder.encoder_dim)
    assert accum_att_weights.shape == (tokens.size(0), encoder.size(1), 1)


def test_zoneout_lstm_cell_uses_previous_state_for_wrapped_lstm_cell():
    torch.manual_seed(4)
    cell = nn.LSTMCell(input_size=3, hidden_size=4)
    zoneout_cell = ZoneoutLSTMCell(cell=cell, zoneout_h=0.0, zoneout_c=0.0)
    inputs = torch.rand((2, 3))
    state = (torch.rand((2, 4)), torch.rand((2, 4)))

    expected = cell(inputs, state)
    actual = zoneout_cell(inputs, state)

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])
