import torch
from torch import nn

from i6_models.decoder.attention import AdditiveAttention, AdditiveAttentionConfig
from i6_models.decoder.attention import AttentionLSTMDecoderV1, AttentionLSTMDecoderV1Config


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
    decoder_cfg = AttentionLSTMDecoderV1Config(
        encoder_dim=5,
        vocab_size=15,
        target_embed_dim=3,
        target_embed_dropout=0.1,
        lstm_hidden_size=12,
        attention_cfg=AdditiveAttentionConfig(attention_dim=10, att_weights_dropout=0.1),
        output_proj_dim=12,
        output_dropout=0.1,
        zoneout_drop_c=0.0,
        zoneout_drop_h=0.0,
    )
    decoder = AttentionLSTMDecoderV1(decoder_cfg)
    target_labels = torch.randint(low=0, high=15, size=(10, 7))  # [B,N]

    decoder_logits, _ = decoder(encoder_outputs=encoder, labels=target_labels, enc_seq_len=encoder_seq_len)

    assert decoder_logits.shape == (10, 7, 15)


def test_zoneout_lstm_cell():
    encoder = torch.rand((10, 20, 5))
    encoder_seq_len = torch.arange(start=10, end=20)  # [10, ..., 19]
    target_labels = torch.randint(low=0, high=15, size=(10, 7))  # [B,N]

    def forward_decoder(zoneout_drop_c: float, zoneout_drop_h: float):
        decoder_cfg = AttentionLSTMDecoderV1Config(
            encoder_dim=5,
            vocab_size=15,
            target_embed_dim=3,
            target_embed_dropout=0.1,
            lstm_hidden_size=12,
            attention_cfg=AdditiveAttentionConfig(attention_dim=10, att_weights_dropout=0.1),
            output_proj_dim=12,
            output_dropout=0.1,
            zoneout_drop_c=zoneout_drop_c,
            zoneout_drop_h=zoneout_drop_h,
        )
        decoder = AttentionLSTMDecoderV1(decoder_cfg)
        decoder_logits, _ = decoder(encoder_outputs=encoder, labels=target_labels, enc_seq_len=encoder_seq_len)
        return decoder_logits

    decoder_logits = forward_decoder(zoneout_drop_c=0.15, zoneout_drop_h=0.05)
    assert decoder_logits.shape == (10, 7, 15)

    decoder_logits = forward_decoder(zoneout_drop_c=0.0, zoneout_drop_h=0.0)
    assert decoder_logits.shape == (10, 7, 15)
