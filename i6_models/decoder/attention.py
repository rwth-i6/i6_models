from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class AdditiveAttentionConfig:
    """
    Attributes:
        attention_dim: attention dimension
        att_weights_dropout: attention weights dropout
    """

    attention_dim: int
    att_weights_dropout: float


class AdditiveAttention(nn.Module):
    """
    Additive attention mechanism. This is defined as:
        energies = v^T * tanh(h + s + beta)  where beta is weight feedback information
        weights = softmax(energies)
        context = weights * h
    """

    def __init__(self, cfg: AdditiveAttentionConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.attention_dim, 1, bias=False)
        self.att_weights_drop = nn.Dropout(cfg.att_weights_dropout)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        weight_feedback: torch.Tensor,
        enc_seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param key: encoder keys of shape [B,T,D_k]
        :param value: encoder values of shape [B,T,D_v]
        :param query: query of shape [B,D_k]
        :param weight_feedback: shape is [B,T,D_k]
        :param enc_seq_len: [B]
        :return: context [B,1,D_v], weights [B,T,1]
        """

        # all inputs are already projected
        energies = self.linear(nn.functional.tanh(key + query.unsqueeze(1) + weight_feedback))  # [B,T,1]
        time_arange = torch.arange(energies.size(1))  # [T]
        seq_len_mask = torch.less(time_arange[None, :], enc_seq_len[:, None])  # [B,T]
        energies = torch.where(seq_len_mask.unsqueeze(2), energies, torch.tensor(-float("inf")))
        weights = nn.functional.softmax(energies, dim=1)  # [B,T,1]
        weights = self.att_weights_drop(weights)
        context = torch.bmm(weights.transpose(1, 2), value)  # [B,1,D_v]
        context = context.reshape(context.size(0), -1)  # [B,D_v]
        return context, weights


@dataclass
class AttentionLstmDecoderV1Config:
    """
    Attributes:
        encoder_dim: encoder dimension
        vocab_size: vocabulary size
        target_embed_dim: embedding dimension
        target_embed_dropout: embedding dropout
        lstm_hidden_size: LSTM hidden size
        attention_cfg: attention config
        output_proj_dim: output projection dimension
        output_dropout: output dropout
    """

    encoder_dim: int
    vocab_size: int
    target_embed_dim: int
    target_embed_dropout: float
    lstm_hidden_size: int
    attention_cfg: AdditiveAttentionConfig
    output_proj_dim: int
    output_dropout: float


class AttentionLstmDecoderV1(nn.Module):
    """
    Single-headed Attention decoder with additive attention mechanism.
    """

    def __init__(self, cfg: AttentionLstmDecoderV1Config):
        super().__init__()

        self.target_embed = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.target_embed_dim)
        self.target_embed_dropout = nn.Dropout(cfg.target_embed_dropout)

        self.s = nn.LSTMCell(
            input_size=cfg.target_embed_dim + cfg.encoder_dim,
            hidden_size=cfg.lstm_hidden_size,
        )
        self.s_transformed = nn.Linear(cfg.lstm_hidden_size, cfg.attention_cfg.attention_dim, bias=False)  # query

        # for attention
        self.enc_ctx = nn.Linear(cfg.encoder_dim, cfg.attention_cfg.attention_dim)
        self.attention = AdditiveAttention(cfg.attention_cfg)

        # for weight feedback
        self.inv_fertility = nn.Linear(cfg.encoder_dim, 1, bias=False)  # followed by sigmoid
        self.weight_feedback = nn.Linear(1, cfg.attention_cfg.attention_dim, bias=False)

        self.readout_in = nn.Linear(cfg.lstm_hidden_size + cfg.target_embed_dim + cfg.encoder_dim, cfg.output_proj_dim)
        self.output = nn.Linear(cfg.output_proj_dim // 2, cfg.vocab_size)
        self.output_dropout = nn.Dropout(cfg.output_dropout)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        labels: torch.Tensor,
        enc_seq_len: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        """
        :param encoder_outputs: encoder outputs of shape [B,T,D]
        :param labels: labels of shape [B,T]
        :param enc_seq_len: encoder sequence lengths of shape [B,T]
        :param state: decoder state
        """
        if state is None:
            lstm_state = None
            att_context = torch.zeros((encoder_outputs.size(0), encoder_outputs.size(2)))
            accum_att_weights = encoder_outputs.new_zeros((encoder_outputs.size(0), encoder_outputs.size(1), 1))
        else:
            lstm_state, att_context, accum_att_weights = state

        target_embeddings = self.target_embed(labels)  # [B,N,D]
        target_embeddings = self.target_embed_dropout(target_embeddings)
        # pad for BOS and remove last token as this represents history and last token is not used
        target_embeddings = nn.functional.pad(target_embeddings, (0, 0, 1, 0), value=0)[:, :-1, :]  # [B,N,D]

        enc_ctx = self.enc_ctx(encoder_outputs)  # [B,T,D]
        enc_inv_fertility = nn.functional.sigmoid(self.inv_fertility(encoder_outputs))  # [B,T,1]

        num_steps = labels.size(1)  # N

        # collect for computing later the decoder logits outside the loop
        s_list = []
        att_context_list = []

        # decoder loop
        for step in range(num_steps):
            target_embed = target_embeddings[:, step, :]  # [B,D]

            lstm_state = self.s(torch.cat([target_embed, att_context], dim=-1), lstm_state)
            lstm_out = lstm_state[0]
            s_transformed = self.s_transformed(lstm_out)  # project query
            s_list.append(lstm_out)

            # attention mechanism
            weight_feedback = self.weight_feedback(accum_att_weights)
            att_context, att_weights = self.attention(
                key=enc_ctx,
                value=encoder_outputs,
                query=s_transformed,
                weight_feedback=weight_feedback,
                enc_seq_len=enc_seq_len,
            )
            att_context_list.append(att_context)
            accum_att_weights = accum_att_weights + att_weights * enc_inv_fertility * 0.5

        # output layer
        s_stacked = torch.stack(s_list, dim=1)  # [B,N,D]
        att_context_stacked = torch.stack(att_context_list, dim=1)  # [B,N,D]
        readout_in = self.readout_in(torch.cat([s_stacked, target_embeddings, att_context_stacked], dim=-1))  # [B,N,D]

        # maxout layer
        assert readout_in.size(-1) % 2 == 0
        readout_in = readout_in.view(readout_in.size(0), readout_in.size(1), -1, 2)  # [B,N,D/2,2]
        readout, _ = torch.max(readout_in, dim=-1)  # [B,N,D/2]

        output = self.output(readout)
        decoder_logits = self.output_dropout(output)

        state = lstm_state, att_context, accum_att_weights

        return decoder_logits, state
