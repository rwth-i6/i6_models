from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
from torch import nn

from i6_models.config import ModelConfiguration

from .zoneout_lstm import ZoneoutLSTMCell


@dataclass
class AdditiveAttentionConfig(ModelConfiguration):
    """
    Attributes:
        attention_dim: Dimension of the common attention space.
        att_weights_dropout: Dropout applied to attention weights after the softmax.
    """

    attention_dim: int
    att_weights_dropout: float


class AdditiveAttention(nn.Module):
    """
    Single-head additive attention with optional weight-feedback features.

    This computes:
        energies = v^T tanh(key + query + weight_feedback)
        weights = softmax(energies)
        context = sum_t weights_t value_t
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
        :param key: Projected encoder keys, shape [B, T, A].
        :param value: Encoder values used for the weighted sum, shape [B, T, E].
        :param query: Projected decoder query, shape [B, A].
        :param weight_feedback: Projected accumulated attention weights, shape [B, T, A].
        :param enc_seq_len: Encoder sequence lengths before padding, shape [B].
        :return: Tuple of attention context [B, E] and attention weights [B, T, 1].
        """
        # all inputs are already projected
        energies = self.linear(
            (key + query.unsqueeze(1) + weight_feedback).tanh()
        )  # [B,T,1]
        time_arange = torch.arange(energies.size(1), device=energies.device)  # [T]
        seq_len_mask = torch.less(time_arange[None, :], enc_seq_len[:, None])  # [B, T]
        energies = torch.where(
            seq_len_mask.unsqueeze(2), energies, energies.new_tensor(-float("inf"))
        )
        weights = nn.functional.softmax(energies, dim=1)  # [B,T,1]
        weights = self.att_weights_drop(weights)
        context = torch.bmm(weights.transpose(1, 2), value)  # [B,1,E]
        context = context.reshape(context.size(0), -1)  # [B,E]
        return context, weights


@dataclass
class AttentionLSTMDecoderV1Config(ModelConfiguration):
    """
    Attributes:
        encoder_dim: Encoder output dimension E.
        vocab_size: Number of decoder output labels.
        target_embed_dim: Target history embedding dimension M.
        target_embed_dropout: Dropout applied to target history embeddings.
        lstm_hidden_size: Decoder LSTM hidden dimension H.
        zoneout_drop_h: Zoneout drop probability for hidden state h.
        zoneout_drop_c: Zoneout drop probability for cell state c.
        attention_cfg: Additive attention configuration.
        output_proj_dim: Readout projection dimension before maxout. Must be even.
        output_dropout: Dropout applied to the maxout readout.
        target_padding_idx: Optional target label index whose embedding stays zero and receives no gradient.
    """

    encoder_dim: int
    vocab_size: int
    target_embed_dim: int
    target_embed_dropout: float
    lstm_hidden_size: int
    zoneout_drop_h: float
    zoneout_drop_c: float
    attention_cfg: AdditiveAttentionConfig
    output_proj_dim: int
    output_dropout: float
    target_padding_idx: Optional[int] = None


LstmState = Tuple[torch.Tensor, torch.Tensor]
DecoderState = Tuple[LstmState, torch.Tensor, torch.Tensor]


class AttentionLSTMDecoderV1(nn.Module):
    """
    Single-headed Attention decoder with additive attention mechanism.
    """

    def __init__(self, cfg: AttentionLSTMDecoderV1Config):
        super().__init__()

        self.encoder_dim = cfg.encoder_dim
        self.vocab_size = cfg.vocab_size
        self.target_embed_dim = cfg.target_embed_dim
        self.lstm_hidden_size = cfg.lstm_hidden_size
        self.attention_dim = cfg.attention_cfg.attention_dim

        self.target_embed = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.target_embed_dim,
            padding_idx=cfg.target_padding_idx,
        )
        self.target_embed_dropout = nn.Dropout(cfg.target_embed_dropout)

        lstm_cell = nn.LSTMCell(
            input_size=cfg.target_embed_dim + cfg.encoder_dim,
            hidden_size=cfg.lstm_hidden_size,
        )
        # if zoneout drop probs are 0, then it is equivalent to normal LSTMCell
        self.s = ZoneoutLSTMCell(
            cell=lstm_cell,
            zoneout_h=cfg.zoneout_drop_h,
            zoneout_c=cfg.zoneout_drop_c,
        )

        self.s_transformed = nn.Linear(
            cfg.lstm_hidden_size, self.attention_dim, bias=False
        )  # query

        # for attention
        self.enc_ctx = nn.Linear(cfg.encoder_dim, self.attention_dim)
        self.attention = AdditiveAttention(cfg.attention_cfg)

        # for weight feedback
        self.inv_fertility = nn.Linear(
            cfg.encoder_dim, 1, bias=False
        )  # followed by sigmoid
        self.weight_feedback = nn.Linear(1, self.attention_dim, bias=False)

        self.readout_in = nn.Linear(
            cfg.lstm_hidden_size + cfg.target_embed_dim + cfg.encoder_dim,
            cfg.output_proj_dim,
        )
        assert (
            cfg.output_proj_dim % 2 == 0
        ), "output projection dimension must be even for the MaxOut op of 2 pieces"
        self.output = nn.Linear(cfg.output_proj_dim // 2, cfg.vocab_size)
        self.output_dropout = nn.Dropout(cfg.output_dropout)

    def get_initial_state(self, encoder_outputs: torch.Tensor) -> DecoderState:
        """
        :param encoder_outputs: Encoder outputs, shape [B, T, E].
        :return:
            Initial decoder state consisting of LSTM state (h [B, H], c [B, H]),
            attention context [B, E], and accumulated attention weights [B, T, 1].
        """
        batch_size = encoder_outputs.size(0)
        max_time = encoder_outputs.size(1)
        zeros = encoder_outputs.new_zeros((batch_size, self.lstm_hidden_size))
        lstm_state = (zeros, zeros)
        att_context = encoder_outputs.new_zeros((batch_size, encoder_outputs.size(2)))
        accum_att_weights = encoder_outputs.new_zeros((batch_size, max_time, 1))
        return lstm_state, att_context, accum_att_weights

    def _get_history_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        """
        :param labels: Decoder history labels, shape [B, U].
        :return: Target history embeddings, shape [B, U, M].
        """
        label_embeddings = self.target_embed(labels)  # [B,U,M]
        return self.target_embed_dropout(label_embeddings)

    def _decode_step(
        self,
        *,
        history_embedding: torch.Tensor,
        lstm_state: LstmState,
        att_context: torch.Tensor,
        accum_att_weights: torch.Tensor,
        encoder_outputs: torch.Tensor,
        enc_ctx: torch.Tensor,
        enc_inv_fertility: torch.Tensor,
        enc_seq_len: torch.Tensor,
    ) -> Tuple[LstmState, torch.Tensor, torch.Tensor, torch.Tensor]:
        lstm_state = self.s(
            torch.cat([history_embedding, att_context], dim=-1), lstm_state
        )
        lstm_out = lstm_state[0]
        query = self.s_transformed(lstm_out)

        weight_feedback = self.weight_feedback(accum_att_weights)
        att_context, att_weights = self.attention(
            key=enc_ctx,
            value=encoder_outputs,
            query=query,
            weight_feedback=weight_feedback,
            enc_seq_len=enc_seq_len,
        )
        accum_att_weights = accum_att_weights + att_weights * enc_inv_fertility * 0.5
        return lstm_state, att_context, accum_att_weights, lstm_out

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        labels: torch.Tensor,
        enc_seq_len: torch.Tensor,
        state: Optional[DecoderState] = None,
    ) -> Tuple[torch.Tensor, DecoderState]:
        """
        Run decoder steps for externally prepared history labels.

        :param encoder_outputs: Encoder outputs, shape [B, T, E]. Same for training and search.
        :param labels:
            Decoder history labels, shape [B, N]. For teacher forcing, prepend a begin
            label externally and drop the final target label before calling this module.
        :param enc_seq_len: Encoder sequence lengths before padding, shape [B].
        :param state:
            Previous decoder state. If None, decoding starts from zero LSTM state, zero attention context,
            and zero accumulated attention weights.
        :return:
            Tuple of decoder logits [B, N, vocab_size] or [B, 1, vocab_size] and the final decoder state:
            LSTM state (h [B, H], c [B, H]), attention context [B, E], accumulated attention weights [B, T, 1].
        """
        if encoder_outputs.dim() != 3:
            raise ValueError(
                f"encoder_outputs must have shape [B, T, E], got {encoder_outputs.shape}"
            )
        if labels.dim() != 2:
            raise ValueError(f"labels must have shape [B, U], got {labels.shape}")
        if enc_seq_len.dim() != 1:
            raise ValueError(
                f"enc_seq_len must have shape [B], got {enc_seq_len.shape}"
            )
        if encoder_outputs.size(0) != labels.size(0) or encoder_outputs.size(
            0
        ) != enc_seq_len.size(0):
            raise ValueError(
                "encoder_outputs, labels, and enc_seq_len must have the same batch size, got "
                f"{encoder_outputs.size(0)}, {labels.size(0)}, and {enc_seq_len.size(0)}"
            )

        if state is None:
            lstm_state, att_context, accum_att_weights = self.get_initial_state(
                encoder_outputs
            )
        else:
            lstm_state, att_context, accum_att_weights = state

        history_embeddings = self._get_history_embeddings(labels)

        enc_ctx = self.enc_ctx(encoder_outputs)  # [B,T,A]
        enc_inv_fertility = self.inv_fertility(encoder_outputs).sigmoid()  # [B,T,1]

        num_steps = labels.size(1)  # N

        # collect for computing later the decoder logits outside the loop
        s_list = []
        att_context_list = []

        # decoder loop
        for step in range(num_steps):
            history_embedding = history_embeddings[:, step, :]  # [B,M]
            lstm_state, att_context, accum_att_weights, lstm_out = self._decode_step(
                history_embedding=history_embedding,
                lstm_state=lstm_state,
                att_context=att_context,
                accum_att_weights=accum_att_weights,
                encoder_outputs=encoder_outputs,
                enc_ctx=enc_ctx,
                enc_inv_fertility=enc_inv_fertility,
                enc_seq_len=enc_seq_len,
            )
            s_list.append(lstm_out)
            att_context_list.append(att_context)

        # output layer
        s_stacked = torch.stack(s_list, dim=1)  # [B,N,H]
        att_context_stacked = torch.stack(att_context_list, dim=1)  # [B,N,E]
        readout_in = self.readout_in(
            torch.cat([s_stacked, history_embeddings, att_context_stacked], dim=-1)
        )

        # maxout layer
        readout_in = readout_in.view(
            readout_in.size(0), readout_in.size(1), -1, 2
        )  # [B,N,D/2,2]
        readout, _ = torch.max(readout_in, dim=-1)  # [B,N,D/2]

        readout_drop = self.output_dropout(readout)
        decoder_logits = self.output(readout_drop)

        state = lstm_state, att_context, accum_att_weights

        return decoder_logits, state


def _split_rasr_encoder_state(
    decoder: AttentionLSTMDecoderV1, encoder_states: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param decoder: Decoder whose encoder/attention dimensions define the packed layout.
    :param encoder_states: Packed encoder state, shape [B, T, E + A + 1].
    :return:
        Encoder outputs [B, T, E], projected encoder keys [B, T, A],
        and inverse fertility [B, T, 1].
    """
    return cast(
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        encoder_states.split(
            [decoder.encoder_dim, decoder.attention_dim, 1],
            dim=2,
        ),
    )


class AttentionLSTMDecoderV1RasrEncoder(nn.Module):
    """
    Pack encoder outputs with decoder-side precomputations for RASR stateful ONNX scoring.
    """

    def __init__(self, decoder: AttentionLSTMDecoderV1):
        super().__init__()
        self.decoder = decoder

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        :param encoder_outputs: Encoder outputs, shape [B, T, E].
        :return: Packed encoder state, shape [B, T, E + A + 1].
        """
        enc_ctx = self.decoder.enc_ctx(encoder_outputs)  # [B,T,A]
        enc_inv_fertility = self.decoder.inv_fertility(
            encoder_outputs
        ).sigmoid()  # [B,T,1]
        return torch.cat([encoder_outputs, enc_ctx, enc_inv_fertility], dim=2)


class AttentionLSTMDecoderV1RasrScorer(nn.Module):
    """
    RASR stateful ONNX scorer for one decoder state.
    """

    def __init__(self, decoder: AttentionLSTMDecoderV1):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        token_embedding: torch.Tensor,
        lstm_state_h: torch.Tensor,
        att_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param token_embedding: Embedding of the history token for this state, shape [B, M].
        :param lstm_state_h: Decoder LSTM hidden state, shape [B, H].
        :param att_context: Attention context, shape [B, E].
        :return: Negative log-probability scores, shape [B, vocab_size].
        """
        readout_in = self.decoder.readout_in(
            torch.cat([lstm_state_h, token_embedding, att_context], dim=1)
        )  # [B, D]

        readout_in = readout_in.view(readout_in.size(0), -1, 2)  # [B,D/2,2]
        readout, _ = torch.max(readout_in, dim=2)  # [B,D/2]

        decoder_logits = self.decoder.output(readout)  # [B,V]
        return -decoder_logits.log_softmax(dim=1)  # [B,V]


class AttentionLSTMDecoderV1RasrStateInitializer(nn.Module):
    """
    RASR stateful ONNX state initializer for one encoded sequence.
    """

    def __init__(self, decoder: AttentionLSTMDecoderV1):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_states_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param encoder_states: Packed encoder state, shape [1, T, E + A + 1].
        :param encoder_states_size: Encoder sequence lengths, shape [1].
        :return:
            token_embedding [1, M], lstm_state_h [1, H], lstm_state_c [1, H],
            att_context [1, E], accum_att_weights [1, T, 1].
        """
        encoder_outputs, enc_ctx, enc_inv_fertility = _split_rasr_encoder_state(
            self.decoder, encoder_states
        )

        lstm_state, att_context, accum_att_weights = self.decoder.get_initial_state(
            encoder_outputs
        )
        token_embedding = encoder_outputs.new_zeros(
            encoder_outputs.size(0), self.decoder.target_embed_dim
        )

        lstm_state, att_context, accum_att_weights, _ = self.decoder._decode_step(
            history_embedding=token_embedding,
            lstm_state=lstm_state,
            att_context=att_context,
            accum_att_weights=accum_att_weights,
            encoder_outputs=encoder_outputs,
            enc_ctx=enc_ctx,
            enc_inv_fertility=enc_inv_fertility,
            enc_seq_len=encoder_states_size,
        )

        return (
            token_embedding,
            lstm_state[0],
            lstm_state[1],
            att_context,
            accum_att_weights,
        )


class AttentionLSTMDecoderV1RasrStateUpdater(nn.Module):
    """
    RASR stateful ONNX state updater for active search hypotheses.
    """

    def __init__(self, decoder: AttentionLSTMDecoderV1):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_states_size: torch.Tensor,
        token: torch.Tensor,
        lstm_state_h: torch.Tensor,
        lstm_state_c: torch.Tensor,
        att_context: torch.Tensor,
        accum_att_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param encoder_states: Packed encoder state, shape [1, T, E + A + 1].
        :param encoder_states_size: Encoder sequence lengths, shape [1].
        :param token: Accepted history token for each active hypothesis, shape [B].
        :param lstm_state_h: Previous LSTM hidden state, shape [B, H].
        :param lstm_state_c: Previous LSTM cell state, shape [B, H].
        :param att_context: Previous attention context, shape [B, E].
        :param accum_att_weights: Previous accumulated attention weights, shape [B, T, 1].
        :return:
            token_embedding [B, M], lstm_state_h [B, H], lstm_state_c [B, H],
            att_context [B, E], accum_att_weights [B, T, 1].
        """
        batch_size = token.size(0)
        encoder_states = encoder_states.expand(batch_size, -1, -1)
        encoder_states_size = encoder_states_size.expand(batch_size)
        encoder_outputs, enc_ctx, enc_inv_fertility = _split_rasr_encoder_state(
            self.decoder, encoder_states
        )

        token_embedding = self.decoder.target_embed(token)  # [B,M]

        lstm_state, att_context, accum_att_weights, _ = self.decoder._decode_step(
            history_embedding=token_embedding,
            lstm_state=(lstm_state_h, lstm_state_c),
            att_context=att_context,
            accum_att_weights=accum_att_weights,
            encoder_outputs=encoder_outputs,
            enc_ctx=enc_ctx,
            enc_inv_fertility=enc_inv_fertility,
            enc_seq_len=encoder_states_size,
        )

        return (
            token_embedding,
            lstm_state[0],
            lstm_state[1],
            att_context,
            accum_att_weights,
        )
