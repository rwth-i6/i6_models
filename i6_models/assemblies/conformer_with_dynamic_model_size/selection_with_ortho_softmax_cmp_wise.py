import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Optional

from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer_structure_prune import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerMHSAWithGateV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer_with_dynamic_model_size.stochastic_depth import StochasticDepth

EPSILON = np.finfo(np.float32).tiny


@dataclass
class ConformerBlockConfig(ModelConfiguration):
    """
    Attributes:
        ff_cfg: configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: configuration for ConformerMHSAV1
        conv_cfg: configuration for ConformerConvolutionV1
        layer_dropout: layer dropout value
        apply_ff_adaptive_dropout: if apply adaptive dropout to feed-forward layers
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: ConformerMHSAV1Config
    conv_cfg: ConformerConvolutionV1Config
    layer_dropout: Optional[float] = None
    apply_ff_adaptive_dropout: Optional[bool] = False


class ConformerBlock(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockConfig):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.cfg = cfg
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAWithGateV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.layer_dropout = None
        self.apply_ff_adaptive_dropout = cfg.apply_ff_adaptive_dropout
        if cfg.layer_dropout is not None:
            self.layer_dropout = StochasticDepth(p=cfg.layer_dropout, mode="row")
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        /,
        sequence_mask: torch.Tensor,
        apply_layer_dropout: torch.Tensor,
        module_gates: torch.Tensor,
        hard_prune=False,
    ) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :param apply_ff_adaptive_dropout: if apply adaptive dropout to feed-forward layers
        :param module_gates:  gates that element-wise multiplied with each module (layer)
        :param hard_prune:
            if hard prune is True, the module_gates will be binary and the layer with module_gates == 0 will be directly jumped
            if hard prune is False, the module_gates could be float number between range 0 and 1
        :return: torch.Tensor of shape [B, T, F]
        """
        if hard_prune and self.apply_ff_adaptive_dropout:
            self.ff1.dropout = self.cfg.ff_cfg.dropout * (sum(module_gates[0:4]) / 4)

        if hard_prune and sum(module_gates[:4]) == 0:
            x = x
        else:
            if apply_layer_dropout[0] is True and self.layer_dropout is not None:
                x = (
                    self.layer_dropout(0.5 * self.ff1(x, channel_chunk_gates=module_gates[0:4], hard_prune=hard_prune))
                    + x
                )
            else:
                x = 0.5 * self.ff1(x, channel_chunk_gates=module_gates[0:4], hard_prune=hard_prune) + x  # [B, T, F]

        if hard_prune and module_gates[4] == 0:
            x = x
        else:
            if apply_layer_dropout[1] is True and self.layer_dropout is not None:
                x = self.layer_dropout(self.conv(x) * module_gates[4]) + x  # [B, T, F]
            else:
                x = self.conv(x) * module_gates[4] + x  # [B, T, F]

        h = self.mhsa.mhsa.h
        if hard_prune and sum(module_gates[5 : 5 + h]) == 0:
            x = x
        else:
            if apply_layer_dropout[2] is True and self.layer_dropout is not None:
                x = (
                    self.layer_dropout(
                        self.mhsa(x, sequence_mask, head_gates=module_gates[5 : 5 + h], hard_prune=hard_prune)
                    )
                    + x
                )
            else:
                x = self.mhsa(x, sequence_mask, head_gates=module_gates[5 : 5 + h], hard_prune=hard_prune) + x

        if hard_prune and self.apply_ff_adaptive_dropout:
            self.ff2.dropout = self.cfg.ff_cfg.dropout * (sum(module_gates[5 + h : 9 + h]) / 4)

        if hard_prune and sum(module_gates[5 + h : 9 + h]) == 0:
            x = x
        else:
            if apply_layer_dropout[3] is True and self.layer_dropout is not None:
                x = (
                    self.layer_dropout(
                        0.5 * self.ff2(x, channel_chunk_gates=module_gates[5 + h : 9 + h], hard_prune=hard_prune)
                    )
                    + x
                )  # [B, T, F]
            else:
                x = (
                    0.5 * self.ff2(x, channel_chunk_gates=module_gates[5 + h : 9 + h], hard_prune=hard_prune) + x
                )  # [B, T, F]

        x = self.final_layer_norm(x)  # [B, T, F]
        return x


@dataclass
class ConformerEncoderConfig(ModelConfiguration):
    """
    Attributes:
        num_layers: number of conformer layers in the conformer encoder
        frontend: s pair of ConformerFrontend and corresponding config
        block_cfg: configuration for ConformerBlockV2
        pct_params_set: a predefined set of expected percentage of number of parameters that will be jointly train together
        softmax_kwargs: define the related argument for temperature anealling and orthogonal constraint, e.g.
            softmax_kwargs={
                "softmax_constraint_norm": "L2_norm",
                "initial_tau": 1,
                "min_tau": 0.01,
                "tau_annealing": 0.999992,
                "softmax_constraint_loss_scale": "linear_increase",
                "max_softmax_constraint_loss_scale": 1,
                "softmax_constraint_warmup_steps": 225000,
            },
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockConfig
    pct_params_set: List[float]
    softmax_kwargs: Dict[str, str | float]


class ConformerEncoder(nn.Module):
    """
    Conformer model encoders with dynamic size based on a supernet and M subnets that share parameters with the supernet,
    the training consists of two stages:
     - stage 1: use OrthoSoftmax to learn a binary mask for each subnet
     - stage 2: determine and fix the binary masks for all subnets, jointly train all networks efficiently with sandwich rule
    the subnet is selected component-wisely from the supernet

    """

    def __init__(self, cfg: ConformerEncoderConfig):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.pct_params_set = sorted(cfg.pct_params_set)
        self.max_pct = max(cfg.pct_params_set)
        self.min_pct = min(cfg.pct_params_set)
        self.module_list = torch.nn.ModuleList([ConformerBlock(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.softmax_kwargs = cfg.softmax_kwargs
        self.layer_gates = torch.nn.Parameter(
            torch.FloatTensor(torch.zeros((cfg.num_layers * 15, cfg.num_layers * 15)))
        )
        self.layer_gates.data.normal_(EPSILON, 0.00)
        self.register_parameter(
            "selected_mod_indices",
            nn.Parameter(torch.tensor([[-1] * len(self.layer_gates)] * len(self.pct_params_set)), requires_grad=False),
        )
        self.recog_param_pct = 1
        self.random_idx = None

    def forward(
        self,
        data_tensor: torch.Tensor,
        /,
        sequence_mask: torch.Tensor,
        global_train_step: int,
        stage_1_global_steps: int,
        params_kwargs: dict,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :param global_train_step: the index of the global train step
        :param stage_1_global_steps: global train steps for stage 1
        :param params_kwargs: define the parameter info
            {
                "num_params": number of parameters for each decomposed component
                "rest_params": rest parameters including front end and final linear layer
                "total_params": total params
            }

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        softmax_constraint = 0

        outputs = [x for _ in range(len(self.pct_params_set))]

        # in training
        if self.training:
            # Stage 1: jointly train one large model and one small model
            if global_train_step <= stage_1_global_steps:
                pct = np.random.choice(self.pct_params_set[:-1])
                print("current pct", pct)
                tau = max(
                    self.softmax_kwargs["initial_tau"] * self.softmax_kwargs["tau_annealing"] ** global_train_step,
                    self.softmax_kwargs["min_tau"],
                )
                gumbel_softmax = torch.nn.functional.softmax(self.layer_gates / tau, dim=1)
                num_params = torch.tensor(params_kwargs["num_params"]).to(gumbel_softmax.device)
                rest_params = torch.tensor(params_kwargs["rest_params"]).to(gumbel_softmax.device)
                num_params_cum_sum = torch.cumsum(torch.sum(num_params * gumbel_softmax, dim=1), dim=0) + rest_params
                small_model_params = torch.tensor(params_kwargs["total_params"]) * pct
                k = abs(num_params_cum_sum - small_model_params).argmin(dim=-1) + 1
                gumbel_softmax = gumbel_softmax[:k]

                gumbel_softmax_matmal = torch.matmul(gumbel_softmax, torch.transpose(gumbel_softmax, 0, 1))
                if self.softmax_kwargs["softmax_constraint_norm"] == "L2_norm_sqrt":
                    softmax_constraint = torch.sqrt(
                        torch.sum(torch.square(torch.triu(gumbel_softmax_matmal, diagonal=1)))
                        + torch.sum(torch.square(torch.diagonal(gumbel_softmax_matmal) - 1))
                    )
                elif self.softmax_kwargs["softmax_constraint_norm"] == "L2_norm":
                    softmax_constraint = torch.sum(
                        torch.square(torch.triu(gumbel_softmax_matmal, diagonal=1))
                    ) + torch.sum(torch.square(torch.diagonal(gumbel_softmax_matmal) - 1))
                elif self.softmax_kwargs["softmax_constraint_norm"] == "L1_norm":
                    softmax_constraint = torch.sum(
                        torch.abs(torch.triu(gumbel_softmax_matmal, diagonal=1))
                    ) + torch.sum(torch.abs(torch.diagonal(gumbel_softmax_matmal) - 1))

                module_gates = torch.sum(gumbel_softmax, dim=0)
                selected_mod_indices = torch.argmax(gumbel_softmax, dim=1)

                # large model
                for i in range(len(self.module_list)):
                    outputs[-1] = self.module_list[i](
                        outputs[-1],
                        sequence_mask,
                        apply_layer_dropout=[False, False, False, False],
                        module_gates=torch.tensor([1] * 15),
                        hard_prune=True,
                    )

                # small model
                for i in range(len(self.module_list)):
                    outputs[0] = self.module_list[i](
                        outputs[0],
                        sequence_mask,
                        apply_layer_dropout=[False, False, False, False],
                        module_gates=module_gates[i * 15 : (i + 1) * 15],
                        hard_prune=False,
                    )

                if global_train_step == int(stage_1_global_steps):
                    for i in range(len(self.pct_params_set)):
                        p = self.pct_params_set[i]
                        if p == self.pct_params_set[-1]:
                            self.selected_mod_indices[-1][:] = torch.tensor(list(range(len(self.layer_gates))))
                        else:
                            gumbel_softmax = torch.nn.functional.softmax(self.layer_gates / tau, dim=1)
                            num_params = torch.tensor(params_kwargs["num_params"]).to(gumbel_softmax.device)
                            rest_params = torch.tensor(params_kwargs["rest_params"]).to(gumbel_softmax.device)
                            num_params_cum_sum = (
                                torch.cumsum(torch.sum(num_params * gumbel_softmax, dim=1), dim=0) + rest_params
                            )
                            small_model_params = torch.tensor(params_kwargs["total_params"]) * p
                            k = abs(num_params_cum_sum - small_model_params).argmin(dim=-1) + 1
                            gumbel_softmax = gumbel_softmax[:k]
                            selected_indices = sorted(list(torch.argmax(gumbel_softmax, dim=1)))
                            self.selected_mod_indices[i][: len(selected_indices)] = torch.tensor(selected_indices)
                    print(self.selected_mod_indices)

                if global_train_step % 200 == 0:
                    print(f"small model components: {k}")
                    print("layer_gates: {}".format(self.layer_gates))
                    print("module_gates: {}".format(module_gates))
                    print("selected_mod_indices: {}".format(sorted([int(i + 1) for i in selected_mod_indices])))

            # Stage 2: fix the selection and jointly train
            else:
                smallest_model_indices = [int(i) for i in self.selected_mod_indices[0] if i != -1]

                print("smallest_model_indices", smallest_model_indices)
                # largest model
                module_gates = torch.tensor([1] * len(self.layer_gates))

                for i in range(len(self.module_list)):
                    apply_layer_dropout = []
                    for l, r in [(0, 4), (4, 5), (5, 11), (11, 15)]:
                        if set(smallest_model_indices).isdisjoint([15 * i + j for j in range(l, r)]):
                            apply_layer_dropout.append(True)
                        else:
                            apply_layer_dropout.append(False)

                    outputs[-1] = self.module_list[i](
                        outputs[-1],
                        sequence_mask,
                        apply_layer_dropout=apply_layer_dropout,
                        module_gates=module_gates[15 * i : 15 * i + 15],
                        hard_prune=True,
                    )

                # smallest model
                for i in range(len(self.module_list)):
                    gates = []
                    for j in range(15):
                        if 15 * i + j not in smallest_model_indices:
                            gates.append(0)
                        else:
                            gates.append(1)
                    outputs[0] = self.module_list[i](
                        outputs[0],
                        sequence_mask,
                        apply_layer_dropout=[False, False, False, False],
                        module_gates=torch.tensor(gates),
                        hard_prune=True,
                    )

                # medium model
                if len(self.pct_params_set) > 2:
                    random_idx = np.random.choice(list(range(len(self.pct_params_set))[1:-1]), 1)[0]
                    self.random_idx = random_idx
                    medium_mod_indices = [int(i) for i in self.selected_mod_indices[random_idx] if i != -1]
                    print("medium_mod_indices", medium_mod_indices)

                    for i in range(len(self.module_list)):
                        gates = []
                        for j in range(15):
                            if 15 * i + j not in medium_mod_indices:
                                gates.append(0)
                            else:
                                gates.append(1)
                        outputs[self.random_idx] = self.module_list[i](
                            outputs[self.random_idx],
                            sequence_mask,
                            apply_layer_dropout=[False, False, False, False],
                            module_gates=torch.tensor(gates),
                            hard_prune=True,
                        )

                if global_train_step % 200 == 0:
                    print(f"selected_mod_indices: {self.selected_mod_indices}")

        # in recognition
        else:
            idx = self.pct_params_set.index(self.recog_param_pct)
            selected_mod_indices = [int(i) for i in self.selected_mod_indices[idx] if i != -1]

            for i in range(len(self.module_list)):
                gates = []
                for j in range(15):
                    if 15 * i + j not in selected_mod_indices:
                        gates.append(0)
                    else:
                        gates.append(1)

                outputs[idx] = self.module_list[i](
                    outputs[idx],
                    sequence_mask,
                    apply_layer_dropout=torch.tensor([False] * 4),
                    module_gates=torch.tensor(gates),
                    hard_prune=True,
                )

        return outputs, softmax_constraint, sequence_mask
