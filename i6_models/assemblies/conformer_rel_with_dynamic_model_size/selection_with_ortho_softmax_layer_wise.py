from __future__ import annotations

__all__ = ["ConformerRelPosBlockV1Config", "ConformerRelPosBlockV1", "ConformerRelPosEncoderV1Config", "ConformerRelPosEncoderV1"]

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

import numpy as np
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV2,
    ConformerConvolutionV2Config,
    ConformerMHSARelPosV1,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.parts.conformer_with_dynamic_model_size.stochastic_depth import StochasticDepth

EPSILON = np.finfo(np.float32).tiny


@dataclass
class ConformerRelPosBlockV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for ConformerMHSAV1
        conv_cfg: Configuration for ConformerConvolutionV1
        layer_dropout: Dropout value to apply layer dropout
        modules: List of modules to use for ConformerBlockV2,
            "ff" for feed forward module, "mhsa" for multi-head self attention module, "conv" for conv module
        scales: List of scales to apply to the module outputs before the residual connection
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV2Config
    mhsa_cfg: ConformerMHSARelPosV1Config
    conv_cfg: ConformerConvolutionV2Config
    layer_dropout: float
    modules: List[str] = field(default_factory=lambda: ["ff", "mhsa", "conv", "ff"])
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])

    def __post__init__(self):
        super().__post_init__()
        assert len(self.modules) == len(self.scales), "modules and scales must have same length"
        for module_name in self.modules:
            assert module_name in ["ff", "mhsa", "conv"], "module not supported"


class ConformerRelPosBlockV1(nn.Module):
    """
    Conformre block
    - In soft prune mode, each module output can be multiple with a layer gate
    - In hard prune model, the modules which are not selected can be directly jumped
    """

    def __init__(self, cfg: ConformerRelPosBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()

        modules = []
        for module_name in cfg.modules:
            if module_name == "ff":
                modules.append(ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg))
            elif module_name == "mhsa":
                modules.append(ConformerMHSARelPosV1(cfg=cfg.mhsa_cfg))
            elif module_name == "conv":
                modules.append(ConformerConvolutionV2(model_cfg=cfg.conv_cfg))
            else:
                raise NotImplementedError

        self.module_list = nn.ModuleList(modules)
        self.scales = cfg.scales
        self.stochastic_depth = StochasticDepth(p=cfg.layer_dropout, mode="row")
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(
        self,
        x: torch.tensor,
        /,
        sequence_mask: torch.tensor,
        layer_gates: torch.tensor,
        if_layer_drop: torch.tensor,
        hard_prune: bool = False,
    ) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :param layer_gates: the layer gate which is multiple with each layer output
        :param if_layer_drop: a bool var for each layer indicating whether to apply layerdrop out on the current layer
        :param hard_prune:
            if hard prune is True, the layer_gates will be binary and the layer with layer_gate == 0 will be directly jumped
            if hard prune is False, the layer_gates could be float number between range 0 and 1
        :return: torch.Tensor of shape [B, T, F]
        """
        assert len(layer_gates) == len(self.scales)
        assert len(if_layer_drop) == len(self.scales)

        for scale, module, layer_gate, apply_layer_drop in zip(
            self.scales, self.module_list, layer_gates, if_layer_drop
        ):
            # assert 0 <= layer_gate <= 1, "layer_gate should be in range between 0 and 1"
            if hard_prune and layer_gate == 0:
                x = x
            elif not apply_layer_drop:
                if isinstance(module, ConformerMHSAV1):
                    x = scale * module(x, sequence_mask) * layer_gate + x
                else:
                    x = scale * module(x) * layer_gate + x
            else:
                # directly jump this layer
                if isinstance(module, ConformerMHSAV1):
                    x = self.stochastic_depth(scale * module(x, sequence_mask) * layer_gate) + x
                else:
                    x = self.stochastic_depth(scale * module(x) * layer_gate) + x

        x = self.final_layer_norm(x)  # [B, T, F]
        return x


@dataclass
class ConformerRelPosEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: number of conformer layers in the conformer encoder
        frontend: a pair of ConformerFrontend and corresponding config
        block_cfg: configuration for ConformerBlockV2
        pct_params_set: a predefined set of expected percentage of number of parameters that will be jointly train together
        layer_dropout_kwargs: define the layer dropout value in two stages
            {
                "layer_dropout_stage_1":
                "layer_dropout_stage_2":
            }
        softmax_kwargs: define the related argument for temperature anealling and orthogonal constraint
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerRelPosBlockV1Config
    pct_params_set: List[float]
    layer_dropout_kwargs: Dict[str, float]
    softmax_kwargs: Dict[str, str | float]


class ConformerRelPosEncoderV1(nn.Module):
    """
    Conformer model encoders with dynamic size based on a supernet and M subnets that share parameters with the supernet,
    the training consists of two stages:
     - stage 1: use OrthoSoftmax to learn a binary mask for each subnet
     - stage 2: determine and fix the binary masks for all subnets, jointly train all networks efficiently with sandwich rule
     the subnet is selected layer-wisely from the supernet
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
        self.module_list = torch.nn.ModuleList([ConformerRelPosBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.layer_dropout_kwargs = cfg.layer_dropout_kwargs
        self.softmax_kwargs = cfg.softmax_kwargs
        self.layer_gates = torch.nn.Parameter(torch.FloatTensor(torch.zeros((cfg.num_layers * 5, cfg.num_layers * 5))))
        self.layer_gates.data.normal_(EPSILON, 0.00)
        self.register_parameter(
            "selected_mod_indices",
            nn.Parameter(torch.tensor([[-1] * len(self.layer_gates)] * len(self.pct_params_set)), requires_grad=False),
        )

        self.random_idx = -1
        self.recog_param_pct = 1

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

                layer_weights = torch.sum(gumbel_softmax, dim=0)
                selected_layer_indices = torch.argmax(gumbel_softmax, dim=1)

                # largest model
                for i in range(len(self.module_list)):
                    if_layer_drop = []
                    for j in range(5):
                        if 5 * i + j not in selected_layer_indices:
                            if_layer_drop.append(True)
                        else:
                            if_layer_drop.append(False)

                    outputs[-1] = self.module_list[i](
                        outputs[-1],
                        sequence_mask,
                        layer_gates=torch.tensor([1, 1, 1, 1, 1]),
                        if_layer_drop=if_layer_drop,
                    )

                # smallest model
                for i in range(len(self.module_list)):
                    outputs[0] = self.module_list[i](
                        outputs[0],
                        sequence_mask,
                        layer_gates=layer_weights[5 * i : 5 * i + 5],
                        if_layer_drop=torch.tensor([False] * 5),
                    )

                if global_train_step % 200 == 0:
                    print(f"small model num_layers: {k}")
                    print("layer_gates: {}".format(self.layer_gates))
                    print("layer_weights: {}".format(layer_weights))
                    print("selected_layer_indices: {}".format(sorted([int(i + 1) for i in selected_layer_indices])))

                if global_train_step == stage_1_global_steps:
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

            # Stage 2: fix the selection and jointly train
            else:
                if global_train_step == stage_1_global_steps + 1:
                    # change the layer dropout to layer_dropout_stage_2
                    for i in range(len(self.module_list)):
                        self.module_list[i].stochastic_depth.p = self.layer_dropout_kwargs["layer_dropout_stage_2"]
                        print(
                            "changed layer dropout in layer_dropout_stage_2 to {}".format(
                                self.layer_dropout_kwargs["layer_dropout_stage_2"]
                            )
                        )

                smallest_model_indices = [int(i) for i in self.selected_mod_indices[0] if i != -1]
                # largest model
                for i in range(len(self.module_list)):
                    if_layer_drop = []
                    for j in range(5):
                        if 5 * i + j not in smallest_model_indices:
                            if_layer_drop.append(True)
                        else:
                            if_layer_drop.append(False)

                    outputs[-1] = self.module_list[i](
                        outputs[-1],
                        sequence_mask,
                        layer_gates=torch.tensor([1, 1, 1, 1, 1]),
                        if_layer_drop=if_layer_drop,
                        hard_prune=True,
                    )

                # smallest model
                for i in range(len(self.module_list)):
                    layer_gates = []
                    for j in range(5):
                        if 5 * i + j not in smallest_model_indices:
                            layer_gates.append(0)
                        else:
                            layer_gates.append(1)
                    outputs[0] = self.module_list[i](
                        outputs[0],
                        sequence_mask,
                        layer_gates=torch.tensor(layer_gates),
                        if_layer_drop=torch.tensor([False] * 5),
                        hard_prune=True,
                    )

                # medium model
                if len(self.pct_params_set) > 2:
                    random_idx = np.random.choice(list(range(len(self.pct_params_set))[1:-1]), 1)[0]
                    self.random_idx = random_idx
                    medium_layers_indices = [int(i) for i in self.selected_mod_indices[random_idx] if i != -1]

                    for i in range(len(self.module_list)):
                        layer_gates = []
                        for j in range(5):
                            if 5 * i + j not in medium_layers_indices:
                                layer_gates.append(0)
                            else:
                                layer_gates.append(1)
                        outputs[self.random_idx] = self.module_list[i](
                            outputs[self.random_idx],
                            sequence_mask,
                            layer_gates=torch.tensor(layer_gates),
                            if_layer_drop=torch.tensor([False] * 5),
                            hard_prune=True,
                        )

                if global_train_step % 200 == 0:
                    print(f"selected_mod_indices: {self.selected_mod_indices}")

        # in recognition
        else:
            idx = self.pct_params_set.index(self.recog_param_pct)
            selected_mod_indices = [int(i) for i in self.selected_mod_indices[idx] if i != -1]

            for i in range(len(self.module_list)):
                layer_gates = []
                for j in range(5):
                    if 5 * i + j in selected_mod_indices:
                        layer_gates.append(1)
                    else:
                        layer_gates.append(0)

                outputs[idx] = self.module_list[i](
                    outputs[idx],
                    sequence_mask,
                    layer_gates=layer_gates,
                    if_layer_drop=torch.tensor([False] * 5),
                    hard_prune=True,
                )

        return outputs, softmax_constraint, sequence_mask
