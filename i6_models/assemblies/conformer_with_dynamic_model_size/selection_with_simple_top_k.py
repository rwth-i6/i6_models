import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

import numpy as np
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerMHSAV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer_with_dynamic_model_size.stochastic_depth import StochasticDepth
from i6_models.parts.conformer_with_dynamic_model_size.relaxed_top_k import RelaxedTopK


@dataclass
class ConformerBlockConfig(ModelConfiguration):
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
    ff_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: ConformerMHSAV1Config
    conv_cfg: ConformerConvolutionV1Config
    layer_dropout: float
    modules: List[str] = field(default_factory=lambda: ["ff", "mhsa", "conv", "ff"])
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])

    def __post__init__(self):
        super().__post_init__()
        assert len(self.modules) == len(self.scales), "modules and scales must have same length"
        for module_name in self.modules:
            assert module_name in ["ff", "mhsa", "conv"], "module not supported"


class ConformerBlock(nn.Module):
    """
    Conformre block
    - In soft prune mode, each module output can be multiple with a layer gate
    - In hard prune model, the modules which are not selected can be directly jumped
    """

    def __init__(self, cfg: ConformerBlockConfig):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()

        modules = []
        for module_name in cfg.modules:
            if module_name == "ff":
                modules.append(ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg))
            elif module_name == "mhsa":
                modules.append(ConformerMHSAV1(cfg=cfg.mhsa_cfg))
            elif module_name == "conv":
                modules.append(ConformerConvolutionV1(model_cfg=cfg.conv_cfg))
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
            assert 0 <= layer_gate <= 1, "layer_gate should be in range between 0 and 1"
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

        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerEncoderConfig(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV2
        num_layers_set: A predefined set of expected number of layers that will be jointly train together
        layer_dropout_kwargs: define the layer dropout value in two stages
            {
                "layer_dropout_stage_1":
                "layer_dropout_stage_2":
            }
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockConfig
    num_layers_set: List[int]
    layer_dropout_kwargs: Dict[str, float]


class ConformerEncoder(nn.Module):
    """
    Conformer model encoders with dynamic size based on a supernet and M subnets that share parameters with the supernet,
    the training consists of two stages:
     - stage 1: aims to learn the layer importance score, one supernet and one subnet are jointly trained, the subnet is progressively
        pruned with a dynamically decreasing siz
     - stage 2: determine and fix the binary masks for all subnets based on layer importance score, jointly train all networks efficiently
        with sandwich rule
    Simple-Top-k is used here in stage 1 to learn the layer importance score
    """

    def __init__(self, cfg: ConformerEncoderConfig):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.num_layers_set = sorted(cfg.num_layers_set)
        self.max_k = max(cfg.num_layers_set)
        self.min_k = min(cfg.num_layers_set)
        self.module_list = torch.nn.ModuleList([ConformerBlock(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.sampler = RelaxedTopK(k=self.min_k)
        self.layer_dropout_kwargs = cfg.layer_dropout_kwargs
        self.layer_gates = torch.nn.Parameter(torch.FloatTensor(torch.zeros(cfg.num_layers * 4)))
        self.layer_gates.data.normal_(0.5, 0.00)
        self.recog_num_mods = 4 * cfg.num_layers

    def forward(
        self,
        data_tensor: torch.Tensor,
        /,
        sequence_mask: torch.Tensor,
        global_train_step: int,
        k_anneal_kwargs: dict,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :param global_train_step: the index of the global train step
        :param k_anneal_kwargs: define the key word arguments for k annealing
            {
                "k_anneal_num_steps_per_iter": how many number of train steps for one annealing iteration
                "k_reduction_per_iter": in each iteration, how much to reduce k
            }

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        k_anneal_num_steps_per_iter = k_anneal_kwargs["k_anneal_num_steps_per_iter"]
        k_reduction_per_iter = k_anneal_kwargs["k_reduction_per_iter"]
        k_anneal_num_iters = (4 * len(self.module_list) - self.min_k) / k_reduction_per_iter

        outputs = [x for _ in range(len(self.num_layers_set))]

        # in training
        if self.training:
            # Stage 1: jointly train one large model and one small model
            if global_train_step <= k_anneal_num_steps_per_iter * k_anneal_num_iters:
                k = max(
                    (48 - k_reduction_per_iter)
                    - (global_train_step // k_anneal_num_steps_per_iter * k_reduction_per_iter),
                    self.min_k,
                )
                self.sampler.k = k
                gumbel_softmax = self.sampler(self.layer_gates)

                _, remove_mods_indices = torch.topk(gumbel_softmax, k=48 - k, largest=False)

                # largest model
                for i in range(len(self.module_list)):
                    if_layer_drop = []
                    for j in range(4):
                        if 4 * i + j in remove_mods_indices:
                            if_layer_drop.append(True)
                        else:
                            if_layer_drop.append(False)

                    outputs[-1] = self.module_list[i](
                        outputs[-1],
                        sequence_mask,
                        layer_gates=torch.tensor([1, 1, 1, 1]),
                        if_layer_drop=if_layer_drop,
                    )

                # smallest model
                for i in range(len(self.module_list)):
                    outputs[0] = self.module_list[i](
                        outputs[0],
                        sequence_mask,
                        layer_gates=gumbel_softmax[4 * i : 4 * i + 4],
                        if_layer_drop=torch.tensor([False] * 4),
                    )

                if global_train_step % 200 == 0:
                    print(f"small model num_layers: {k}")
                    print("layer_gates: {}".format(self.layer_gates))
                    print("remove_mods_indices: {}".format(sorted([int(i + 1) for i in remove_mods_indices])))

            # Stage 2: fix the selection and jointly train
            else:
                if global_train_step == k_anneal_num_steps_per_iter * k_anneal_num_iters + 1:
                    # change the layer dropout to layer_dropout_stage_2
                    for i in range(len(self.module_list)):
                        self.module_list[i].stochastic_depth.p = self.layer_dropout_kwargs["layer_dropout_stage_2"]
                        print(
                            "changed layer dropout in layer_dropout_stage_2 to {}".format(
                                self.layer_dropout_kwargs["layer_dropout_stage_2"]
                            )
                        )

                _, remove_layer_indices = torch.topk(self.gates, k=48 - self.min_k, largest=False)

                # largest model
                for i in range(len(self.module_list)):
                    if_layer_drop = []
                    for j in range(4):
                        if 4 * i + j in remove_layer_indices:
                            if_layer_drop.append(True)
                        else:
                            if_layer_drop.append(False)

                    outputs[-1] = self.module_list[i](
                        outputs[-1],
                        sequence_mask,
                        layer_gates=torch.tensor([1, 1, 1, 1]),
                        if_layer_drop=if_layer_drop,
                        hard_prune=True,
                    )

                # smallest model
                for i in range(len(self.module_list)):
                    layer_gates = []
                    for j in range(4):
                        if 4 * i + j in remove_layer_indices:
                            layer_gates.append(0)
                        else:
                            layer_gates.append(1)
                    outputs[0] = self.module_list[i](
                        outputs[0],
                        sequence_mask,
                        layer_gates=torch.tensor(layer_gates),
                        if_layer_drop=torch.tensor([False] * 4),
                        hard_prune=True,
                    )

                # medium model
                if len(self.num_layers_set) > 2:
                    random_idx = np.random.choice(list(range(len(self.num_layers_set))[1:-1]), 1)[0]
                    self.random_idx = random_idx
                    _, medium_remove_layers_indices = torch.topk(
                        self.gates, k=48 - self.num_layers_set[random_idx], largest=False
                    )

                    for i in range(len(self.module_list)):
                        layer_gates = []
                        for j in range(4):
                            if 4 * i + j in medium_remove_layers_indices:
                                layer_gates.append(0)
                            else:
                                layer_gates.append(1)
                        outputs[1 + self.random_idx] = self.module_list[i](
                            outputs[1 + self.random_idx],
                            sequence_mask,
                            layer_gates=torch.tensor(layer_gates),
                            if_layer_drop=torch.tensor([False] * 4),
                            hard_prune=True,
                        )

                if global_train_step % 200 == 0:
                    print("remove_layer_indices: {}".format(sorted([int(i + 1) for i in remove_layer_indices])))
                    if len(self.num_layers_set) > 2:
                        print(
                            "medium_remove_layers_indices: {}".format(
                                sorted([int(i + 1) for i in medium_remove_layers_indices])
                            )
                        )

        # in recognition
        else:
            idx = self.num_layers_set.index(self.recog_num_mods)
            remove_mods_indices = torch.topk(
                self.gates, k=4 * len(self.module_list) - self.recog_num_mods, largest=False
            )
            for i in range(len(self.module_list)):
                layer_gates = []
                for j in range(4):
                    if 4 * i + j in remove_mods_indices:
                        layer_gates.append(0)
                    else:
                        layer_gates.append(1)

                outputs[idx] = self.module_list[i](
                    outputs[idx],
                    sequence_mask,
                    layer_gates=layer_gates,
                    if_layer_drop=torch.tensor([False] * 4),
                    hard_prune=True,
                )

        return outputs, sequence_mask
