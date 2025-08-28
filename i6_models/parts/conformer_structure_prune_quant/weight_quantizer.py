import torch
from torch import nn
from torch.ao.quantization.utils import check_min_max_valid
import torch.ao.quantization as torch_quant
from typing import Optional, Union

def get_quantization_range_from_bit_precision(bits, dtype):

    if bits == 1.5:
        quant_min = -1
        quant_max = 1
    elif bits == 2.5:
        quant_min = -2
        quant_max = 3
    elif bits == 3.5:
        quant_min = -4
        quant_max = 4
    elif dtype == torch.qint8:
        quant_min = -(2 ** (bits - 1))
        quant_max = (2 ** (bits - 1)) - 1
    elif dtype == torch.quint8:
        quant_min = 0
        quant_max = (2**bits) - 1
    else:
        raise ValueError(f"Unrecognized dtype {dtype}")
    return quant_min, quant_max


class WeightQuantizer(nn.Module):
    def __init__(
        self,
        bit_precision: int,
        dtype: torch.dtype,
        method: str,
        observer_only_in_train: bool,
        reduce_range: bool = False,
    ):
        super().__init__()

        self.quant_min, self.quant_max = get_quantization_range_from_bit_precision(bit_precision, dtype)
        self.dtype = dtype
        self.reduce_range = reduce_range
        self.quant_fn, self.observer = None, None
        self.method = None
        self.quant_fn, self.observer = self.__get_quant_fn_and_observer_for_method(method)
        self.scale = None
        self.zero_point = None
        self.observer_only_in_train = observer_only_in_train

    def __get_quant_fn_and_observer_for_method(self, method):
        if self.quant_fn is not None and self.observer is not None:
            return self.quant_fn, self.observer
        if method == "per_tensor":
            quant_fn = torch.fake_quantize_per_tensor_affine
            self.method = torch.per_tensor_affine
            observer = torch_quant.observer.MinMaxObserver(
                quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype, reduce_range=self.reduce_range
            )
        elif method == "per_tensor_symmetric":
            quant_fn = torch.fake_quantize_per_tensor_affine
            self.method = torch.per_tensor_symmetric
            observer = torch_quant.observer.MinMaxObserver(
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.dtype,
                reduce_range=self.reduce_range,
                qscheme=torch.per_tensor_symmetric,
            )
        else:
            raise ValueError(f"Unknown quantization method: {method}!")

        return quant_fn, observer

    def forward(self, tensor: torch.Tensor):
        if self.observer_only_in_train is False or self.training:
            tensor = self.observer(tensor)
        self.set_scale_and_zp()
        assert self.scale is not None and self.zero_point is not None
        tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        assert check_min_max_valid(self.observer.min_val, self.observer.max_val), "Need to init observer first"
        scale, zero_point = self.observer.calculate_qparams()
        self.scale = scale.to(dtype=torch.float32)
        self.zero_point = zero_point.to(dtype=torch.int32)


class ActivationQuantizer(nn.Module):
    def __init__(
        self,
        bit_precision: int,
        dtype: torch.dtype,
        method: str,
        channel_axis: Optional[int],
        moving_avrg: Optional[float],  # default if enabled should be 0.01, if set enables moving average
        observer_only_in_train: bool,
        reduce_range: bool = False,
    ):
        super().__init__()
        self.quant_min, self.quant_max = get_quantization_range_from_bit_precision(bit_precision, dtype)
        self.dtype = dtype
        self.channel_axis = channel_axis
        self.moving_avrg = moving_avrg
        self.reduce_range = reduce_range
        self.quant_fn, self.observer, self.base_observer_args = None, None, None
        self.quant_fn, self.observer, self.base_observer_args = self.__get_quant_fn_and_observer_for_method(method)
        self.zero_point = None
        self.scale = None
        self.observer_only_in_train = observer_only_in_train

    def __get_quant_fn_and_observer_for_method(self, method):
        if all(x is not None for x in [self.quant_fn, self.base_observer_args, self.observer]):
            return self.quant_fn, self.base_observer_args, self.observer
        if method == "per_tensor":
            quant_fn = torch.fake_quantize_per_tensor_affine
            base_observer_args = [self.quant_min, self.quant_max]
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAverageMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                )
            else:
                observer = torch_quant.observer.MinMaxObserver(
                    quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype, reduce_range=self.reduce_range
                )
        elif method == "per_channel":
            quant_fn = torch.fake_quantize_per_channel_affine
            base_observer_args = [self.channel_axis, self.quant_min, self.quant_max]
            assert self.channel_axis is not None
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAveragePerChannelMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    ch_axis=self.channel_axis,
                    reduce_range=self.reduce_range,
                )
            else:
                observer = torch_quant.observer.PerChannelMinMaxObserver(
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    ch_axis=self.channel_axis,
                )
        elif method == "per_tensor_symmetric":
            quant_fn = torch.fake_quantize_per_tensor_affine
            base_observer_args = [self.quant_min, self.quant_max]
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAverageMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    qscheme=torch.per_tensor_symmetric,
                )
            else:
                observer = torch_quant.observer.MinMaxObserver(
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    qscheme=torch.per_tensor_symmetric,
                )
        else:
            raise ValueError(f"Unknown quantization method: {method}!")

        return quant_fn, observer, base_observer_args

    def forward(self, tensor: torch.Tensor):
        if self.observer_only_in_train is False or self.training:
            tensor = self.observer(tensor)
        self.set_scale_and_zp()
        assert (
            self.scale is not None and self.zero_point is not None
        ), "Need to calibrate before applying quant, disable apply_calibration"
        tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        assert check_min_max_valid(self.observer.min_val, self.observer.max_val), "Need to init observer first"
        scale, zero_point = self.observer.calculate_qparams()
        self.scale = scale.to(dtype=torch.float32)
        self.zero_point = zero_point.to(dtype=torch.int32)