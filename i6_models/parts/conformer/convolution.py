from torch import nn


class ConformerConvolutionV1(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13
    """

    def __init__(self, channels, kernel_size, dropout=0.1):
        """
        :param int channels: number of channels for conv layers
        :param int kernel_size: kernel size of conv layers
        """
        super().__init__()

        # kernel size has to be odd to get same input length without zero padding when using odd strides.
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )
        self.layer_norm = nn.LayerNorm(channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor):
        """
        :param torch.Tensor tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        out_tensor = self.layer_norm(tensor)

        # conv layers expect shape [B,F,T] so we have to transpose here
        out_tensor = out_tensor.transpose(1, 2)  # [B,F,T]

        out_tensor = self.pointwise_conv1(out_tensor)  # [B,2F,T]
        out_tensor = nn.functional.glu(out_tensor, dim=1)  # [B,F,T]

        out_tensor = self.depthwise_conv(out_tensor)
        out_tensor = nn.functional.silu(self.batch_norm(out_tensor), inplace=True)  # apply swish activation

        out_tensor = self.pointwise_conv2(out_tensor)

        return tensor + self.dropout(out_tensor.transpose(1, 2))
