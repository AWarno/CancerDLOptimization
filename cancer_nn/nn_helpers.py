"""Scripts for helper function for torch neural networks"""

import torch.nn as nn


def attention_module(embed_dim: int = 4) -> nn.MultiheadAttention:
    """Script for attention layer

    Args:
    ----------
        embed_dim (int, optional): embeding dimensio
        (feature dimension). Defaults to 4.

    Returns:
    ----------
        nn.MultiheadAttention: Multihead attention module
        with batch_first=True
    """
    return nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=embed_dim, dropout=0.2, batch_first=True
    )


def base_linear_module(in_f: int, out_f: int) -> nn.Sequential:
    """Basic linear layer with
    LeakyRelu activation and Batchnormalization

    Args:
    ----------
        in_f (int): input size (for linear layer)
        out_f (int): output size (for linear layer)

    Returns:
    ----------
        nn.Sequential: torch nn>sequential with:
        Linear layer, activation layer and
         batchnormalization1D layer
    """
    return nn.Sequential(
        nn.Linear(in_f, out_f), nn.LeakyReLU(), nn.BatchNorm1d(num_features=out_f)
    )


def calculate_matching_padding(i: int, o: int, k: int, d: int, s: int) -> int:
    """calculate matching padding for CNN1 D
     so that the outpt had o size.

    Args:
    ----------
        i (int): input size
        o (int): desired output size
        k (int): kernel size
        d (int): dilatation size
        s (int): stride size

    Returns:
    ----------
        int: matching padding
    """
    p = ((o - 1) * s + (k - 1) * (d - 1) - i + k) / 2
    return int(p)


def transconv_matching_padding(i: int, o: int, s: int, p: int, k: int) -> int:
    """calculate matching padding for
    transposed convolution
     so that the outpt had o size.

    Args:
    ----------
        i (int): input size
        o (int): desired output size
        s (int): stride size
        d (int): padding size
        k (int): kernel size

    Returns:
    ----------
        int: matching padding
    """
    output_padding = ((i - 1) * s - 2 * p + k - o) * (-1)
    return output_padding


def base_conv1_layer(
    n_inputs: int,
    n_outputs: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Basic CNN 1D layer,
    with groupnormalization layer and
    LeakyreLU activation function

    Args:
    ----------
        n_inputs (int): input channels size
        n_outputs (int): output channels size
        kernel_size (int): kernel size
        stride (int, optional): stride size. Defaults to 1.
        padding (int, optional): padding size. Defaults to 1.
        dilation (int, optional): dilation size. Defaults to 1.

    Returns:
    ----------
        nn.Sequential: nn.Sequential with
        CNN 1D layer, GroupNorm layer and activation
        fucntion (LeakyreLU )
    """
    return nn.Sequential(
        nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        nn.GroupNorm(1, n_outputs),
        nn.LeakyReLU(inplace=True),
    )


def calculate_output_shape(i: int, p: int, k: int, d: int, s: int) -> int:
    """Calculate output channels size for
     convolution with given parameters

    Args:
    ----------
        i (int): input channels size
        p (int): padding size
        k (int): kerenl size
        d (int): dilation size
        s (int): stride size

    Returns:
    ----------
        int: _description_
    """
    o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
    return int(o)
