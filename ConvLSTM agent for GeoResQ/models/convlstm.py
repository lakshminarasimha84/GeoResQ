"""
ConvLSTM building blocks used by IndiaConvLSTM_UNet for temporal fusion of
the satellite frame sequence at every U-Net encoder scale.

This was previously referenced in the code as an undefined `RNN` class
(never implemented anywhere in the project) which made the model impossible
to construct. This file provides a standard, working ConvLSTM cell.
"""
import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell operating on (B, C, H, W) feature maps."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device, dtype):
        h, w = spatial_size
        shape = (batch_size, self.hidden_dim, h, w)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )


class ConvLSTM(nn.Module):
    """
    Runs a ConvLSTMCell over a time sequence.

    Input:  x_seq of shape (B, T, C, H, W)
    Output: final hidden state of shape (B, hidden_dim, H, W)
            (i.e. the temporally-fused feature map summarizing the whole
            observed sequence up to the last timestep, used as the
            "nowcasting context" at that encoder scale).
    """

    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, bias=True):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size=kernel_size, bias=bias)

    def forward(self, x_seq):
        b, t, c, h, w = x_seq.shape
        h_t, c_t = self.cell.init_hidden(b, (h, w), x_seq.device, x_seq.dtype)
        for step in range(t):
            h_t, c_t = self.cell(x_seq[:, step], h_t, c_t)
        return h_t
