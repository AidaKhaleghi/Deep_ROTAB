import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Single conv computes all four gates (input, forget, cell, output) at once
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.gates(torch.cat([x, h_prev], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, height, width, device):
        shape = (batch, self.hidden_channels, height, width)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


class BNet(nn.Module):
    """Predicts the coefficient vector b[t] from the input frame.

    The network outputs a residual delta added to the fixed warm-start b0
    (singular values from the initialization SVD). The final layer is
    zero-initialized so the very first prediction equals b0 exactly.
    """

    def __init__(self, rank, b0, in_channels=2, hidden_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cell = ConvLSTMCell(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden_channels, rank)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        b0 = torch.as_tensor(b0, dtype=torch.float32)
        self.register_buffer("b0", b0)

    def forward(self, x, state=None):
        # x: (1, in_channels, H, W)
        feat = self.encoder(x)
        if state is None:
            state = self.cell.init_state(
                feat.shape[0], feat.shape[2], feat.shape[3], feat.device
            )
        h, c = self.cell(feat, state)
        pooled = self.pool(h).flatten(1)
        delta = self.head(pooled).squeeze(0)
        b = self.b0 + delta
        return b, (h, c)
