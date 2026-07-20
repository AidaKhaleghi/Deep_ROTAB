import torch
import torch.nn as nn


class SNet(nn.Module):
    """Learned proximal operator for the sparse component S, adapted from the
    SparseModule of RPCANet++.

    Computes S = x - epsilon * convs(x) with x = D - L. Plain
    soft-thresholding is the special case where the correction equals
    lam' * sign(x) (clipped), so this is a learnable generalization of the
    shrinkage step.
    """

    def __init__(self, channel=32, layers=6):
        super().__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for _ in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.epsilon = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def forward(self, D, L):
        # D, L: (1, 1, H, W)
        x = D - L
        S = x - self.epsilon * self.convs(x)
        return S
