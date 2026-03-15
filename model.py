"""
model.py
--------
Custom PyTorch neural network for biometric classification.

Two separate lightweight MLPs:
  - FaceClassifier  : input 256-d LBPH histogram
  - IrisClassifier  : input 512-d binary descriptor

Architecture (MLP with BatchNorm + Dropout):
  Input -> FC(256) -> BN -> ReLU -> Dropout(0.3)
        -> FC(128) -> BN -> ReLU -> Dropout(0.3)
        -> FC(64)  -> ReLU
        -> FC(num_classes)   [raw logits, use CrossEntropyLoss]
"""

import torch
import torch.nn as nn


class BiometricMLP(nn.Module):
    """
    Generic MLP classifier for biometric feature vectors.
    Works for both face (256-d) and iris (512-d) inputs.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceClassifier(BiometricMLP):
    def __init__(self, num_classes: int):
        super().__init__(input_dim=256, num_classes=num_classes, hidden=[256, 128, 64])


class IrisClassifier(BiometricMLP):
    def __init__(self, num_classes: int):
        super().__init__(input_dim=512, num_classes=num_classes, hidden=[256, 128, 64])