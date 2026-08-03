"""
modality_classifier/models/classifier_head.py
=============================================
Modular PyTorch classification head with Global Pooling, BatchNorm, Dropout, and Linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModularClassifierHead(nn.Module):
    """
    Modular classifier head:
    AdaptiveAvgPool2d -> Flatten -> (BatchNorm1d) -> Dropout -> Linear Softmax
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = torch.mean(x, dim=[2, 3])  # Global Average Pooling if spatial

        # BatchNorm1d requires > 1 sample per batch during training.
        # For batch_size=1 (e.g. last batch or small test sets), temporarily
        # switch to eval mode to use running statistics instead.
        if self.training and x.size(0) == 1:
            self.eval()
            out = self.classifier(x)
            self.train()
            return out

        return self.classifier(x)
