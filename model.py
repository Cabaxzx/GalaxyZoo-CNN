from typing import Literal
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConvolutionBlock(nn.Module):
    """
    Standard sequential convolutional block equipped with Batch Normalization and residual capabilities.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, maintain_size=True):
        super().__init__()
        # Uniform spatial dimensions through the block.
        padding_size = kernel_size // 2 if maintain_size else 0

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_size, bias=False)
        # BatchNorm accelerates convergence by reducing internal covariate shift.
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding_size, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        # Residual shortcut integration.
        self.add_residual = maintain_size
        if self.add_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.add_residual:
            out = out + self.shortcut(x)
        return self.activation(out)


class CNN(nn.Module):
    """
    Modular Convolutional Neural Network processing multi-task objectives.
    Implements distinct forward pass based on required architecture.
    """

    def __init__(self, in_channels, out_channels, input_shape, hidden_channels, final_hidden_channels, n_blocks,
                 architecture: Literal["classification", "binary", "tree", "flat"] = "classification"):
        super().__init__()
        self.architecture = architecture

        cnn_layers = []
        curr_in = in_channels
        curr_out = hidden_channels
        for _ in range(n_blocks):
            cnn_layers.append(DoubleConvolutionBlock(curr_in, curr_out))
            curr_in = curr_out
            # MaxPool reduces spatial dimensionality by a factor of 2, establishing translation invariance.
            cnn_layers.append(nn.MaxPool2d(kernel_size=2))
            if curr_out < 512:
                curr_out *= 2
        self.cnn_blocks = nn.Sequential(*cnn_layers)

        #Hidden layer
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(curr_in, final_hidden_channels),
            nn.ReLU(),
            # Dropout deactivates a percentage of neurons, mitigating model overfitting.
            nn.Dropout(p=0.35),
        )

        #Task-Specific Output Heads
        if architecture == "classification":
            self.standard_head = nn.Linear(final_hidden_channels, out_channels)
        elif architecture == "binary":
            self.binary_head = nn.Linear(final_hidden_channels, 1)
        elif architecture == "flat":
            self.flat_head = nn.Linear(final_hidden_channels, out_channels)
        elif architecture == "tree":
            # Initializes explicit linear layers for each logical node in the Bayesian decision tree.
            self.q1_head = nn.Linear(final_hidden_channels, 3)
            self.q2_head = nn.Linear(final_hidden_channels, 2)
            self.q7_head = nn.Linear(final_hidden_channels, 3)
            self.q6_head = nn.Linear(final_hidden_channels, 2)
            self.q8_head = nn.Linear(final_hidden_channels, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared_features(self.cnn_blocks(x))

        if self.architecture == "classification":
            return self.standard_head(features)
        elif self.architecture == "binary":
            return self.binary_head(features)

        elif self.architecture == "flat":
            preds = self.flat_head(features)
            return preds

        elif self.architecture == "tree":
            # Softmax is enforced internally to validate Bayesian conditional distributions sum to 1.0.
            q1_cond = F.softmax(self.q1_head(features), dim=1)
            q2_cond = F.softmax(self.q2_head(features), dim=1)
            q7_cond = F.softmax(self.q7_head(features), dim=1)
            q6_cond = F.softmax(self.q6_head(features), dim=1)
            q8_cond = F.softmax(self.q8_head(features), dim=1)

            if not self.training:
                # Noise filtration thresholding.
                q1_cond = torch.where(q1_cond < 0.04, torch.tensor(0.01, device=q1_cond.device), q1_cond)
                q2_cond = torch.where(q2_cond < 0.04, torch.tensor(0.01, device=q2_cond.device), q2_cond)
                q7_cond = torch.where(q7_cond < 0.04, torch.tensor(0.01, device=q7_cond.device), q7_cond)
                q6_cond = torch.where(q6_cond < 0.04, torch.tensor(0.01, device=q6_cond.device), q6_cond)
                q8_cond = torch.where(q8_cond < 0.04, torch.tensor(0.01, device=q8_cond.device), q8_cond)

                # Reversion formatting utilizes class-specific denominators to guarantee boundary compliance.
                q1_cond = torch.clamp((q1_cond - 0.01) / 0.97, min=0.0, max=1.0)
                q2_cond = torch.clamp((q2_cond - 0.01) / 0.98, min=0.0, max=1.0)
                q7_cond = torch.clamp((q7_cond - 0.01) / 0.97, min=0.0, max=1.0)
                q6_cond = torch.clamp((q6_cond - 0.01) / 0.98, min=0.0, max=1.0)
                q8_cond = torch.clamp((q8_cond - 0.01) / 0.93, min=0.0, max=1.0)

            # Mathematical node construction via the chain rule of probability: P(A ∩ B) = P(A|B) * P(B).
            q1_final = q1_cond
            q2_final = q2_cond * q1_cond[:, 1:2]
            q7_final = q7_cond * q1_cond[:, 0:1]
            q6_final = q6_cond
            q8_final = q8_cond * q6_final[:, 0:1]

            return torch.cat([q1_final, q2_final, q7_final, q6_final, q8_final], dim=1)