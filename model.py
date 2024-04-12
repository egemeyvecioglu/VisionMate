import torch.nn as nn
from torch.nn.functional import tanh
from numpy import pi
import timm

# Combining FastViT with MLP
class FastViTMLP(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.fastvit = timm.create_model(
            "fastvit_sa12.apple_in1k",
            pretrained=True,
            num_classes=0,  # Remove the final classifier layer
        ).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(self.fastvit.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        ).to(self.device)

        # self.mlp = nn.Linear(self.fastvit.num_features, 2).to(self.device).to(self.device)

    def forward(self, x):
        x = self.fastvit(x).to(self.device)
        x = self.mlp(x)  # Apply MLP without Tanh

        angular_output = x[:, :2]
        angular_output[:, 0:1] = angular_output[:, 0:1] * pi
        angular_output[:, 1:2] = angular_output[:, 1:2] * pi/2

        return angular_output
