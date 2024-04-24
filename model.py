import torch.nn as nn
from torch.nn.functional import tanh
from numpy import pi
import timm
from fastervit import create_model


# Combining FastViT with MLP
class FastViTMLP(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.fastvit = timm.create_model(
            "fastvit_t8.apple_in1k",
            pretrained=True,
            num_classes=0,  # Remove the final classifier layer
        ).to(self.device)

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.fastvit.num_features, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, 2),
        # ).to(self.device)

        self.mlp = nn.Linear(self.fastvit.num_features, 2).to(self.device).to(self.device)

    def forward(self, x):
        x = self.fastvit(x).to(self.device)
        x = self.mlp(x)  # Apply MLP without Tanh

        angular_output = x[:, :2]
        angular_output[:, 0:1] = tanh(angular_output[:, 0:1]) * pi
        angular_output[:, 1:2] = tanh(angular_output[:, 1:2]) * pi / 2

        return angular_output


class FastViTL2CS(nn.Module):
    def __init__(self, device, num_bins):
        super().__init__()

        self.device = device

        self.fastvit = timm.create_model(
            "fastvit_sa12.apple_in1k",
            pretrained=True,
            num_classes=0,  # Remove the final classifier layer
        ).to(self.device)

        self.fc_yaw_gaze = nn.Linear(1024, num_bins).to(self.device)
        self.fc_pitch_gaze = nn.Linear(1024, num_bins).to(self.device)

    def forward(self, x):
        x = self.fastvit(x)

        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)

        return pre_yaw_gaze, pre_pitch_gaze


class FasterViT(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.fastervit = create_model(
            "faster_vit_0_224", pretrained=True, model_path="/tmp/faster_vit_0.pth.tar"
        ).to(self.device)

        # self.fastervit.head = nn.Sequential(
        #     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2)
        # ).to(self.device)

        self.fastervit.head = nn.Linear(512, 2).to(self.device).to(self.device)

        print("Model initialized")

    def forward(self, x):
        x = self.fastervit(x).to(self.device)

        angular_output = x[:, :2]
        angular_output[:, 0:1] = tanh(angular_output[:, 0:1]) * pi
        angular_output[:, 1:2] = tanh(angular_output[:, 1:2]) * pi / 2

        return angular_output
