import torch.nn as nn
from torch.nn.functional import tanh
from numpy import pi
import timm
import math
import torch

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
    
class ResNET50L2CS(nn.Module):
    def __init__(self, device, num_bins, block, layers):
        self.inplanes = 64
        self.device = device
        super(ResNET50L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.layer1 = self._make_layer(block, 64, layers[0]).to(device)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2).to(device)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2).to(device)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins).to(device)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins).to(device)

       # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False).to(self.device),
                nn.BatchNorm2d(planes * block.expansion).to(self.device),
            ).to(self.device)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        x.to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        
        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze