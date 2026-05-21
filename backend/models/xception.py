"""
Xception-based Deepfake Detector
Modified Xception architecture optimized for face manipulation detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class XceptionBlock(nn.Module):
    """Modified Xception block"""
    
    def __init__(self, in_channels, out_channels, reps=1, stride=1, start_with_relu=True):
        super(XceptionBlock, self).__init__()
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.start_with_relu = start_with_relu
        
        rep = []
        for i in range(reps):
            rep.append(SeparableConv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            rep.append(nn.ReLU(inplace=True))
        
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        
        self.rep = nn.Sequential(*rep)
        
    def forward(self, x):
        if self.start_with_relu:
            x = self.relu(x)
        
        return self.rep(x) + self.skip(x)


class XceptionDetector(nn.Module):
    """
    Modified Xception for deepfake detection
    Based on "Xception: Deep Learning with Depthwise Separable Convolutions"
    """
    
    def __init__(self, num_classes: int = 1):
        super(XceptionDetector, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.block1 = XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2)
        self.block3 = XceptionBlock(256, 728, reps=2, stride=2)
        
        # Middle flow
        self.middle_blocks = nn.Sequential(*[
            XceptionBlock(728, 728, reps=3, stride=1) for _ in range(8)
        ])
        
        # Exit flow
        self.block12 = XceptionBlock(728, 1024, reps=2, stride=2)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.middle_blocks(x)
        
        # Exit flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x):
        """Extract feature embeddings"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle_blocks(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


if __name__ == "__main__":
    model = XceptionDetector()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
