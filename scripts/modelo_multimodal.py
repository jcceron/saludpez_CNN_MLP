
# scripts/modelo_multimodal.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# 7) Definir modelo multimodal(ResNet18 + MLP)
class MultiModalNet(nn.Module):
    def __init__(self, sensor_dim, n_classes):
        super().__init__()
        # Usar weights en lugar de pretrained para no generar warning
        #self.cnn = models.resnet18(pretrained=True)
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(sensor_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512+64, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x_img, x_sens):
        f_img = self.cnn(x_img)
        f_sens = self.mlp(x_sens)
        return self.classifier(torch.cat([f_img, f_sens], dim=1))
