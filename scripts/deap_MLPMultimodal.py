# scripts/deap_MLPMultimodal.py
import torch.nn as nn
import torch
# =========================
# 2) DEFINICIÓN DEL MODELO
# =========================
class MLPMultimodal(nn.Module):
    def __init__(self, 
                 img_size=(224,224), 
                 sensor_dim=5, 
                 hidden_dim=64, 
                 n_classes=3):
        """
        Modelo CNN+MLP multimodal:
         - CNN sencilla sobre la imagen
         - MLP de una capa para los sensores
         - Clasificador final que concatena ambas salidas
        """
        super().__init__()
        
        # 2.1 Bloque CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # calcular dimensión de salida de la CNN
        with torch.no_grad():
            dummy = torch.zeros(1,3,*img_size)
            feat_dim = self.cnn(dummy).shape[1]
        
        # 2.2 Bloque MLP sensores
        self.mlp = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.ReLU()
        )
        # 2.3 Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, img, sens):
        f_img = self.cnn(img)
        f_sen = self.mlp(sens)
        return self.classifier(torch.cat([f_img,f_sen], dim=1))