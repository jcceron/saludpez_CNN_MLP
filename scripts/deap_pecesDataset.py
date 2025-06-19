# scripts/deap_pecesDataset.py
# Dataset multimodal 
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os
from torchvision import transforms

class PecesDataset(Dataset):
    def __init__(self, df, img_dir, sensor_cols, label_col, img_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.sensor_cols = sensor_cols
        self.label_col   = label_col
        self.img_col     = img_col

        # Mapear etiquetas string → índice
        self.classes   = sorted(self.df[self.label_col].unique())
        self.class2idx = {lbl: idx for idx,lbl in enumerate(self.classes)}

        # Transformaciones de imagen
        self.tx = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # --- 1) Cargar imagen con manejo de errores ---
        label_name = row[self.label_col]
        img_name   = row[self.img_col]
        img_path   = os.path.join(self.img_dir, label_name, img_name)

        try:
            # Intentamos abrir y convertir la imagen
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            # Si falla, avisamos y creamos un placeholder negro
            print(f"⚠️ Warning: no se pudo cargar '{img_path}': {e}")
            img = Image.new("RGB", (224,224), color=(0,0,0))

        # Aplicamos las transformaciones
        img = self.tx(img)

        # --- 2) Vector de sensores (forzando float) ---
        sens_list = []
        for col in self.sensor_cols:
            sens_list.append(float(row[col]))
        sens = torch.tensor(sens_list, dtype=torch.float32)

        # --- 3) Etiqueta numérica ---
        label = self.class2idx[label_name]

        return img, sens, label