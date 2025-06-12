# scripts/dataset_multimodal.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

# 5) Dataset multimodal
class FishDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, sensor_cols, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.sensor_cols = sensor_cols
        self.transform = transform
        self.label2idx = {'enfermo':0, 'estres_leve':1, 'saludable':2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['etiqueta_kmeans'], row['imagen'])
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224,224))
        if self.transform:
            img = self.transform(img)
        sensors = []
        for c in self.sensor_cols:
            sensors.append(float(row[c]))
        sensors = torch.tensor(sensors, dtype=torch.float32)
        label = self.label2idx[row['etiqueta_kmeans']]
        return img, sensors, label