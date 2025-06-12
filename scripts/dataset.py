import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class FishDataset(Dataset):
    def __init__(self, labels_csv, image_dir, transform=None):
        """
        Dataset para imágenes de peces con sensores y etiquetas.
        :param labels_csv: Ruta al CSV con columnas: imagen, sensores, etiqueta
        :param image_dir: Carpeta donde están las imágenes
        :param transform: Transformaciones a aplicar a las imágenes
        """
        self.df = pd.read_csv(labels_csv)
        self.clase2idx = {"saludable": 0, "estres_leve": 1, "enfermo": 2}
        self.image_dir = image_dir
        self.transform = transform

        self.cols_sens = ["temperatura", "pH", 
                          "conductividad", "TDS", 
                          "salinidad", "presion", "oxigeno_mgL"]

        sens_values = self.df[self.cols_sens].values.astype(float)
        self.min_vals = sens_values.min(axis=0)
        self.max_vals = sens_values.max(axis=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["imagen"]
        etiqueta_str = row["etiqueta"]
        etiqueta_int = self.clase2idx.get(etiqueta_str, -1)

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Normalizar sensores
        sens_raw = row[self.cols_sens].values.astype(float)
        sens_norm = (sens_raw - self.min_vals) / (self.max_vals - self.min_vals + 1e-6)
        sens_tensor = torch.tensor(sens_norm, dtype=torch.float32)

        return image, sens_tensor, etiqueta_int

