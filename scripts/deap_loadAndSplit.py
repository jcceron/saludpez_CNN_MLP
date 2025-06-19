# scripts/deap_loadAndSplit.py
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from scripts.deap_pecesDataset import PecesDataset

# =========================
# 1) CARGA Y SPLIT ESTRATIFICADO
# =========================
def load_and_split(test_size=0.2, 
                   val_size=0.1, 
                   random_state=42,
                   label_csv="../data/labels2_kmeans_limpio.csv",
                   sensor_cols=['temperatura','pH','conductividad','TDS','DO_mgL'],
                   label_col = "etiqueta_kmeans",
                   preproc_dir="../data/images_preproc",
                   nombre_img_col="imagen"):
    """
    - Lee el CSV de etiquetas
    - Construye el Dataset PyTorch
    - Divide en train/val/test de forma estratificada
    - Devuelve los Subsets de train y val (test no lo usamos en la búsqueda)
    """
    # Leer CSV y forzar numérico en sensores
    df = pd.read_csv(label_csv)
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=sensor_cols).reset_index(drop=True)

    labels = df[label_col].values
    dataset = PecesDataset(df, preproc_dir, sensor_cols, label_col, nombre_img_col)

    # Primer split estratificado (train+val vs test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tv_idx, _ = next(sss1.split(np.zeros(len(labels)), labels))
    labels_tv = labels[tv_idx]

    # Segundo split estratificado (train vs val)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )
    tr_sub, val_sub = next(sss2.split(np.zeros(len(labels_tv)), labels_tv))
    train_idx = tv_idx[tr_sub]
    val_idx   = tv_idx[val_sub]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)