# scripts/deap_splitDataset.py
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
# Separación de conjunto de datos en train+val vs test

def split_dataset(dataset, labels,
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  random_state: int = 42):
    """
    Divide un Dataset de PyTorch en tres Subsets estratificados: train, val y test.
    """
    # 1) Separa el conjunto completo en train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1,
                                  test_size=test_size,
                                  random_state=random_state)
    idx_tv, idx_test = next(
        sss1.split(np.zeros(len(labels)), labels)
    )
    labels_tv = labels[idx_tv]

    # 2) Dentro de train+val separa train vs val
    sss2 = StratifiedShuffleSplit(n_splits=1,
                                  test_size=val_size/(1-test_size),
                                  random_state=random_state)
    idx_train, idx_val = next(
        sss2.split(np.zeros(len(labels_tv)), labels_tv)
    )

    # 3) Convierte índices relativos a absolutos
    train_idx = idx_tv[idx_train]
    val_idx   = idx_tv[idx_val]

    # 4) Devuelve tres Subsets de PyTorch
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, idx_test))
