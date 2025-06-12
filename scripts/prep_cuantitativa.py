# scripts/prep_cuantitativa.py

import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scripts.prep_auxiliares_v3 import (
    recortar_tanque,
    eliminar_timestamp,
    balance_clahe,
    quitar_sombras,
    segmentar_peces,
    separar_watershed
)

def visualize_preprocessing_with_coverage(
    label_csv,
    raw_img_dir,
    num_samples=5,
    # hiperparámetros (idénticos al pipeline)
    desplazar_arriba=60,
    timestamp_rect=(0,0,200,5),
    clahe_clip_limit=3.0,
    clahe_tile_grid_size=(8,8),
    hsv_ranges=[(0,60,60,30,255,255),(150,60,60,180,255,255)],
    lab_ranges=[(120,255,98,158,98,158)],
    min_contour_area=30,
    watershed_dist_thresh=0.3,
    final_size=(224,224)
):
    df = pd.read_csv(label_csv)
    clases = df['etiqueta_kmeans'].dropna().unique()

    for clase in clases:
        imgs = df[df['etiqueta_kmeans']==clase]['imagen'].tolist()
        muestras = random.sample(imgs, min(num_samples, len(imgs)))

        fig, axes = plt.subplots(len(muestras), 5, figsize=(20,4*len(muestras)))
        fig.suptitle(f"Clase: {clase}", fontsize=18)

        coverages = []
        for i, fn in enumerate(muestras):
            img = cv2.imread(os.path.join(raw_img_dir, fn))
            img = img.copy()

            # 1) Pipeline hasta mask_sep
            roi        = recortar_tanque(img, desplazar_arriba)
            ts         = eliminar_timestamp(roi, timestamp_rect)
            eq         = balance_clahe(ts, clip_limit=clahe_clip_limit,
                                         tile_grid_size=clahe_tile_grid_size)
            nos        = quitar_sombras(eq)
            mask_clean = segmentar_peces(nos,
                                         hsv_ranges=hsv_ranges,
                                         lab_ranges=lab_ranges)
            mask_sep   = separar_watershed(mask_clean,
                                           min_area=min_contour_area,
                                           dist_thresh=watershed_dist_thresh)

            # 2) Coverage directo en NumPy (solo mask_sep)
            cov = np.count_nonzero(mask_sep==255) / mask_sep.size
            coverages.append(cov)

            # 3) Mostrar: original, mask_clean, mask_sep, final overlay + texto
            axes[i,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i,0].set_title("Original"); axes[i,0].axis("off")

            axes[i,1].imshow(mask_clean, cmap="gray")
            axes[i,1].set_title("Mask Clean"); axes[i,1].axis("off")

            axes[i,2].imshow(mask_sep, cmap="gray")
            axes[i,2].set_title("Mask Sep"); axes[i,2].axis("off")

            # Overlay de máscara sobre la imagen
            overlay = cv2.bitwise_and(nos, nos, mask=mask_sep)
            axes[i,3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i,3].set_title("Overlay"); axes[i,3].axis("off")

            # Coverage como texto
            axes[i,4].text(0.5, 0.5, f"Coverage:\n{cov:.2%}",
                           ha="center", va="center", fontsize=16)
            axes[i,4].axis("off")

        # Boxplot de las coberturas mostradas
        plt.figure(figsize=(4,4))
        plt.boxplot(coverages, labels=[clase])
        plt.title("Coverage samples")
        plt.ylabel("Pct píxeles útiles")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

# Uso:
# visualize_preprocessing_with_coverage(
#     label_csv="data/labels2_kmeans_limpio.csv",
#     raw_img_dir="data/images",
#     num_samples=5,
#     desplazar_arriba=hp["desplazar_arriba"],
#     timestamp_rect=hp["timestamp_rect"],
#     clahe_clip_limit=hp["clahe_clip_limit"],
#     clahe_tile_grid_size=hp["clahe_tile_grid_size"],
#     hsv_ranges=hp["hsv_ranges"],
#     lab_ranges=hp["lab_ranges"],
#     min_contour_area=hp["min_contour_area"],
#     watershed_dist_thresh=hp["watershed_dist_thresh"],
#     final_size=hp["final_size"]
# )

def compute_coverage_direct(
    label_csv,
    raw_img_dir,
    desplazar_arriba=60,
    timestamp_rect=(0,0,200,5),
    clahe_clip_limit=3.0,
    clahe_tile_grid_size=(8,8),
    hsv_ranges=[(0,60,60,30,255,255),(150,60,60,180,255,255)],
    lab_ranges=[(120,255,98,158,98,158)],
    min_contour_area=30,
    watershed_dist_thresh=0.3
):
    """
    Cálculo de coverage directo sobre `mask_sep`, con barra de progreso
    y captura de errores por imagen.
    """
    df = pd.read_csv(label_csv)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculando coverage"):
        fn, cls = row['imagen'], row['etiqueta_kmeans']
        if pd.isna(cls):
            continue

        try:
            img = cv2.imread(os.path.join(raw_img_dir, fn))
            if img is None:
                raise RuntimeError("No se pudo leer la imagen")
            img = img.copy()

            # Pipeline
            roi     = recortar_tanque(img, desplazar_arriba).copy()
            ts      = eliminar_timestamp(roi, timestamp_rect).copy()
            eq      = balance_clahe(ts,
                                    clip_limit=clahe_clip_limit,
                                    tile_grid_size=clahe_tile_grid_size).copy()
            nos     = quitar_sombras(eq).copy()
            m_clean = segmentar_peces(nos,
                                      hsv_ranges=hsv_ranges,
                                      lab_ranges=lab_ranges).copy()
            m_sep   = separar_watershed(m_clean,
                                        min_area=min_contour_area,
                                        dist_thresh=watershed_dist_thresh).copy()

            # Coverage = pixeles de pez / total
            coverage = np.count_nonzero(m_sep == 255) / m_sep.size
            records.append({'class': cls, 'coverage': coverage})

        except Exception as e:
            tqdm.write(f"⚠️ Error en {fn}: {e}")
            continue

    return pd.DataFrame(records)