import os
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scripts.prep_auxiliares import (
    recortar_tanque,      # HoughCircles + desplazamiento
    eliminar_timestamp,   # pintado negro de la zona de timestamp
    balance_clahe,        # balance de blancos + CLAHE
    quitar_sombras,       # máscara HSV + morfología
    segmentar_peces,      # umbrales HSV/LAB + morfología
    separar_watershed     # Watershed + fallback
)

def visualize_preprocessing_samples(
    label_csv,
    raw_img_dir,
    num_samples=3,
    # Hiperparámetros configurables:
    desplazar_arriba=50,
    timestamp_rect=(0, 0, 250, 100),
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    hsv_ranges=None,
    lab_ranges=None,
    min_contour_area=20,
    watershed_dist_thresh=0.4,
    final_size=(224, 224)
):
    """
    Visualiza pasos de preprocesamiento para muestras aleatorias de cada clase.

    Parámetros:
      - label_csv: ruta al CSV con 'imagen' y 'etiqueta_kmeans'.
      - raw_img_dir: carpeta con imágenes originales.
      - num_samples: número de muestras por clase.

    Hiperparámetros:
      - desplazar_arriba: desplazamiento vertical en recorte de tanque.
      - timestamp_rect: rectángulo (x,y,w,h) para eliminar timestamp.
      - clahe_clip_limit: parámetro clipLimit para CLAHE.
      - clahe_tile_grid_size: parámetro tileGridSize para CLAHE.
      - hsv_ranges: lista de rangos HSV para segmentación, e.g. [(h1,s1,v1,h2,s2,v2),...].
      - lab_ranges: lista de rangos LAB para segmentación, e.g. [(l1,l2,a1,a2,b1,b2),...].
      - min_contour_area: área mínima para filtrar contornos antes de Watershed.
      - watershed_dist_thresh: umbral en distancia para semillas de Watershed.
      - final_size: tamaño (w,h) al redimensionar la imagen final.
    """
    # Valores por defecto si no se pasan
    if hsv_ranges is None:
        hsv_ranges = [(0,60,60,30,255,255), (150,60,60,180,255,255)]
    if lab_ranges is None:
        lab_ranges = [(140,255,108,186,92,194)]

    # Leer dataset con etiquetas
    df = pd.read_csv(label_csv)
    clases = df['etiqueta_kmeans'].dropna().unique()

    pasos = ['Original', 'Recorte', 'SinTimestamp', 'Balance+CLAHE',
             'SinSombras', 'MaskClean', 'MaskSep', 'Final']

    for clase in clases:
        imgs = df[df['etiqueta_kmeans'] == clase]['imagen'].tolist()
        muestras = random.sample(imgs, min(num_samples, len(imgs)))

        fig, axes = plt.subplots(len(muestras), len(pasos),
                                 figsize=(4*len(pasos), 4*len(muestras)))
        fig.suptitle(f"Preprocesamiento - Clase: {clase}", fontsize=18)

        for i, img_name in enumerate(muestras):
            # Cargar y convertir BGR->RGB
            src = os.path.join(raw_img_dir, img_name)
            img = cv2.imread(src)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Paso a paso con parámetros externos
            roi         = recortar_tanque(img, desplazar_arriba)
            ts          = eliminar_timestamp(roi, timestamp_rect)
            eq          = balance_clahe(ts, clip_limit=clahe_clip_limit,
                                         tile_grid_size=clahe_tile_grid_size)
            nos         = quitar_sombras(eq)
            mask_clean  = segmentar_peces(nos,
                                          umbral_L=None, delta_ab=None,
                                          hsv_ranges=hsv_ranges,
                                          lab_ranges=lab_ranges)
            mask_sep    = separar_watershed(mask_clean,
                                           min_area=min_contour_area,
                                           dist_thresh=watershed_dist_thresh)

            seg_clean   = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2RGB)
            seg_sep     = cv2.cvtColor(mask_sep, cv2.COLOR_GRAY2RGB)
            final_img   = cv2.resize(
                cv2.bitwise_and(nos, nos, mask=mask_sep),
                final_size
            )
            final_rgb   = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

            imgs_to_show = [img_rgb,
                            cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
                            cv2.cvtColor(ts, cv2.COLOR_BGR2RGB),
                            cv2.cvtColor(eq, cv2.COLOR_BGR2RGB),
                            cv2.cvtColor(nos, cv2.COLOR_BGR2RGB),
                            seg_clean, seg_sep, final_rgb]

            for j, paso in enumerate(pasos):
                ax = axes[i, j] if len(muestras) > 1 else axes[j]
                ax.imshow(imgs_to_show[j])
                if i == 0:
                    ax.set_title(paso, fontsize=14)
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



