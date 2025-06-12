import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.prep_auxiliares_v3 import (
    recortar_tanque,
    eliminar_timestamp,
    balance_clahe,
    quitar_sombras,
    segmentar_peces,
    separar_watershed
)

def preprocess_and_save_dataset(
    label_csv: str,
    raw_img_dir: str,
    output_dir: str,
    # Hiperparámetros
    desplazar_arriba: int = 60,
    timestamp_rect: tuple = (0, 0, 200, 5),
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid_size: tuple = (8, 8),
    hsv_ranges: list = [(0,60,60,30,255,255), (150,60,60,180,255,255)],
    lab_ranges: list = [(120,255,98,158,98,158)],
    min_contour_area: int = 30,
    watershed_dist_thresh: float = 0.3,
    final_size: tuple = (224, 224)
):
    """
    Preprocesa y guarda imágenes según sus etiquetas,
    mostrando una barra de progreso y capturando errores por imagen.
    """
    df = pd.read_csv(label_csv)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocesando imágenes"):
        fn = row['imagen']
        etiqueta = row['etiqueta_kmeans']
        if pd.isna(etiqueta):
            continue

        clase_dir = os.path.join(output_dir, str(etiqueta))
        os.makedirs(clase_dir, exist_ok=True)

        src = os.path.join(raw_img_dir, fn)
        dst = os.path.join(clase_dir, fn)

        try:
            img = cv2.imread(src)
            if img is None:
                raise RuntimeError("Imagen no encontrada o lectura fallida")
            img = img.copy()

            # Pipeline completo con copias contiguas
            roi        = recortar_tanque(img, desplazar_arriba).copy()
            ts         = eliminar_timestamp(roi, timestamp_rect).copy()
            eq         = balance_clahe(
                            ts,
                            clip_limit=clahe_clip_limit,
                            tile_grid_size=clahe_tile_grid_size
                        ).copy()
            nos        = quitar_sombras(eq).copy()
            mask_clean = segmentar_peces(
                            nos,
                            hsv_ranges=hsv_ranges,
                            lab_ranges=lab_ranges
                        ).copy()
            mask_sep   = separar_watershed(
                            mask_clean,
                            min_area=min_contour_area,
                            dist_thresh=watershed_dist_thresh
                        ).copy()

            preproc = cv2.bitwise_and(nos, nos, mask=mask_sep).copy()
            final   = cv2.resize(preproc, final_size).copy()

            cv2.imwrite(dst, final)

        except Exception as e:
            tqdm.write(f"⚠️ Error procesando {fn}: {e}")
            continue

    print("\n✅ Preprocesamiento completo. Imágenes guardadas en:", output_dir)