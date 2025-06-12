import os
import pandas as pd
from datetime import datetime
from IPython.display import display

def generar_labels_csv(img_dir, sensores_csv, output_csv):
    """
    Genera un CSV que empata cada imagen con los datos de sensores más cercanos.
    :param img_dir: Carpeta con imágenes
    :param sensores_csv: CSV limpio con datos de sensores
    :param output_csv: Ruta donde guardar labels.csv
    """
    df_sens = pd.read_csv(sensores_csv)
    df_sens.columns = df_sens.columns.str.strip()  # elimina espacios en blanco
    df_sens["timestamp"] = pd.to_datetime(df_sens["timestamp"])

    registros = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".jpg", ".png")):
            continue
        ts_im = extraer_timestamp_de_nombre(fn)
        if ts_im is None:
            continue
        fila_sens = encontrar_sensor_mas_cercano(df_sens, ts_im)

        registro = {
            "imagen": fn,
            "timestamp_imagen": ts_im,
            "timestamp_sens": fila_sens["timestamp"],
            "temperatura": fila_sens.get("temperatura", None),
            "pH": fila_sens.get("pH", None),
            "conductividad": fila_sens.get("conductividad", None),
            "TDS": fila_sens.get("TDS", None),
            "salinidad": fila_sens.get("Sal.[psu]", None),
            "presion": fila_sens.get("Press.[psi]", None),
            "oxigeno_mgL": fila_sens.get("DO_mgL", None),
            "etiqueta": ""
        }
        registros.append(registro)

    df_labels = pd.DataFrame(registros)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_labels.to_csv(output_csv, index=False)
    return df_labels

def extraer_timestamp_de_nombre(fn):
    """
    Convierte un nombre como 'YYYY-MM-DD_HH-MM-SS.jpg' o con sufijo '_XXs' a datetime.
    """
    base = os.path.splitext(fn)[0]
    if "_" in base and base.split("_")[-1].endswith("s"):
        base = "_".join(base.split("_")[:-1])
    try:
        return datetime.strptime(base, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None

def encontrar_sensor_mas_cercano(df_sens, ts_image):
    """
    Dado un DataFrame con 'timestamp' y un datetime, retorna la fila más cercana.
    """
    diffs = (df_sens["timestamp"] - ts_image).abs()
    idx_min = diffs.idxmin()
    return df_sens.loc[idx_min]


def generar_labels_csv_normalizado(
    img_dir,
    sensores_norm_csv,
    output_csv,
    do_thresh=(0.4, 0.7),
    ph_thresh=(0.3, 0.8),
    max_diff_seconds=None
):
    """
    Empata imágenes con datos de sensores normalizados y clasifica según umbrales normalizados.
    
    Parámetros:
    - img_dir: carpeta con imágenes (.jpg, .png)
    - sensores_norm_csv: CSV con datos de sensores ya normalizados (columnas 'timestamp', 'temperatura','pH','conductividad','TDS','DO_mgL')
    - output_csv: ruta donde guardar el CSV resultante
    - do_thresh: tupla (stress_low, healthy_low) en escala [0,1] para DO normalizado
    - ph_thresh: tupla (ph_low, ph_high) en escala [0,1] para pH normalizado
    - max_diff_seconds: desfase máximo permitido para empatar timestamp imagen ↔ sensor (en segundos)
    
    Salida:
    - DataFrame con columnas ['imagen','timestamp_imagen','timestamp_sens',...,'etiqueta']
    """
    # Cargar sensores normalizados
    df_s = pd.read_csv(sensores_norm_csv, parse_dates=['timestamp'])
    df_s = df_s.sort_values('timestamp').reset_index(drop=True)
    
    # Leer imágenes y extraer timestamp
    registros = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(('.jpg', '.png')):
            continue
        base = os.path.splitext(fn)[0]
        parts = base.split('_')
        if parts[-1].endswith('s'):
            parts = parts[:-1]  # eliminar sufijo "_XXs"
        try:
            ts_img = datetime.strptime('_'.join(parts), "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        registros.append({'imagen': fn, 'timestamp_imagen': ts_img})
    df_img = pd.DataFrame(registros).sort_values('timestamp_imagen')
    
    # Empate con merge_asof
    df_labels = pd.merge_asof(
        df_img,
        df_s.rename(columns={'timestamp':'timestamp_sens'}),
        left_on='timestamp_imagen',
        right_on='timestamp_sens',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=max_diff_seconds) if max_diff_seconds else None
    )
    
    # Clasificación usando umbrales normalizados
    low_do, high_do = do_thresh
    low_ph, high_ph = ph_thresh
    
    def clasificar_norm(fila):
        do = fila['DO_mgL']
        ph = fila['pH']
        if pd.isna(do) or pd.isna(ph):
            return 'sin_dato'
        if (do >= high_do) and (low_ph <= ph <= high_ph):
            return 'saludable'
        if (low_do <= do < high_do) or not (low_ph <= ph <= high_ph):
            return 'estres_leve'
        return 'enfermo'
    
    df_labels['etiqueta'] = df_labels.apply(clasificar_norm, axis=1)
    
    # Guardar CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_labels.to_csv(output_csv, index=False)
    print(f"Labels guardados en: {output_csv}")
    display(df_labels.head())
    return df_labels

# Ejemplo:
# df_labels = generar_labels_csv_normalizado(
#     img_dir="../data/images",
#     sensores_norm_csv="../data/sensores_preprocesado.csv",
#     output_csv="../data/labels_norm.csv",
#     do_thresh=(0.4,0.7),
#     ph_thresh=(0.3,0.8),
#     max_diff_seconds=30
# )

