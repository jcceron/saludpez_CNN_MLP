import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import math

def cargar_y_limpiar_sensores(excel_path, output_csv_path, hoja=None):
    """
    Lee un archivo Excel con datos de sensores, limpia y guarda un CSV.
    :param excel_path: Ruta al archivo sensores.xlsx
    :param output_csv_path: Ruta donde se guardará sensors_limpio.csv
    """
    # Lee la hoja de Excel (se asume que está en la primera hoja)
    df = pd.read_excel(excel_path, sheet_name=hoja, engine="openpyxl")

    # Convierte la columna 'timestamp' a tipo datetime (dd/mm/yyyy HH:MM)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

    # Elimina filas con timestamp inválido
    df = df.dropna(subset=["timestamp"])

    # Renombra columnas
    df = df.rename(columns={
        "Temp": "temperatura",
        "pH": "pH",
        "D.O.[mg/L]": "DO_mgL",
        "D.O.[%]": "DO_pct",
        "EC": "conductividad",
        "RES[KOhm-cm]": "resistencia",
        "TDS [ppt]": "TDS",
    })

    # Elimina columnas innecesarias si existen
    columnas_a_omitir = [
        "mV[pH]",
        "ORP[mV]",
        "EC Abs",
        "resistencia",
        "Sigma T[sT]",
        "DO_pct"
    ]

    #df = df.drop(columns=[col for col in columnas_a_omitir if col in df.columns], errors="ignore")
    df = df.drop(columns=[col for col in columnas_a_omitir if col in df.columns])

    # Ordena por timestamp y elimina duplicados
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    # Guarda el CSV limpio
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    return df


def clean_and_analyze_sensores(excel_path, sheet_name, output_csv_path, resample_interval='5S'):
    """
    Carga, limpia y analiza paso a paso un dataset de sensores acuícolas.
    En cada etapa imprime información y genera gráficos para diagnóstico.
    
    Parámetros:
    - excel_path: ruta al archivo .xlsx con datos crudos
    - sheet_name: nombre de la hoja en el Excel
    - output_csv_path: ruta donde guardar el CSV limpio
    - resample_interval: intervalo para resample/interpolar (por defecto '5S')
    
    Devuelve:
    - df_clean: DataFrame limpio y preprocesado
    """
    # 1) Carga de datos
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = df.columns.str.strip()
    print("=== 1) Datos crudos cargados ===")
    print("Columnas:", df.columns.tolist())
    print("Forma:", df.shape)
    display(df.head())
    df.info()

    # Seleccionar sólo columnas numéricas (excluyendo timestamp)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    # Histograma de valores iniciales
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    
    # Histograma de valores iniciales con subplots dinámicos
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()
    
    for ax, col in zip(axes, numeric_cols):
        df[col].hist(ax=ax, bins=30)
        ax.set_title(col)
    
    # Eliminar ejes sobrantes
    for ax in axes[n:]:
        fig.delaxes(ax)
    
    fig.suptitle("Distribución inicial de variables (crudas)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # 2) Timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    n_invalid_ts = df['timestamp'].isna().sum()
    print(f"\n=== 2) Conversión de timestamp (invalidos: {n_invalid_ts}) ===")
    df = df.dropna(subset=['timestamp'])
    
    # 3) Renombrar y eliminar columnas innecesarias
    df = df.rename(columns={
        "Temp": "temperatura",
        "pH": "pH",
        "D.O.[mg/L]": "DO_mgL",
        "D.O.[%]": "DO_pct",
        "EC": "conductividad",
        "RES[KOhm-cm]": "resistencia",
        "TDS [ppt]": "TDS",
    })
    #df.columns = df.columns.str.strip()
    to_drop = ["mV[pH]", "ORP[mV]", "EC Abs", "resistencia", "Sigma T[sT]", "DO_pct"]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    print("\n=== 3) Columnas tras renombrar y eliminar innecesarias ===")
    print(df.columns.tolist())
    
    # 4) Filtrado de rangos físicos
    cond = (
        df['pH'].between(0, 14) &
        df['DO_mgL'].between(0, 20) &
        (df['conductividad'] > 0) &
        (df['TDS'] > 0)
    )
    before = len(df)
    df = df[cond]
    after = len(df)
    print(f"\n=== 4) Filtrado de rangos físicos (filtradas: {before-after}) ===")
    
    # Gráficos tras filtrado
    fig2 = plt.figure(figsize=(12,8))
    df[['pH','DO_mgL','conductividad','TDS']].hist(ax=fig2.subplots(nrows=2, ncols=2), bins=30)
    fig2.suptitle("Distribución tras filtrado de rangos físicos", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # 5) Detección y eliminación de outliers (IQR)
    print("\n=== 5) Detección de outliers (IQR) ===")
    outlier_counts = {}
    for col in ['temperatura','pH','conductividad','TDS','DO_mgL']:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        mask = df[col].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        outlier_counts[col] = len(df) - mask.sum()
        df = df[mask]
    print("Conteo de outliers eliminados por variable:", outlier_counts)
    
    # Boxplots tras outliers
    plt.figure(figsize=(12,6))
    df[['temperatura','pH','conductividad','TDS','DO_mgL']].boxplot()
    plt.title("Boxplots tras eliminación de outliers")
    plt.show()
    
    # 6) Manejo de valores faltantes
    print("\n=== 6) Valores faltantes por columna antes de resample/interpolación ===")
    print(df.isna().sum())
    
    df = df.set_index('timestamp').sort_index()
    df_resampled = df.resample(resample_interval).mean()
    df_interpolated = df_resampled.interpolate(method='time')
    print("\nValores faltantes tras resample/interpolación:")
    print(df_interpolated.isna().sum())
    
    # Serie temporal de una variable (pH)
    plt.figure(figsize=(10,4))
    df_interpolated['pH'].plot(title="pH tras resample/interpolación")
    plt.ylabel("pH")
    plt.show()
    
    # 7) Normalización Min-Max
    print("\n=== 7) Normalización Min-Max ===")
    norm_cols = ['temperatura','pH','conductividad','TDS','DO_mgL']
    df_norm = (df_interpolated[norm_cols] - df_interpolated[norm_cols].min()) / \
              (df_interpolated[norm_cols].max() - df_interpolated[norm_cols].min())
    display(df_norm.describe())
    
    # Histogramas normalizados
    m = len(norm_cols)
    ncols2 = 3
    nrows2 = math.ceil(m / ncols2)
    fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(12, 4 * nrows2))
    axes2 = axes2.flatten()
    for ax, col in zip(axes2, norm_cols):
        df_norm[col].hist(ax=ax, bins=30)
        ax.set_title(col)
    for ax in axes2[m:]:
        fig2.delaxes(ax)
    fig2.suptitle("Histograma de variables normalizadas", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # 8) Guardar CSV limpio
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_norm.to_csv(output_csv_path, index_label='timestamp')
    print(f"\nCSV limpio guardado en: {output_csv_path}")
    
    return df_norm

# Ejemplo de uso:
# df_clean = clean_and_analyze_sensores(
#     excel_path="../data/sensores.xlsx",
#     sheet_name="nov17",
#     output_csv_path="../data/sensores_preprocesado.csv",
#     resample_interval="5S"
# )

