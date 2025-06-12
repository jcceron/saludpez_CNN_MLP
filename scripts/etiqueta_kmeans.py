import pandas as pd
from sklearn.cluster import KMeans

def etiquetar_con_kmeans(csv_input, csv_output):
    """
    Carga un dataset con columnas 'DO_mgL' y 'pH',
    aplica KMeans(k=3) sobre estas dos columnas,
    mapea los clusters a etiquetas 'enfermo', 'estres_leve', 'saludable'
    según el valor de DO del centroide,
    y guarda el dataset con la nueva columna 'etiqueta_kmeans'.
    """
    # 1) Cargar datos
    df = pd.read_csv(csv_input, parse_dates=["timestamp_imagen","timestamp_sens"])
    
    # 2) Ejecutar KMeans con k=3
    X = df[["DO_mgL","pH"]].dropna().values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    # 3) Asignar labels
    # Creamos una serie igual al índice del df, para filas con NaN mantenemos NaN
    labels = pd.Series([None] * len(df), index=df.index)
    non_na_idx = df[["DO_mgL","pH"]].dropna().index
    labels.loc[non_na_idx] = kmeans.labels_
    
    # 4) Mapear clusters a etiquetas según DO creciente del centroide
    centers = kmeans.cluster_centers_
    # Orden de clusters según DO (centers[:,0])
    order = centers[:,0].argsort()
    mapping = { order[0]: "enfermo",
                order[1]: "estres_leve",
                order[2]: "saludable" }
    df["etiqueta_kmeans"] = labels.map(mapping)
    
    # 5) Guardar nuevo CSV
    df.to_csv(csv_output, index=False)
    print(f"Dataset con etiquetas KMeans guardado en: {csv_output}")
    # Mostrar conteo de clases
    print("\nDistribución de etiquetas KMeans:")
    print(df["etiqueta_kmeans"].value_counts())
    return df

# Uso de ejemplo:
# df_nuevo = etiquetar_con_kmeans("label2.csv", "label2_kmeans.csv")

