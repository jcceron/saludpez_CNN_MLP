import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def mostrar_preprocesadas_por_clase(cls, dir_root, n_samples=25, cols=5, figsize=(12, 8)):
    """
    Muestra un grid de imágenes preprocesadas para la clase dada.
    :param cls: nombre de la clase ("saludable", "estres_leve", "enfermo")
    :param dir_root: directorio raíz donde están subcarpetas por clase
    :param n_samples: cuántas imágenes mostrar (aleatorias)
    :param cols: número de columnas en el grid
    :param figsize: tamaño de la figura matplotlib
    """
    class_dir = os.path.join(dir_root, cls)
    if not os.path.isdir(class_dir):
        print(f"No existe la carpeta para la clase '{cls}'. Verifica la ruta.")
        return
    
    all_images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not all_images:
        print(f"No se encontraron imágenes en {class_dir}.")
        return
    
    # Si hay menos imágenes de las solicitadas, mostrarlas todas
    if len(all_images) < n_samples:
        muestras = all_images
        print(f"Advertencia: sólo {len(all_images)} imágenes disponibles para la clase '{cls}'. Mostrando todas.")
    else:
        np.random.seed(42)
        muestras = list(np.random.choice(all_images, size=n_samples, replace=False))
    
    rows = int(np.ceil(len(muestras) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f"Preprocesado - Clase: {cls}", fontsize=18)
    axes = axes.flatten()
    
    for ax, img_name in zip(axes, muestras):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
        except Exception as e:
            ax.set_title("Error al cargar")
            ax.axis("off")
    
    # Ocultar ejes sobrantes
    for ax in axes[len(muestras):]:
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
