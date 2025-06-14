# Clasificación Multimodal de Salud de Peces en Sistemas de Recirculación Controlada (RAS)

Este repositorio contiene el código y los recursos para el proyecto de **Clasificación Multimodal de Salud de Peces en Sistemas de Recirculación Controlada (RAS)**. El objetivo es desarrollar un pipeline automatizado que combina preprocesamiento rule-based de imágenes y datos de sensores fisicoquímicos con un modelo de red neuronal profunda (CNN + MLP) para clasificar la salud de los peces en tres categorías: `enfermo`, `estres_leve` y `saludable`.

## 📂 Estructura del proyecto

```bash
SALUDPEZ_CNN_MLP/
├── .venv/                             # Entorno virtual
├── data/                              # Carpeta con datos e imágenes
├── notebooks/                         # Notebooks de análisis y modelado
│   ├── models/                        # Modelos entrenados
│   │   └── best_model.pt
│   ├── pruebas/                       # Notebooks de pruebas y experimentos
│   ├── 0_test_env.ipynb               # Verificación de entorno
│   ├── 1_extraer_frames.ipynb         # Extracción de frames de vídeo
│   ├── 2_generar_labels.ipynb         # Generación de labels K-Means
│   ├── 3_etiquetado.ipynb             # Etiquetado con reglas y sensores
│   ├── 4_verificar_etiquetado.ipynb   # Verificación del etiquetado
│   ├── 5_preproc_image.ipynb          # Preprocesamiento de imágenes
│   ├── 6_modelo_CNN+MLP.ipynb         # Definición y entrenamiento del modelo
│   └── fish_health_classification.ipynb # En desarrollo actividad sección 4
├── scripts/                           # Módulos Python
│   ├── __pycache__/
│   ├── pruebas/                       # Scripts de prueba
│   ├── __init__.py
│   ├── dataset_miltimodal.py          # Dataset multimodal para PyTorch
│   ├── dataset.py                     # Dataset solo de imágenes o sensores
│   ├── etiqueta_kmeans.py             # Generación de etiquetas con K-Means
│   ├── load_sensors.py                # Limpieza y carga de datos de sensores
│   ├── match_images_sensors.py        # Emparejamiento imagen-sensor
│   ├── modelo_multimodal.py           # Definición del modelo CNN+MLP
│   ├── prep_auxiliares_v3.py          # Funciones de preprocesamiento v3
│   ├── prep_auxiliares.py             # Funciones de preprocesamiento v2
│   ├── prep_cuantitativa.py           # Cálculo de coverage y estadísticas
│   ├── prep_save_dataset.py           # Guardado de imágenes preprocesadas
│   └── prep_ver_imagenes.py           # Visualización de pasos de preprocesamiento
├── requirements.txt                   # Dependencias del proyecto
└── README.md                          # Documentación principal
 ```

## 🔧 Instalación

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/jcceron/saludpez_CNN_MLP.git
   cd /saludpez_CNN_MLP
   ```

2. **Crear entorno virtual**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/Mac
   .venv\Scripts\activate       # Windows
   ```

3. **Instalar dependencias**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🚀 Uso

### 1. Preprocesamiento de imágenes

```bash
python -m scripts.prep_save_dataset \
  --label_csv data/labels2_kmeans_limpio.csv \
  --raw_img_dir data/images \
  --output_dir data/images_preproc \
  --desplazar_arriba 60 \
  --timestamp_rect 0 0 200 5 \
  --clahe_clip_limit 3.0 \
  --clahe_tile_grid_size 8 8 \
  --hsv_ranges 0 60 60 30 255 255 150 60 60 180 255 255 \
  --lab_ranges 120 255 98 158 98 158 \
  --min_contour_area 30 \
  --watershed_dist_thresh 0.3 \
  --final_size 224 224
```

### 2. Generación de etiquetas

```bash
python -m notebooks.2_etiquetado.ipynb
```

### 3. Entrenamiento y validación

```bash
python -m notebooks.6_modelo_CNN+MLP.ipynb
```

### 4. Evaluación final

```bash
python -m notebooks.6_modelo_CNN+MLP.ipynb
```

## 📦 Datos preprocesados

Los conjuntos de datos preprocesados se pueden descargar desde la sección **Releases**:

- [images_preproc.zip (143 MB)](https://github.com/jcceron/saludpez_CNN_MLP/releases/download/v1.0.0/images_preproc.zip)  
- [images.zip (487 MB)](https://github.com/jcceron/saludpez_CNN_MLP/releases/download/v1.0.0/images.zip)  


## 📈 Resultados

* **Accuracy:** 97 %
* **Precision/Recall/F₁-score** por clase (ver Tabla en la sección de resultados)
* **Coverage:** enfermo (9.2 %), estrés leve (7.7 %), saludable (11.0 %)

Para más detalles, consulte los notebooks en la carpeta `notebooks/`.

## 📜 Licencia

Este proyecto está bajo la licencia Apache 2.0. Consulte el archivo `LICENSE` para más información.

## ✉️ Contacto

Juan C. Cerón – **[juan.ceron@ustabuca.edu.co](mailto:juan.ceron@ustabuca.edu.co)**
Repositorio: [https://github.com/jcceron/saludpez_CNN_MLP](https://github.com//jcceron/saludpez_CNN_MLP)

## Bibliografía

- Chollet, F. (2021). Deep Learning with Python (2ª ed.). Shelter Island: Manning Publications Co.

- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Canadá: O’Reilly Media, Inc.

- Medina Tobón, J. D., & Giraldo, L. F. (26 de mayo de 2022). Video footage of fish and water quality variables in a fish farming scenario, Version 2. Obtenido de figshare: https://figshare.com/articles/dataset/Video_footage_of_fish_and_water_quality_variables_in_a_fish_farming_scenario/19653321

- Szeliski, R. (2022). Computer Vision: Algorithms and Applications (2ª ed.). Obtenido de https://szeliski.org/Book/