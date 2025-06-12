import cv2
import numpy as np

def separar_peces_con_watershed(mask_filtered):
    """
    Recibe:
      - mask_filtered: máscara binaria (uint8, 0-255) después del filtrado
        por área (solo peces grandes, sin motas pequeñas).
    Devuelve:
      - markers_ws: máscara etiquetada (int32) en la que cada región
        corresponde a un pez separado (0 = fondo, 1..N = distintos “objetos”).
    """

    # 1) Dado que mask_filtered puede contener huecos internos, dilatémosla ligeramente
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_dil = cv2.dilate(mask_filtered, kernel, iterations=2)

    # 2) Transformada de distancia: pixeles blancos pasan a valor de dist. al borde
    dist_transform = cv2.distanceTransform(mask_dil, cv2.DIST_L2, 5)

    # 3) Encontrar picos (local maxima) en la transformada de distancia
    #    para usarlos como “semillas” en el watershed:
    ret, sure_fg = cv2.threshold(dist_transform,
                                 dist_transform.max() * 0.4,  # puedes ajustar 0.6–0.7
                                 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 4) Encontrar “fondo seguro” (pixeles de fondo a cierta distancia):
    sure_bg = cv2.dilate(mask_dil, kernel, iterations=3)

    # 5) Región “desconocida” = fondo seguro - primer plano seguro
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6) Etiquetar cada componente conectada en “primer plano seguro”
    ret, markers = cv2.connectedComponents(sure_fg)

    # 7) Incrementar todos los marcadores en 1, de modo que 0 quede reservado para el fondo
    markers = markers + 1

    # 8) Marcar la región “desconocida” como 0
    markers[unknown == 255] = 0

    # 9) Aplicar Watershed sobre la imagen original (en color)
    #    Para watershed hay que usar una imagen BGR como guía:
    img_bgr = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img_bgr, markers)

    # markers_ws contendrá: 
    #   - Valor -1 en los bordes de “corte” entre regiones
    #   - Valores 1,2,3... para cada pieza separada 
    #   - Valor 0 en el fondo
    return markers_ws
