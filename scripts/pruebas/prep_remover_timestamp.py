import cv2
import numpy as np

def eliminar_timestamp_manualmente(img, alto_ts=5, ancho_ts=250):
    """
    Elimina el rectángulo superior izquierdo donde suele ir el timestamp.
    Ajusta alto_ts y ancho_ts según el tamaño real del timestamp.
    """
    img2 = img.copy()
    h, w = img2.shape[:2]
    h_ts = min(alto_ts, h)
    w_ts = min(ancho_ts, w)
    img2[:h_ts, :w_ts] = 0
    return img2

def eliminar_timestamp_por_color(img):
    """
    Quita pixeles de color verde brillante (timestamp) y los reemplaza por negro.
    Ajusta los rangos de HSV según el tono exacto del texto.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Rango aproximado para verde brillante, ajústalo si es otro tono:
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Invertir máscara para conservar todo salvo el texto verde
    mask_no_green = cv2.bitwise_not(mask_green)
    result = cv2.bitwise_and(img, img, mask=mask_no_green)
    return result
