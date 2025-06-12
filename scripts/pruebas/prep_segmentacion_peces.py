import cv2
import numpy as np

# Umbral por color en espacios HSV

def segmentar_peces_por_color(img, lower1, upper1, lower2, upper2):
    """
    Segmenta peces usando dos rangos en HSV: naranja y rojo.
    Devuelve la máscara binaria y la imagen segmentada (solo peces).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask_orange = cv2.bitwise_or(mask1, mask2)

    # Suavizar pero no borrar demasiado
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_close = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel, iterations=1)

    segmented = cv2.bitwise_and(img, img, mask=mask_clean)
    return mask_clean, segmented, hsv


def segmentar_peces_color_gris_naranja(
    img,
    bounds_color1, bounds_color2,
    umbral_L=180,
    delta_ab=20
):
    """
    Segmenta peces combinando:
      1) Rango HSV para tonos naranja/rojo.
      2) Rango en espacio LAB para tonos grises/plateados,
         asegurando que |a-128| y |b-128| sean bajos (tonos neutros).
    Parámetros:
      - img: imagen BGR corregida (balance + contraste).
      - bounds_color1: (lower1, upper1) HSV para anaranjados.
      - bounds_color2: (lower2, upper2) HSV para rojizos.
      - umbral_L: mínimo en canal L para considerar “claro/plateado”.
      - delta_ab: tolerancia en a/b alrededor de 128 para tono neutro.
    """
    # 1) Detectar naranja/rojo en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1, upper1 = bounds_color1
    lower2, upper2 = bounds_color2
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask_naranja = cv2.bitwise_or(mask1, mask2)

    # 2) Detectar gris/plateado en LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_chan, a_chan, b_chan = cv2.split(lab)
    mask_L = cv2.inRange(L_chan, umbral_L, 255)
    mask_a = cv2.inRange(a_chan, 128 - delta_ab, 128 + delta_ab)
    mask_b = cv2.inRange(b_chan, 128 - delta_ab, 128 + delta_ab)
    mask_gris = cv2.bitwise_and(mask_L, cv2.bitwise_and(mask_a, mask_b))

    # 3) Combinar máscaras
    mask_comb = cv2.bitwise_or(mask_naranja, mask_gris)

    # 4) Morfología básica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_close = cv2.morphologyEx(mask_comb, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) Segmentado “crudo”
    segmented_crudo = cv2.bitwise_and(img, img, mask=mask_clean)

    return mask_clean, segmented_crudo, hsv, lab


# Sustracción de fondo

# Si el fondo (agua + algas) es relativamente estático y las imágenes tienen
# cambios mínimos de iluminación, usar:

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

def segmentar_por_sustraccion(img):
    """
    Usa un modelito MOG2 para estimar fondo y extraer “movimiento”
    (útil si cargas varias imágenes seguidas).
    """
    fgmask = fgbg.apply(img)
    # Elimina ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Extraer primer plano
    res = cv2.bitwise_and(img, img, mask=fgmask)
    return fgmask, res
