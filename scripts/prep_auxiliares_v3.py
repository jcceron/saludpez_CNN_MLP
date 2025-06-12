# scripts/prep_auxiliares_v3.py

import cv2
import numpy as np

# Módulo prep_auxiliares_v3: funciones de preprocesamiento de imágenes de peceras

def recortar_tanque(img, desplazar_arriba=50):
    """
    Detecta el contorno circular del tanque, crea máscara y recorta la ROI.
    Siempre retorna una copia contigua.

    Parámetros:
      - img: array BGR de entrada
      - desplazar_arriba: píxeles para elevar el centro y no cortar peces arriba
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=gray.shape[0]/2,
        param1=50, param2=30,
        minRadius=int(gray.shape[0]/4),
        maxRadius=int(gray.shape[0]/2)
    )
    if circles is not None:
        x, y, r = np.round(circles[0,0]).astype(int)
        y = max(r, y - desplazar_arriba)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        roi = cv2.bitwise_and(img, img, mask=mask)
        x0, y0 = x - r, y - r
        cropped = roi[y0:y0+2*r, x0:x0+2*r]
        return cropped.copy()
    return img.copy()


def eliminar_timestamp(img, rect=(0, 0, 250, 5)):
    """
    Pinta de negro un rectángulo donde aparece el timestamp.
    Retorna copia.
    """
    out = img.copy()
    x, y, w, h = rect
    cv2.rectangle(out, (x, y), (x+w, y+h), (0,0,0), -1)
    return out.copy()


def balance_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Aplica balance de blancos y CLAHE en LAB.
    Retorna copia.
    """
    tmp = img.astype(np.float32)
    avg = tmp.mean()
    for c in range(3):
        tmp[:,:,c] *= (avg / (tmp[:,:,c].mean() + 1e-8))
    tmp = np.clip(tmp,0,255).astype(np.uint8)
    lab = cv2.cvtColor(tmp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    merged = cv2.merge((l2,a,b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result.copy()


def quitar_sombras(img):
    """
    Elimina sombras basadas en HSV.
    Retorna copia.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask1 = ((h>=25)&(h<=85)&(s<80)).astype(np.uint8)*255
    mask2 = ((s<40)&(v>180)).astype(np.uint8)*255
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(img, img, mask=mask_inv)
    return result.copy()


def segmentar_peces(img, hsv_ranges=None, lab_ranges=None):
    """
    Segmenta peces con umbrales HSV y LAB + morfología.
    Retorna copia.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if hsv_ranges:
        for h1,s1,v1,h2,s2,v2 in hsv_ranges:
            m = cv2.inRange(hsv, (h1,s1,v1), (h2,s2,v2))
            mask = cv2.bitwise_or(mask, m)
    if lab_ranges:
        for l1,l2,a1,a2,b1,b2 in lab_ranges:
            m = cv2.inRange(lab, (l1,a1,b1), (l2,a2,b2))
            mask = cv2.bitwise_or(mask, m)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.copy()


def separar_watershed(mask, min_area=20, dist_thresh=0.4):
    """
    Separa agrupaciones con Watershed.
    Retorna copia.
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filt = np.zeros_like(mask, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filt, [cnt], -1, 255, -1)
    if cv2.countNonZero(filt) < min_area:
        filt = mask.copy()
    dist = cv2.distanceTransform(filt, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, cv2.THRESH_BINARY)
    fg = fg.astype(np.uint8)
    bg = cv2.dilate(filt, np.ones((3,3), np.uint8), iterations=2)
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_bgr = cv2.cvtColor(filt, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_bgr, markers)
    mask_ws = (markers > 1).astype(np.uint8) * 255
    return mask_ws.copy()
