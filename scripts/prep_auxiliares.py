import cv2
import numpy as np

# Módulo auxiliaries: funciones para preprocesamiento de imágenes de peceras

def recortar_tanque(img, desplazar_arriba=50):
    """
    Detecta un círculo aproximado al contorno del tanque usando HoughCircles,
    crea una máscara circular y recorta la región de interés,
    con un desplazamiento vertical adicional para no cortar peces cerca del borde.

    Parámetros:
    - img: imagen BGR de entrada
    - desplazar_arriba: pixeles para desplazar la máscara hacia arriba
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0]/2,
        param1=50,
        param2=30,
        minRadius=int(gray.shape[0]/4),
        maxRadius=int(gray.shape[0]/2)
    )
    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype(int)
        # Ajustamos el y con el desplazamiento
        y = max(r, y - desplazar_arriba)
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        # Aplicar máscara y recortar al bounding box del círculo
        roi = cv2.bitwise_and(img, img, mask=mask)
        x0, y0 = x - r, y - r
        return roi[y0:y0 + 2 * r, x0:x0 + 2 * r]
    # Si falla detección, devolver imagen original
    return img


def eliminar_timestamp(img, rect=(0, 0, 250, 5)):
    """
    Pinta de negro un rectángulo fijo donde aparece el timestamp.

    Parámetros:
    - img: imagen BGR de entrada
    - rect: (x, y, ancho, alto) del área a tapar
    """
    x, y, w, h = rect
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return out


def balance_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Aplica un balance de blancos sencillo y CLAHE en el canal L de LAB.

    Parámetros:
    - clip_limit: límite de recorte para CLAHE
    - tile_grid_size: tamaño de célula para CLAHE
    """
    # Balance de blancos por canal
    result = img.copy().astype(np.float32)
    avg_gray = np.mean(result)
    for c in range(3):
        avg_c = np.mean(result[:,:,c])
        result[:,:,c] = result[:,:,c] * (avg_gray / (avg_c + 1e-8))
    result = np.clip(result, 0, 255).astype(np.uint8)
    # CLAHE en L
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def quitar_sombras(img):
    """
    Elimina sombras verdes/verdosas pintando esas zonas de negro.

    Parámetros:
    - img: imagen BGR corregida
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Mascara de sombra verde-oliva
    mask1 = (h >= 25) & (h <= 85) & (s < 80)
    # Mascara de sombra blanc-verdosa
    mask2 = (s < 40) & (v > 180)
    mask = np.uint8((mask1 | mask2) * 255)
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Invertir y aplicar
    mask_inv = cv2.bitwise_not(mask)
    return cv2.bitwise_and(img, img, mask=mask_inv)


def segmentar_peces(
    img,
    umbral_L=140,
    delta_ab=20,
    hsv_ranges=None,
    lab_ranges=None,
    white_ranges=None
):
    """
    Segmenta los peces combinando umbrales HSV y LAB y limpiando con morfología.

    Parámetros:
    - img: imagen BGR sin sombras
    - umbral_L: umbral mínimo L en LAB
    - delta_ab: rango +/- alrededor de 128 para a y b
    - hsv_ranges: lista de tuplas (Hmin,Smin,Vmin,Hmax,Smax,Vmax)
    - lab_ranges: lista de tuplas (Lmin,Lmax,amin,amax,bmin,bmax)
    - white_ranges: lista de tuplas para peces blancos si se desea
    """
    # Inicializar máscara
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Aplicar rangos HSV
    if hsv_ranges:
        for (h1,s1,v1,h2,s2,v2) in hsv_ranges:
            m = cv2.inRange(hsv, (h1,s1,v1), (h2,s2,v2))
            mask = cv2.bitwise_or(mask, m)
    # Aplicar rangos LAB
    if lab_ranges:
        for (l1,l2,a1,a2,b1,b2) in lab_ranges:
            m = cv2.inRange(lab, (l1, a1, b1), (l2, a2, b2))
            mask = cv2.bitwise_or(mask, m)
    # Morfología para limpiar
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def separar_watershed(mask, min_area=20, dist_thresh=0.4):
    """
    Aplica Watershed para separar peces agrupados y usa fallback si no hay separación.

    Parámetros:
    - mask: máscara binaria previa (0/255)
    - min_area: área mínima de contorno para filtrar
    - dist_thresh: factor multiplicador para semillas en la transformada de distancia
    """
    # Filtrar por área mínima
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filt = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(mask_filt, [cnt], -1, 255, -1)
    # Si casi vacía, usar mask original
    if cv2.countNonZero(mask_filt) < min_area:
        mask_filt = mask.copy()
    # Preparar marcadores
    dist = cv2.distanceTransform(mask_filt, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask_filt, np.ones((3,3),np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Etiquetar marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # Watershed
    img_color = cv2.cvtColor(mask_filt, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)
    # Generar máscara final (>1 son regiones)
    mask_ws = np.uint8(markers > 1) * 255
    return mask_ws
