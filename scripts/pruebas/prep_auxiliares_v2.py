import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# recortar_tanque_con_desplazamiento_superior:
#   Detecta el círculo del tanque vía Hough, lo enmascara y recorta.
#   Si Hough no encuentra nada, devolvemos la imagen completa.
# ─────────────────────────────────────────────────────────────────────────
def recortar_tanque_con_desplazamiento_superior(img, desplazar_arriba=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30,
        minRadius=int(min(img.shape[:2]) * 0.3),
        maxRadius=int(min(img.shape[:2]) * 0.9)
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        y1 = max(y - r - desplazar_arriba, 0)
        y2 = min(y + r, img.shape[0])
        x1 = max(x - r, 0)
        x2 = min(x + r, img.shape[1])
        return masked[y1:y2, x1:x2]
    else:
        return img.copy()


# ─────────────────────────────────────────────────────────────────────────
# eliminar_timestamp_manualmente:
#   Pinta de negro un rectángulo fijo en la esquina superior izquierda.
#   (Se deja igual, aunque se recomienda luego evaluar detección dinámica de timestamp).
# ─────────────────────────────────────────────────────────────────────────
def eliminar_timestamp_manualmente(img, alto_ts=50, ancho_ts=250):
    img2 = img.copy()
    h, w = img2.shape[:2]
    h_ts = min(alto_ts, h)
    w_ts = min(ancho_ts, w)
    img2[:h_ts, :w_ts] = 0
    return img2


# ─────────────────────────────────────────────────────────────────────────
# supresion_reflejos_gamma:
#   Aplica una corrección gamma ligera para atenuar reflejos muy brillantes
# ─────────────────────────────────────────────────────────────────────────
def supresion_reflejos_gamma(img, gamma=0.8):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


# ─────────────────────────────────────────────────────────────────────────
# balance_blancos_simple:
#   Corrección muy sencilla por canales RGB (igual que antes)
# ─────────────────────────────────────────────────────────────────────────
def balance_blancos_simple(img):
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3 + 1e-6
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────
# ajustar_brillo_contraste:
#   Aplica CLAHE sobre canal L en LAB para mejorar contraste (igual que antes).
# ─────────────────────────────────────────────────────────────────────────
def ajustar_brillo_contraste(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────────────────────────────────
# separar_peces_con_watershed:
#   Watershed más agresivo (umbral 0.25 en lugar de 0.4).
# ─────────────────────────────────────────────────────────────────────────
def separar_peces_con_watershed(mask_filtered):
    # 1) Dilatar ligeramente para cerrar huecos internos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dil = cv2.dilate(mask_filtered, kernel, iterations=2)

    # 2) Transformada de distancia
    dist_transform = cv2.distanceTransform(mask_dil, cv2.DIST_L2, 5)

    # 3) Umbral de semillas: 0.3 * max para más semillas internas
    ret, sure_fg = cv2.threshold(dist_transform, dist_transform.max() * 0.25, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 4) Dilatación para definir fondo “seguro”
    sure_bg = cv2.dilate(mask_dil, kernel, iterations=3)

    # 5) Región “desconocida” = fondo seguro - primer plano seguro
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6) Conectar componentes en sure_fg para marcadores
    ret, markers = cv2.connectedComponents(sure_fg)

    # 7) Incrementar marcadores+1 para reservar 0 al fondo
    markers = markers + 1

    # 8) Marcar “región desconocida” como 0
    markers[unknown == 255] = 0

    # 9) Aplicar Watershed
    # Convertimos mask_filtered a BGR para alimentar a watershed (no importa el color, solo se usa como base)
    img_bgr = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img_bgr, markers)

    return markers_ws


def filtro_contornos_por_forma(mask, min_area_abs=80, min_circularity=0.10, aspect_ratio_range=(1.0, 8.0)):
    """
    mask: máscara binaria (uint8) con regiones a filtrar.
    min_area_abs: área mínima absoluta para considerar un contorno (en píxeles).
    min_circularity: 4π·A/(P²) mínima (0-1). Valores bajos (0.1) permiten formas oblongas.
    aspect_ratio_range: (min_ar, max_ar) relación ancho/alto para aceptar contorno.
    """
    mask_out = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_abs:
            continue

        perim = cv2.arcLength(cnt, True) + 1e-6
        circularity = 4 * np.pi * (area / (perim * perim))

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / (h + 1e-6)

        if (circularity >= min_circularity) and (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            cv2.drawContours(mask_out, [cnt], -1, 255, -1)

    return mask_out

