import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# recortar_tanque_con_desplazamiento_superior:
#   Detecta el círculo del tanque vía Hough, lo enmascara y recorta.
# ─────────────────────────────────────────────────────────────────────────
def recortar_tanque_con_desplazamiento_superior(img, desplazar_arriba=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray_blur, 50, 150)
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
        roi = masked[y1:y2, x1:x2]
        return roi
    else:
        return img.copy()


# ─────────────────────────────────────────────────────────────────────────
# eliminar_timestamp_manualmente:
#   Pinta de negro el rectángulo superior izquierdo donde aparece el timestamp.
# ─────────────────────────────────────────────────────────────────────────
def eliminar_timestamp_manualmente(img, alto_ts=5, ancho_ts=250):
    img2 = img.copy()
    h, w = img2.shape[:2]
    h_ts = min(alto_ts, h)
    w_ts = min(ancho_ts, w)
    img2[:h_ts, :w_ts] = 0
    return img2


# ─────────────────────────────────────────────────────────────────────────
# balance_blancos_simple:
#   Aplica corrección de balance de blancos muy sencilla por canales RGB.
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
#   Aplica CLAHE sobre canal L en LAB para mejorar contraste.
# ─────────────────────────────────────────────────────────────────────────
def ajustar_brillo_contraste(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_eq


# ─────────────────────────────────────────────────────────────────────────
# separar_peces_con_watershed:
#   Ejecuta Watershed para separar blobs de peces muy unidos.
#   Modificación clave: Umbral 0.4 en lugar de 0.6 → más semillas internas.
# ─────────────────────────────────────────────────────────────────────────
def separar_peces_con_watershed(mask_filtered):
    # 1) Dilatar ligeramente para cerrar huecos internos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dil = cv2.dilate(mask_filtered, kernel, iterations=2)

    # 2) Transformada de distancia
    dist_transform = cv2.distanceTransform(mask_dil, cv2.DIST_L2, 5)

    # 3) Umbral de semillas: ahora 0.4 * max, en vez de 0.6
    #    Esto genera más regiones “seguros fg”, así Watershed no agrupa tanto los peces.
    ret, sure_fg = cv2.threshold(dist_transform, dist_transform.max() * 0.4, 255, cv2.THRESH_BINARY)
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
    img_bgr = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img_bgr, markers)

    return markers_ws
