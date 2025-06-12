import cv2
import numpy as np

def recortar_tanque(img):
    """
    Detecta el borde circular del tanque y devuelve solo el interior.
    Para simplificar, asumimos que el tanque es casi un círculo centrado.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Detectar bordes con Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Usar HoughCircles para encontrar el círculo aproximado del borde del tanque
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=int(min(img.shape[:2]) * 0.3),
        maxRadius=int(min(img.shape[:2]) * 0.9)
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Tomamos el primer círculo detectado (x, y, r)
        x, y, r = circles[0]
        # Crear máscara circular
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        # Aplicar máscara sobre la imagen original
        masked = cv2.bitwise_and(img, img, mask=mask)
        # Recortar al bounding box del círculo
        x1, y1 = x - r, y - r
        x2, y2 = x + r, y + r
        # Asegurar límites dentro de la imagen
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
        roi = masked[y1:y2, x1:x2]
        return roi
    else:
        # Si no se detecta círculo, devolvemos la imagen completa para no perder datos
        return img.copy()

def recortar_tanque_con_desplazamiento_superior(img, desplazar_arriba=30):
    """
    Detecta el círculo del tanque, pero extiende el recorte unos píxeles hacia arriba
    para no cortar peces que nadan cerca del borde superior.
    :param img: imagen BGR completa
    :param desplazar_arriba: cuántos píxeles más incluir hacia arriba del círculo
    """
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