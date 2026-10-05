import cv2
import numpy as np


class Atividade1():
    def __init__(self):
        self.f = 0
        self.H = 0.127

        self.ciano = {
            'lower': np.array([80, 100, 100]),
            'upper': np.array([100, 255, 255])
        }

        self.magenta = {
            'lower': np.array([140, 100, 100]),
            'upper': np.array([170, 255, 255])
        }

        self.kernel = np.ones((5, 5), np.uint8)

    def encontrar_foco(self, D: float, H: float, h: float):
        f = h * D / H
        return f

    def encontrar_centros(self, bgr: np.ndarray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask_ciano = cv2.inRange(
            hsv, self.ciano['lower'], self.ciano['upper']
        )
        mask_magenta = cv2.inRange(
            hsv, self.magenta['lower'], self.magenta['upper']
        )

        mask_ciano = cv2.morphologyEx(
            mask_ciano, cv2.MORPH_OPEN, self.kernel
        )
        mask_ciano = cv2.morphologyEx(
            mask_ciano, cv2.MORPH_CLOSE, self.kernel
        )
        mask_magenta = cv2.morphologyEx(
            mask_magenta, cv2.MORPH_OPEN, self.kernel
        )
        mask_magenta = cv2.morphologyEx(
            mask_magenta, cv2.MORPH_CLOSE, self.kernel
        )

        contornos_ciano, _ = cv2.findContours(
            mask_ciano, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contornos_magenta, _ = cv2.findContours(
            mask_magenta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contornos_ciano) == 0 or len(contornos_magenta) == 0:
            return None, None

        maior_ciano = max(contornos_ciano, key=cv2.contourArea)
        maior_magenta = max(contornos_magenta, key=cv2.contourArea)

        M_ciano = cv2.moments(maior_ciano)
        M_magenta = cv2.moments(maior_magenta)

        if M_ciano['m00'] == 0 or M_magenta['m00'] == 0:
            return None, None

        centro_ciano = (
            int(M_ciano['m10'] / M_ciano['m00']),
            int(M_ciano['m01'] / M_ciano['m00'])
        )
        centro_magenta = (
            int(M_magenta['m10'] / M_magenta['m00']),
            int(M_magenta['m01'] / M_magenta['m00'])
        )

        return centro_ciano, centro_magenta

    def calcular_h(self, centro_ciano: tuple, centro_magenta: tuple):
        x1, y1 = centro_ciano
        x2, y2 = centro_magenta
        distancia = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distancia

    def encontrar_D(self, f: float, H: float, h: float):
        D = f * H / h
        return D

    def calcular_theta(self, centro_ciano: tuple, centro_magenta: tuple):
        x_ciano, y_ciano = centro_ciano
        x_magenta, y_magenta = centro_magenta

        delta_x = x_ciano - x_magenta
        delta_y = y_ciano - y_magenta

        angulo = np.degrees(np.arctan2(delta_x, delta_y))