import cv2
import numpy as np


class ProcessImage():
    def __init__(self):
        self.bgr = None

    def filter_bw(self, img, cor_menor, cor_maior):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, cor_menor, cor_maior)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, arvore = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        return contornos

    def filter_hsv(self, img, cor_menor, cor_maior):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cor_menor, cor_maior)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, arvore = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        return contornos

    def get_center(self, contornos):
        centros = []

        for contorno in contornos:
            M = cv2.moments(contorno)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centros.append((cX, cY))

        return centros

    def find_latinhas(self, img, contornos):
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
        latinhas = contornos[:4]

        img_contornos = img.copy()
        cv2.drawContours(img_contornos, latinhas, -1, (255, 0, 0), 3)

        return img_contornos, latinhas

    def find_latinha_life(self, img, latinhas):
        centros_latinhas = self.get_center(latinhas)

        cor_menor = np.array([35, 50, 50])
        cor_maior = np.array([90, 255, 255])

        contornos_verdes = self.filter_hsv(img, cor_menor, cor_maior)
        maior_verde = max(contornos_verdes, key=cv2.contourArea)
        centro_verde = self.get_center([maior_verde])[0]

        menor_distancia = None
        indice_life = 0

        for i in range(len(centros_latinhas)):
            x1, y1 = centros_latinhas[i]
            x2, y2 = centro_verde

            distancia = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            if menor_distancia is None or distancia < menor_distancia:
                menor_distancia = distancia
                indice_life = i

        latinha_life = latinhas[indice_life]
        cv2.drawContours(img, [latinha_life], -1, (0, 255, 0), 5)

        return img, latinha_life

    def run_image(self, img):
        contornos = self.filter_bw(img, 0, 220)
        img, latinhas = self.find_latinhas(img, contornos)
        img, latinha_life = self.find_latinha_life(img, latinhas)

        self.bgr = img

    def show_image(self):
        cv2.imshow("image", self.bgr)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    img = cv2.imread("img/coke-cans.jpg")

    processor = ProcessImage()
    processor.run_image(img)
    processor.show_image()


if __name__ == "__main__":
    main()
