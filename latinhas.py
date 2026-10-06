import argparse
from pathlib import Path

import cv2
import numpy as np


class ProcessImage:
    def __init__(self):
        self.bgr = None
        self.kernel = np.ones((5, 5), dtype=np.uint8)

    def filter_bw(self, img, cor_menor, cor_maior):
        if img is None or img.size == 0:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, cor_menor, cor_maior)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return contornos

    def filter_hsv(self, img, cor_menor, cor_maior):
        if img is None or img.size == 0:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cor_menor, cor_maior)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return contornos

    def get_center(self, contornos):
        centros = []
        for contorno in contornos:
            momentos = cv2.moments(contorno)
            if momentos["m00"] != 0:
                cX = int(round(momentos["m10"] / momentos["m00"]))
                cY = int(round(momentos["m01"] / momentos["m00"]))
                centros.append((cX, cY))
        return centros

    def find_latinhas(self, img, contornos):
        img_contornos = img.copy()
        contornos_validos = [
            contorno for contorno in contornos if cv2.contourArea(contorno) > 100
        ]
        latinhas = sorted(
            contornos_validos, key=cv2.contourArea, reverse=True
        )[:4]

        if latinhas:
            cv2.drawContours(img_contornos, latinhas, -1, (255, 0, 0), 3)
        return img_contornos, latinhas

    def find_latinha_life(self, img, latinhas):
        if not latinhas:
            return img, None

        centros_latinhas = self.get_center(latinhas)
        if not centros_latinhas:
            return img, None

        cor_menor = np.array([35, 50, 50], dtype=np.uint8)
        cor_maior = np.array([90, 255, 255], dtype=np.uint8)
        contornos_verdes = self.filter_hsv(img, cor_menor, cor_maior)
        contornos_verdes = [
            contorno
            for contorno in contornos_verdes
            if cv2.contourArea(contorno) > 50
        ]
        if not contornos_verdes:
            return img, None

        maior_verde = max(contornos_verdes, key=cv2.contourArea)
        centros_verdes = self.get_center([maior_verde])
        if not centros_verdes:
            return img, None
        centro_verde = centros_verdes[0]

        distancias = [
            np.hypot(
                centro_latinha[0] - centro_verde[0],
                centro_latinha[1] - centro_verde[1],
            )
            for centro_latinha in centros_latinhas
        ]
        indice_life = int(np.argmin(distancias))
        latinha_life = latinhas[indice_life]
        cv2.drawContours(img, [latinha_life], -1, (0, 255, 0), 5)

        return img, latinha_life

    def run_image(self, img):
        if img is None or img.size == 0:
            self.bgr = img
            return self.bgr

        contornos = self.filter_bw(img, 0, 220)
        imagem, latinhas = self.find_latinhas(img, contornos)
        imagem, _ = self.find_latinha_life(imagem, latinhas)
        self.bgr = imagem
        return self.bgr

    def show_image(self, titulo="Latinhas"):
        if self.bgr is None:
            raise RuntimeError("Nenhuma imagem foi processada.")
        cv2.imshow(titulo, self.bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def processar_arquivo(caminho, pasta_saida=None, exibir=True):
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")

    processor = ProcessImage()
    resultado = processor.run_image(img)

    caminho_saida = None
    if pasta_saida is not None:
        pasta_saida = Path(pasta_saida)
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_saida = pasta_saida / f"resultado_{Path(caminho).stem}.jpg"
        cv2.imwrite(str(caminho_saida), resultado)

    if exibir:
        processor.show_image(Path(caminho).name)

    return resultado, caminho_saida


def main():
    parser = argparse.ArgumentParser(description="APS 5 - identificacao de latinhas")
    parser.add_argument(
        "imagens",
        nargs="*",
        default=[
            "img/coke-cans.jpg",
            "img/coke-cans2.PNG",
            "img/coke-cans3.PNG",
        ],
        help="Imagens que serao processadas.",
    )
    parser.add_argument("--output-dir", default="img/resultados_latinhas")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    for caminho in args.imagens:
        _, caminho_saida = processar_arquivo(
            caminho,
            pasta_saida=args.output_dir,
            exibir=not args.no_display,
        )
        print(f"Resultado salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
