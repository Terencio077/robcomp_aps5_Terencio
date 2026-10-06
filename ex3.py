import argparse
from pathlib import Path

import cv2
import numpy as np


# Ajuste estes valores para a sua medicao real antes de gravar o video.
# D e H precisam usar a mesma unidade. Aqui, ambos estao em metros.
DISTANCIA_CALIBRACAO = 0.80
DISTANCIA_ENTRE_CIRCULOS = 0.127


class Atividade1:
    def __init__(self):
        self.f = 0.0
        self.H = DISTANCIA_ENTRE_CIRCULOS

        self.ciano = {
            "lower": np.array([80, 100, 100], dtype=np.uint8),
            "upper": np.array([100, 255, 255], dtype=np.uint8),
        }

        self.magenta = {
            "lower": np.array([140, 100, 100], dtype=np.uint8),
            "upper": np.array([170, 255, 255], dtype=np.uint8),
        }

        self.kernel = np.ones((5, 5), dtype=np.uint8)
        self.area_minima = 50.0

    def encontrar_foco(self, D: float, H: float, h: float):
        """Calcula a distancia focal em pixels: f = h * D / H."""
        if D is None or H is None or h is None or D <= 0 or H <= 0 or h <= 0:
            return -1
        return float(h * D / H)

    def _centro_do_maior_contorno(self, mask: np.ndarray):
        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contornos = [
            contorno
            for contorno in contornos
            if cv2.contourArea(contorno) >= self.area_minima
        ]
        if not contornos:
            return None

        maior = max(contornos, key=cv2.contourArea)
        momentos = cv2.moments(maior)
        if momentos["m00"] == 0:
            return None

        return (
            int(round(momentos["m10"] / momentos["m00"])),
            int(round(momentos["m01"] / momentos["m00"])),
        )

    def encontrar_centros(self, bgr: np.ndarray):
        """Retorna os centros dos circulos ciano e magenta, nessa ordem."""
        if bgr is None or bgr.size == 0:
            return None, None

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        centros = []

        for faixa in (self.ciano, self.magenta):
            mask = cv2.inRange(hsv, faixa["lower"], faixa["upper"])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            centros.append(self._centro_do_maior_contorno(mask))

        return centros[0], centros[1]

    def calcular_h(self, centro_ciano: tuple, centro_magenta: tuple):
        """Calcula a distancia euclidiana, em pixels, entre os centros."""
        if centro_ciano is None or centro_magenta is None:
            return -1

        x1, y1 = centro_ciano
        x2, y2 = centro_magenta
        return float(np.hypot(x2 - x1, y2 - y1))

    def encontrar_D(self, f: float, H: float, h: float):
        """Estima a distancia real pela relacao D = f * H / h."""
        if f is None or H is None or h is None or f <= 0 or H <= 0 or h <= 0:
            return -1
        return float(f * H / h)

    def calcular_theta(self, centro_ciano: tuple, centro_magenta: tuple):
        """Calcula o angulo orientado do vetor magenta->ciano com a vertical."""
        if centro_ciano is None or centro_magenta is None:
            return -1

        x_ciano, y_ciano = centro_ciano
        x_magenta, y_magenta = centro_magenta
        delta_x = x_ciano - x_magenta
        delta_y = y_ciano - y_magenta

        if delta_x == 0 and delta_y == 0:
            return -1

        return float(np.degrees(np.arctan2(delta_x, delta_y)))

    def escrever_texto(
        self, bgr: np.ndarray, distancia: float, angulo: float
    ) -> np.ndarray:
        """Escreve D e theta na imagem com duas casas decimais."""
        if bgr is None:
            return bgr

        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cor = (0, 255, 0)
        espessura = 2

        texto_distancia = (
            f"D: {distancia:.2f}"
            if distancia is not None and distancia != -1
            else "D: indisponivel"
        )
        texto_angulo = (
            f"theta: {angulo:.2f} graus"
            if angulo is not None and angulo != -1
            else "theta: indisponivel"
        )

        cv2.putText(bgr, texto_distancia, (20, 35), fonte, 0.8, cor, espessura)
        cv2.putText(bgr, texto_angulo, (20, 70), fonte, 0.8, cor, espessura)
        return bgr

    def calibration(
        self, bgr: np.ndarray, D: float = None, H: float = None
    ):
        """Calibra a camera usando uma imagem e as medidas reais D e H."""
        D_real = DISTANCIA_CALIBRACAO if D is None else D
        H_real = self.H if H is None else H

        _, _, angulo, h = self.run(bgr)
        foco = self.encontrar_foco(D_real, H_real, h)
        if foco == -1:
            self.f = -1
            saida, _, _, _ = self.run(bgr)
            return saida, -1, -1, -1, -1

        self.H = float(H_real)
        self.f = foco
        saida, distancia, angulo, h = self.run(bgr)
        return saida, distancia, angulo, h, self.f

    def run(self, bgr: np.ndarray):
        """Detecta o padrao e retorna imagem, D, theta e h."""
        if bgr is None or bgr.size == 0:
            return bgr, -1, -1, -1

        saida = bgr.copy()
        centro_ciano, centro_magenta = self.encontrar_centros(bgr)

        if centro_ciano is None or centro_magenta is None:
            saida = self.escrever_texto(saida, -1, -1)
            return saida, -1, -1, -1

        h = self.calcular_h(centro_ciano, centro_magenta)
        angulo = self.calcular_theta(centro_ciano, centro_magenta)
        distancia = self.encontrar_D(self.f, self.H, h)

        cv2.circle(saida, centro_ciano, 8, (255, 255, 0), -1)
        cv2.circle(saida, centro_magenta, 8, (255, 0, 255), -1)
        cv2.line(saida, centro_magenta, centro_ciano, (0, 255, 0), 3)
        saida = self.escrever_texto(saida, distancia, angulo)

        return saida, distancia, angulo, h


def rodar_frame(
    caminho_imagem="img/calib01.jpg",
    caminho_calibracao="img/calib01.jpg",
    D=DISTANCIA_CALIBRACAO,
    H=DISTANCIA_ENTRE_CIRCULOS,
    salvar_em=None,
    exibir=True,
):
    atividade = Atividade1()
    calibracao = cv2.imread(str(caminho_calibracao))
    if calibracao is None:
        raise FileNotFoundError(f"Imagem de calibracao nao encontrada: {caminho_calibracao}")

    _, _, _, _, foco = atividade.calibration(calibracao, D=D, H=H)
    if foco == -1:
        raise RuntimeError("Nao foi possivel detectar o padrao na imagem de calibracao.")

    bgr = cv2.imread(str(caminho_imagem))
    if bgr is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho_imagem}")

    saida, distancia, angulo, h = atividade.run(bgr)
    if salvar_em is not None:
        cv2.imwrite(str(salvar_em), saida)

    if exibir:
        cv2.imshow("Estimativa de pose", saida)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return saida, distancia, angulo, h, foco


def rodar_webcam(
    camera=0,
    D=DISTANCIA_CALIBRACAO,
    H=DISTANCIA_ENTRE_CIRCULOS,
):
    """Executa em tempo real. Pressione C para calibrar e Q para sair."""
    atividade = Atividade1()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera {camera}.")

    calibrada = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if calibrada:
                exibicao, _, _, _ = atividade.run(frame)
                aviso = "C: recalibrar | Q: sair"
            else:
                exibicao = frame.copy()
                aviso = f"Padrao a D={D:.2f}; pressione C para calibrar | Q: sair"

            cv2.putText(
                exibicao,
                aviso,
                (20, exibicao.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Estimativa de pose", exibicao)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            if tecla == ord("c"):
                _, _, _, _, foco = atividade.calibration(frame, D=D, H=H)
                calibrada = foco != -1
                if not calibrada:
                    print("Padrao nao detectado. Reposicione a folha e tente novamente.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="APS 5 - estimativa de pose")
    parser.add_argument(
        "--image",
        help="Processa uma imagem; sem esta opcao, abre a webcam.",
    )
    parser.add_argument(
        "--calibration-image",
        default="img/calib01.jpg",
        help="Imagem usada para calibrar antes de processar --image.",
    )
    parser.add_argument("--distance", type=float, default=DISTANCIA_CALIBRACAO)
    parser.add_argument("--circle-distance", type=float, default=DISTANCIA_ENTRE_CIRCULOS)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", help="Caminho para salvar a imagem processada.")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if args.image:
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        rodar_frame(
            caminho_imagem=args.image,
            caminho_calibracao=args.calibration_image,
            D=args.distance,
            H=args.circle_distance,
            salvar_em=args.output,
            exibir=not args.no_display,
        )
    else:
        rodar_webcam(camera=args.camera, D=args.distance, H=args.circle_distance)


if __name__ == "__main__":
    main()
