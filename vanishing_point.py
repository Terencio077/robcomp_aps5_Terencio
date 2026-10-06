import argparse
from pathlib import Path

import cv2
import numpy as np


def line_intersection(line1, line2):
    """Retorna a intersecao de duas retas ou None se forem paralelas."""
    x1, y1, x2, y2 = map(float, line1)
    x3, y3, x4, y4 = map(float, line2)

    denominador = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominador) < 1e-9:
        return None

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominador
    y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominador
    return float(x), float(y)


def _linhas_da_pista(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Faixa para linhas brancas: brilho alto e saturacao baixa.
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 160], dtype=np.uint8),
        np.array([180, 100, 255], dtype=np.uint8),
    )

    altura, largura = mask.shape
    # Mantem a faixa onde as bordas da pista aparecem. Cortar metade da
    # imagem eliminava justamente os trechos proximos ao ponto de fuga em
    # fotos verticais, enquanto usar a parte inferior incluia juntas do piso.
    mask[: int(altura * 0.25), :] = 0
    mask[int(altura * 0.60) :, :] = 0
    bordas = cv2.Canny(mask, 50, 150)

    linhas = cv2.HoughLinesP(
        bordas,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=max(40, largura // 12),
        maxLineGap=max(30, largura // 18),
    )
    if linhas is None:
        return [], []

    positivas = []
    negativas = []
    for linha in linhas.reshape(-1, 4):
        x1, y1, x2, y2 = linha
        delta_x = x2 - x1
        delta_y = y2 - y1
        if delta_x == 0:
            continue

        inclinacao = delta_y / delta_x
        comprimento = float(np.hypot(delta_x, delta_y))
        meio_x = (x1 + x2) / 2

        # A borda direita tem inclinacao positiva e fica majoritariamente na
        # metade direita; a esquerda segue o caso simetrico. O limite superior
        # descarta linhas quase verticais do piso e dos objetos do laboratorio.
        if 0.25 < inclinacao < 1.5 and meio_x > largura * 0.55:
            positivas.append((comprimento, linha))
        elif -1.5 < inclinacao < -0.25 and meio_x < largura * 0.45:
            negativas.append((comprimento, linha))

    positivas.sort(key=lambda item: item[0], reverse=True)
    negativas.sort(key=lambda item: item[0], reverse=True)
    return positivas[:20], negativas[:20]


def find_vanishing_point(img):
    """Encontra o ponto de fuga a partir das linhas brancas da pista."""
    if img is None or img.size == 0:
        return None

    positivas, negativas = _linhas_da_pista(img)
    if not positivas or not negativas:
        return None

    altura, largura = img.shape[:2]
    candidatos = []
    pesos = []

    for comprimento_pos, linha_pos in positivas:
        for comprimento_neg, linha_neg in negativas:
            ponto = line_intersection(linha_pos, linha_neg)
            if ponto is None:
                continue

            x, y = ponto
            if -largura <= x <= 2 * largura and -altura <= y <= altura:
                candidatos.append((x, y))
                pesos.append(comprimento_pos * comprimento_neg)

    if not candidatos:
        return None

    pontos = np.asarray(candidatos, dtype=float)
    pesos = np.asarray(pesos, dtype=float)

    # A mediana fornece uma primeira estimativa robusta. Em seguida, a media
    # ponderada usa somente intersecoes proximas dessa estimativa.
    mediana = np.median(pontos, axis=0)
    distancias = np.linalg.norm(pontos - mediana, axis=1)
    mad = float(np.median(distancias))
    limite = max(10.0, 2.5 * mad)
    inliers = distancias <= limite

    if not np.any(inliers):
        return float(mediana[0]), float(mediana[1])

    ponto = np.average(pontos[inliers], axis=0, weights=pesos[inliers])
    return float(ponto[0]), float(ponto[1])


def desenhar_ponto_de_fuga(img, ponto):
    resultado = img.copy()
    if ponto is None:
        cv2.putText(
            resultado,
            "Ponto de fuga nao detectado",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return resultado

    x, y = int(round(ponto[0])), int(round(ponto[1]))
    cv2.circle(resultado, (x, y), 12, (0, 0, 255), -1)
    cv2.circle(resultado, (x, y), 20, (255, 255, 255), 2)
    cv2.putText(
        resultado,
        f"Vanishing point: ({x}, {y})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    return resultado


def processar_arquivo(caminho, caminho_saida=None, exibir=True):
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")

    ponto = find_vanishing_point(img)
    resultado = desenhar_ponto_de_fuga(img, ponto)

    if caminho_saida is not None:
        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(caminho_saida), resultado)

    if exibir:
        cv2.imshow("Vanishing Point", resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return resultado, ponto


def main():
    parser = argparse.ArgumentParser(description="APS 5 - ponto de fuga")
    parser.add_argument("imagem", nargs="?", default="img/frame01.jpg")
    parser.add_argument(
        "--output",
        help="Imagem de saida. Se omitido, usa <nome>_vp.jpg.",
    )
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    entrada = Path(args.imagem)
    saida = Path(args.output) if args.output else entrada.with_name(
        f"{entrada.stem}_vp.jpg"
    )
    _, ponto = processar_arquivo(
        entrada,
        caminho_saida=saida,
        exibir=not args.no_display,
    )

    if ponto is None:
        print("Ponto de fuga nao detectado.")
    else:
        print(f"Ponto de fuga: ({ponto[0]:.2f}, {ponto[1]:.2f})")
        print(f"Resultado salvo em: {saida}")


if __name__ == "__main__":
    main()
