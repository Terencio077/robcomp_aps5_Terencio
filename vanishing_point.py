import cv2
import numpy as np


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y4 - y3) / (x4 - x3)

    h1 = y1 - m1 * x1
    h2 = y3 - m2 * x3

    x = (h2 - h1) / (m1 - m2)
    y = m1 * x + h1

    return x, y


def find_vanishing_point(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 180])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)

    altura, largura = mask.shape
    mask[:altura // 2, :] = 0

    bordas = cv2.Canny(mask, 50, 150)

    linhas = cv2.HoughLinesP(
        bordas,
        1,
        np.pi / 180,
        50,
        minLineLength=50,
        maxLineGap=50
    )

    positivas = []
    negativas = []

    if linhas is None:
        return None

    for linha in linhas:
        linha = linha.flatten()
        x1, y1, x2, y2 = linha

        if x2 == x1:
            continue

        inclinacao = (y2 - y1) / (x2 - x1)
        tamanho = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        if inclinacao > 0.3:
            positivas.append((tamanho, linha))
        elif inclinacao < -0.3:
            negativas.append((tamanho, linha))

    if len(positivas) == 0 or len(negativas) == 0:
        return None

    linha_positiva = max(positivas, key=lambda item: item[0])[1]
    linha_negativa = max(negativas, key=lambda item: item[0])[1]

    ponto = line_intersection(linha_positiva, linha_negativa)
    return ponto


def main():
    img = cv2.imread('img/frame01.jpg')
    ponto = find_vanishing_point(img)

    if ponto is not None:
        ponto_img = (int(ponto[0]), int(ponto[1]))
        cv2.circle(img, ponto_img, 10, (0, 0, 255), -1)

    cv2.imshow('Vanishing Point', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
