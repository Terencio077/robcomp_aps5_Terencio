# APS 5 — Identificação e Modelo de Câmera

## Integrantes

- Guilherme Terencio Machado

## Dependências

```bash
pip install opencv-python numpy
```

## Exercício 1 — Latinhas

O programa encontra os contornos das quatro latinhas e destaca em verde a
Coca-Cola Life.

Para processar e exibir as três imagens oficiais:

```bash
python3 latinhas.py
```

Os resultados também são salvos em `img/resultados_latinhas/`.

**Link do vídeo:** https://youtu.be/G9HlS2NcHIY

## Exercício 2 — Vanishing Point

Para processar a imagem oficial do handout:

```bash
python3 vanishing_point.py img/frame01.jpg
```

Para processar a foto feita no laboratório e salvar a versão anotada:

```bash
python3 vanishing_point.py img/pista.jpeg --output img/pista_vp.jpeg
```

**Link do vídeo:** https://youtu.be/DhOELOIcukg

## Exercício 3 — Estimando Pose

O programa detecta os círculos ciano e magenta, calcula a distância entre os
centros em pixels, estima a distância da câmera e calcula o ângulo em relação à
vertical.

Antes de gravar, confira no início de `ex3.py` os valores medidos:

```python
DISTANCIA_CALIBRACAO = 0.80
DISTANCIA_ENTRE_CIRCULOS = 0.127
```

As duas medidas precisam estar na mesma unidade. Os valores acima estão em
metros e correspondem ao exemplo de 80 cm e 12,7 cm.

Para testar uma imagem do handout:

```bash
python3 ex3.py --image img/angulo02.jpg --calibration-image img/calib01.jpg
```

Para executar com a webcam:

```bash
python3 ex3.py
```

Na janela da webcam, coloque o padrão na distância de calibração configurada e
pressione `C`. Depois, varie a distância e o ângulo da folha. Pressione `Q` para
encerrar.

**Link do vídeo:** Não realizado

## Observações

- O vídeo não deve ser adicionado ao Git; somente o link público do YouTube.
- A foto da pista feita no laboratório e sua versão anotada estão em
  `img/pista.jpeg` e `img/pista_vp.jpeg`.
