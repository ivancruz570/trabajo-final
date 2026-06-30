import cv2
import numpy as np


def detectar_circulos(imagen):
    """
    Detecta los círculos del examen utilizando
    la Transformada Circular de Hough.
    """

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Suavizar la imagen para eliminar ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 2)

    # Detectar círculos
    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=100
        param2=30
        minRadius=10,
        maxRadius=30
    )

    lista = []

    if circulos is not None:

        circulos = np.round(circulos[0, :]).astype("int")

        for (x, y, r) in circulos:
            lista.append((x, y, r))

    # Ordenar primero por filas y luego por columnas
    lista = sorted(lista, key=lambda c: (c[1], c[0]))

    return lista


def mostrar_circulos(imagen, circulos):
    """
    Dibuja todos los círculos detectados.
    """

    copia = imagen.copy()

    for (x, y, r) in circulos:

        cv2.circle(copia, (x, y), r, (0, 255, 0), 2)

        cv2.circle(copia, (x, y), 2, (0, 0, 255), 3)

    return copia


def imprimir_circulos(circulos):

    print()

    print("===== CIRCULOS DETECTADOS =====")

    for i, (x, y, r) in enumerate(circulos):

        print(f"{i+1}: Centro=({x},{y}) Radio={r}")
