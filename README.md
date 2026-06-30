# Sistema de Calificación Automática de Exámenes de Opción Múltiple

## Descripción

Este proyecto implementa un sistema de Reconocimiento Óptico de Marcas (OMR) para la calificación automática de exámenes de opción múltiple utilizando técnicas de visión por computador.

El sistema emplea la Transformada Circular de Hough para localizar los círculos correspondientes a las respuestas del examen y operaciones de conteo de píxeles y momentos para determinar cuál opción fue marcada por el estudiante.

## Objetivos

- Detectar automáticamente los círculos de respuesta.
- Identificar la respuesta seleccionada.
- Comparar las respuestas con una plantilla.
- Obtener la calificación del examen.

## Tecnologías

- Python 3
- OpenCV
- NumPy

## Estructura

images/
results/
main.py
omr.py
utils.py

## Estado del proyecto

Avance del 50%.

Actualmente el sistema:

✔ Lee imágenes.

✔ Detecta círculos mediante Transformada de Hough.

✔ Calcula el área marcada usando conteo de píxeles y momentos.

✔ Determina la opción marcada.

## Integrantes

- CRUZ CHINGUEL IVAN
- CAJO MANAYAY FERNANDO JOSE
- opencv-python
numpy
{
    "1":"A",
    "2":"C",
    "3":"D",
    "4":"B",
    "5":"A"

}

import cv2
from omr import detectar_circulos
from utils import detectar_respuestas

# Cargar imagen
<img width="938" height="643" alt="examen" src="https://github.com/user-attachments/assets/e4c61a04-ac12-4b28-81c4-b6a3159b4ed1" />

imagen = cv2.imread("images/examen.jpg")

if imagen is None:
    print("No se pudo abrir la imagen.")
    exit()

# Detectar círculos

circulos = detectar_circulos(imagen)

print("Número de círculos encontrados:", len(circulos))

# Dibujar círculos

for (x,y,r) in circulos:

    cv2.circle(imagen,(x,y),r,(0,255,0),2)
    cv2.circle(imagen,(x,y),2,(0,0,255),3)

# Detectar respuestas

respuestas = detectar_respuestas(imagen,circulos)

print()

print("Resultados")

for r in respuestas:

    print(r)

cv2.imshow("Deteccion",imagen)

cv2.waitKey(0)

cv2.destroyAllWindows()
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
        param1=50,
        param2=20,
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
        import cv2
import numpy as np
import json


def contar_pixeles(imagen_binaria, x, y, r):
    """
    Cuenta los píxeles blancos dentro de un círculo.
    """

    mascara = np.zeros(imagen_binaria.shape, dtype=np.uint8)

    cv2.circle(mascara, (x, y), r, 255, -1)

    region = cv2.bitwise_and(imagen_binaria, imagen_binaria, mask=mascara)

    return cv2.countNonZero(region)


def calcular_momento(imagen_binaria, x, y, r):
    """
    Calcula el área usando momentos.
    """

    mascara = np.zeros(imagen_binaria.shape, dtype=np.uint8)

    cv2.circle(mascara, (x, y), r, 255, -1)

    region = cv2.bitwise_and(imagen_binaria, imagen_binaria, mask=mascara)

    momentos = cv2.moments(region)

    return momentos["m00"]


def detectar_respuestas(imagen, circulos):
    """
    Determina cuál círculo fue marcado.
    """

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    _, binaria = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    datos = []

    for (x, y, r) in circulos:

        pixeles = contar_pixeles(binaria, x, y, r)

        momento = calcular_momento(binaria, x, y, r)

        datos.append({
            "x": x,
            "y": y,
            "r": r,
            "pixeles": pixeles,
            "momento": momento
        })

    letras = ["A", "B", "C", "D"]

    respuestas = []

    # Se asume que cada pregunta tiene 4 opciones
    for i in range(0, len(datos), 4):

        grupo = datos[i:i+4]

        if len(grupo) < 4:
            break

        grupo = sorted(grupo, key=lambda c: c["x"])

        indice = max(
            range(4),
            key=lambda j: grupo[j]["pixeles"]
        )

        respuestas.append(letras[indice])

    return respuestas


def cargar_respuestas():
    """
    Lee la plantilla de respuestas.
    """

    with open("respuestas.json", "r") as archivo:

        return json.load(archivo)


def calificar(respuestas_detectadas):
    """
    Compara las respuestas detectadas
    con la plantilla.
    """

    plantilla = cargar_respuestas()

    aciertos = 0

    for i, respuesta in enumerate(respuestas_detectadas):

        correcta = plantilla[str(i + 1)]

        if respuesta == correcta:
            aciertos += 1

    porcentaje = (aciertos / len(plantilla)) * 100

    return aciertos, porcentaje


def mostrar_resultados(respuestas_detectadas):

    aciertos, nota = calificar(respuestas_detectadas)

    print()

    print("===== RESPUESTAS DETECTADAS =====")

    for i, respuesta in enumerate(respuestas_detectadas):

        print(f"Pregunta {i+1}: {respuesta}")

    print()

    print("===== CALIFICACIÓN =====")

    print("Aciertos:", aciertos)

    print("Nota:", round(nota, 2), "%")
