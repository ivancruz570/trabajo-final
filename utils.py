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
