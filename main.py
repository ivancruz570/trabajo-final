import cv2
from omr import detectar_circulos
from utils import detectar_respuestas

# Cargar imagen

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
