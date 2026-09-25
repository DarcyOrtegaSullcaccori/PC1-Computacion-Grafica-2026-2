import os
import cv2
import numpy as np

VIDEO_ORIGINAL = 'video_n1.MOV'
VIDEO_ESTABILIZADO = 'video_estabilizado.mp4'
USAR_ESTABILIZADO = True
RUTA_CARTAS_PKL = 'cartas.pkl'
RUTA_MESA_VACIA = 'mesa_vacia.png'

MARGEN_X = 0.17
MARGEN_Y = 0.07
CORTE_COLUMNAS = (0.33, 0.67)
CORTE_FILAS = (0.33, 0.67)

ANCHO_CANON = 300
ALTO_CANON = 400

UMBRAL_DIFERENCIA = 30


def ruta_video():
    if USAR_ESTABILIZADO and os.path.exists(VIDEO_ESTABILIZADO):
        return VIDEO_ESTABILIZADO
    return VIDEO_ORIGINAL


def recortar(img, zona):
    x, y, w, h = zona
    return img[y:y + h, x:x + w]


def zonas_de_carta(carta):
    x, y, w, h = carta
    mx, my = int(w * MARGEN_X), int(h * MARGEN_Y)
    ix, iy, iw, ih = x + mx, y + my, w - 2 * mx, h - 2 * my
    c1, c2 = ix + int(iw * CORTE_COLUMNAS[0]), ix + int(iw * CORTE_COLUMNAS[1])
    f1, f2 = iy + int(ih * CORTE_FILAS[0]), iy + int(ih * CORTE_FILAS[1])
    fin_x, fin_y = ix + iw, iy + ih
    zonas = {'interior': (ix, iy, iw, ih)}
    for nombre, (x0, x1) in (('izquierda', (ix, c1)), ('centro', (c1, c2)), ('derecha', (c2, fin_x))):
        zonas[nombre] = (x0, iy, x1 - x0, ih)
        for fila, (y0, y1) in (('arriba', (iy, f1)), ('medio', (f1, f2)), ('abajo', (f2, fin_y))):
            zonas[f'{nombre}_{fila}'] = (x0, y0, x1 - x0, y1 - y0)
    return zonas


def desfase_exposicion(gris, gris_vacia):
    a = cv2.resize(gris, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
    b = cv2.resize(gris_vacia, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
    return float(np.median(a - b))


def fraccion_cambio(gris_recorte, vacia_recorte, desfase=0.0):
    referencia = np.clip(vacia_recorte.astype(np.int16) + int(round(desfase)), 0, 255).astype(np.uint8)
    diferencia = cv2.absdiff(gris_recorte, referencia)
    _, binaria = cv2.threshold(diferencia, UMBRAL_DIFERENCIA, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(binaria) / binaria.size if binaria.size else 0.0


def dibujar_cuadricula(img, carta, color, grosor=2):
    x, y, w, h = carta
    cv2.rectangle(img, (x, y), (x + w, y + h), color, grosor)
    z = zonas_de_carta(carta)
    for nombre in ('izquierda', 'centro', 'derecha'):
        for fila in ('arriba', 'medio', 'abajo'):
            zx, zy, zw, zh = z[f'{nombre}_{fila}']
            cv2.rectangle(img, (zx, zy), (zx + zw, zy + zh), color, 1)
