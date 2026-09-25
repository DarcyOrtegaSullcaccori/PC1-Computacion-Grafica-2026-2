import cv2
import pickle
import numpy as np

from comun import (ruta_video, RUTA_CARTAS_PKL, RUTA_MESA_VACIA, UMBRAL_DIFERENCIA,
                   dibujar_cuadricula, desfase_exposicion)

MAX_ANCHO_VENTANA = 1200
MAX_ALTO_VENTANA = 700
PASO_MUESTREO = 10
VENTANA_FRAME = 'Elegir frame'
VENTANA_MARCAR = 'Marcar cartas'


def contar_frames(ruta):
    video = cv2.VideoCapture(ruta)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while video.grab():
            total += 1
    video.release()
    return total


def sugerir_frame_con_cartas(ruta, vacia):
    gris_vacia = cv2.resize(cv2.cvtColor(vacia, cv2.COLOR_BGR2GRAY), (480, int(480 * vacia.shape[0] / vacia.shape[1])))
    video = cv2.VideoCapture(ruta)
    mejor, mejor_idx, idx = -1, 0, 0
    while video.grab():
        if idx % PASO_MUESTREO == 0:
            check, frame = video.retrieve()
            if check:
                gris = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gris_vacia.shape[::-1])
                desfase = desfase_exposicion(gris, gris_vacia)
                referencia = np.clip(gris_vacia.astype(np.int16) + int(round(desfase)), 0, 255).astype(np.uint8)
                _, cambio = cv2.threshold(cv2.absdiff(gris, referencia), UMBRAL_DIFERENCIA, 255, cv2.THRESH_BINARY)
                n = cv2.countNonZero(cambio)
                if n > mejor:
                    mejor, mejor_idx = n, idx
        idx += 1
    video.release()
    return mejor_idx


def leer_frame(video, idx):
    video.set(cv2.CAP_PROP_POS_FRAMES, idx)
    check, frame = video.read()
    return frame if check else None


def escala_para_pantalla(frame):
    h, w = frame.shape[:2]
    return min(1.0, MAX_ANCHO_VENTANA / w, MAX_ALTO_VENTANA / h)


def con_texto(img, lineas):
    salida = img.copy()
    for n, linea in enumerate(lineas):
        y = 28 + 28 * n
        cv2.putText(salida, linea, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(salida, linea, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return salida


def elegir_frame(ruta, sugerido, total, titulo):
    video = cv2.VideoCapture(ruta)
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    salto = int(round(fps))
    print(titulo)
    cv2.namedWindow(VENTANA_FRAME, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('frame', VENTANA_FRAME, sugerido, max(1, total - 1), lambda v: None)

    frame, actual = None, -1
    while True:
        pos = cv2.getTrackbarPos('frame', VENTANA_FRAME)
        if pos != actual:
            nuevo = leer_frame(video, pos)
            if nuevo is not None:
                frame, actual = nuevo, pos
                esc = escala_para_pantalla(frame)
                vista = cv2.resize(frame, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
                cv2.imshow(VENTANA_FRAME, con_texto(vista, [
                    titulo,
                    f'Frame {actual} ({actual / fps:.1f}s)',
                    'A/D: -/+1 frame   J/L: -/+1 seg   ENTER: usar este frame   ESC: salir']))
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (13, 10, 32) and frame is not None:
            break
        if tecla == 27:
            video.release()
            cv2.destroyAllWindows()
            raise SystemExit('Cancelado.')
        pasos = {ord('a'): -1, ord('d'): 1, ord('j'): -salto, ord('l'): salto}
        if tecla in pasos:
            cv2.setTrackbarPos('frame', VENTANA_FRAME, min(max(0, pos + pasos[tecla]), total - 1))

    video.release()
    cv2.destroyWindow(VENTANA_FRAME)
    return frame, actual


def marcar_cartas(frame):
    esc = escala_para_pantalla(frame)
    vista = cv2.resize(frame, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    cartas = []
    while True:
        guia = con_texto(vista, [f'Carta {len(cartas) + 1}: arrastra el contorno y pulsa ENTER',
                                 'Para terminar: ENTER sin arrastrar'])
        x, y, w, h = cv2.selectROI(VENTANA_MARCAR, guia, showCrosshair=False)
        if w == 0 or h == 0:
            break
        cartas.append((int(round(x / esc)), int(round(y / esc)), int(round(w / esc)), int(round(h / esc))))
        dibujar_cuadricula(vista, (x, y, w, h), (0, 200, 255), 2)
    cv2.destroyAllWindows()
    return cartas


def main():
    ruta = ruta_video()
    print(f'Video: {ruta}')
    total = contar_frames(ruta)
    if total == 0:
        raise RuntimeError(f'No se pudo leer {ruta}.')

    vacia, idx_vacia = elegir_frame(ruta, 0, total, 'Paso 1 de 2: elige un frame con la mesa VACIA')
    cv2.imwrite(RUTA_MESA_VACIA, vacia)

    print('Buscando el frame con mas cartas sobre la mesa...')
    sugerido = sugerir_frame_con_cartas(ruta, vacia)
    frame, idx = elegir_frame(ruta, sugerido, total, 'Paso 2 de 2: elige un frame con TODAS las cartas')
    cv2.imwrite('frame_referencia.png', frame)

    cartas = marcar_cartas(frame)
    with open(RUTA_CARTAS_PKL, 'wb') as f:
        pickle.dump(cartas, f)

    preview = frame.copy()
    for n, carta in enumerate(cartas, start=1):
        dibujar_cuadricula(preview, carta, (0, 200, 255), 2)
        cv2.putText(preview, str(n), (carta[0] + 5, carta[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite('zonas_preview.png', preview)

    print(f'Mesa vacia: frame {idx_vacia} (guardada en {RUTA_MESA_VACIA})')
    print(f'Cartas: frame {idx}. {len(cartas)} posiciones guardadas en {RUTA_CARTAS_PKL}')
    for n, c in enumerate(cartas, start=1):
        print(f'  carta {n} = {c}')
    print('Revisen zonas_preview.png: la cuadricula debe caer sobre los simbolos de cada carta.')


if __name__ == '__main__':
    main()
