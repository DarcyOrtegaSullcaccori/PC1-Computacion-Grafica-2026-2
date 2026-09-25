import cv2
import numpy as np

ANCHO_PROCESO = 960
MAX_ESQUINAS = 500
CALIDAD_MINIMA = 0.005
DISTANCIA_MINIMA = 10
MIN_PUNTOS = 25
ERROR_IDA_VUELTA = 1.0
UMBRAL_RANSAC = 2.0
VENTANA_MEDIANA = 1
AVISO_MOVIMIENTO = 0.03
FRAME_REFERENCIA = 0

PARAMETROS_LK = dict(winSize=(21, 21), maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _reducir(frame, escala):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)


def _puntos_de_fondo(gris_reducido):
    return cv2.goodFeaturesToTrack(gris_reducido, maxCorners=MAX_ESQUINAS, qualityLevel=CALIDAD_MINIMA,
                                   minDistance=DISTANCIA_MINIMA, blockSize=7)


def _estimar(ref_gris, ref_pts, gris):
    actuales, st1, _ = cv2.calcOpticalFlowPyrLK(ref_gris, gris, ref_pts, None, **PARAMETROS_LK)
    if actuales is None:
        return None
    vuelta, st2, _ = cv2.calcOpticalFlowPyrLK(gris, ref_gris, actuales, None, **PARAMETROS_LK)
    error = np.linalg.norm((ref_pts - vuelta).reshape(-1, 2), axis=1)
    buenos = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (error < ERROR_IDA_VUELTA)
    if buenos.sum() < MIN_PUNTOS:
        return None
    m, inliers = cv2.estimateAffinePartial2D(actuales[buenos], ref_pts[buenos], method=cv2.RANSAC,
                                             ransacReprojThreshold=UMBRAL_RANSAC)
    if m is None or inliers is None or inliers.sum() < MIN_PUNTOS:
        return None
    angulo = np.arctan2(m[1, 0], m[0, 0])
    return np.array([m[0, 2], m[1, 2], angulo])


def _matriz(parametros, escala):
    tx, ty, angulo = parametros
    return np.array([[np.cos(angulo), -np.sin(angulo), tx / escala],
                     [np.sin(angulo), np.cos(angulo), ty / escala]], dtype=np.float32)


def _mediana_temporal(parametros, ventana):
    r = ventana // 2
    relleno = np.pad(parametros, ((r, r), (0, 0)), mode='edge')
    return np.array([np.median(relleno[i:i + ventana], axis=0) for i in range(len(parametros))])


def estabilizar_video(ruta_entrada, ruta_salida, mostrar_progreso=True):
    video = cv2.VideoCapture(ruta_entrada)
    if not video.isOpened():
        raise RuntimeError(f'No se pudo abrir {ruta_entrada}.')
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escala = min(1.0, ANCHO_PROCESO / w)

    video.set(cv2.CAP_PROP_POS_FRAMES, FRAME_REFERENCIA)
    check, referencia = video.read()
    if not check:
        raise RuntimeError('No se pudo leer el frame de referencia.')
    ref_gris = _reducir(referencia, escala)
    ref_pts = _puntos_de_fondo(ref_gris)
    if ref_pts is None:
        raise RuntimeError('No se encontraron puntos de referencia.')
    print(f'Referencia: frame {FRAME_REFERENCIA}, {len(ref_pts)} puntos')

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    parametros, fallidos = [], 0
    ultimo = np.array([0.0, 0.0, 0.0])
    while True:
        check, frame = video.read()
        if not check:
            break
        estimado = _estimar(ref_gris, ref_pts, _reducir(frame, escala))
        if estimado is None:
            fallidos += 1
            estimado = ultimo
        parametros.append(estimado)
        ultimo = estimado
        if mostrar_progreso and len(parametros) % 300 == 0:
            print(f'  midiendo movimiento... frame {len(parametros)}')
    video.release()

    parametros = _mediana_temporal(np.array(parametros), VENTANA_MEDIANA)
    desplazamiento = np.hypot(parametros[:, 0], parametros[:, 1]) / escala

    print(f'Movimiento de camara corregido: promedio {desplazamiento.mean():.1f} px, maximo {desplazamiento.max():.1f} px')
    print(f'Frames sin estimacion confiable (se reuso la anterior): {fallidos}')
    if desplazamiento.max() <= 1.5:
        print('La camara ya estaba practicamente fija: el video estabilizado queda igual al original.')
    if desplazamiento.max() > AVISO_MOVIMIENTO * w:
        print('Aviso: la camara se movio bastante; las cartas pegadas al borde pueden quedar parcialmente fuera de cuadro.')

    video = cv2.VideoCapture(ruta_entrada)
    writer = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    i = 0
    while True:
        check, frame = video.read()
        if not check or i >= len(parametros):
            break
        salida = cv2.warpAffine(frame, _matriz(parametros[i], escala), (w, h), borderMode=cv2.BORDER_REPLICATE)
        writer.write(salida)
        i += 1
        if mostrar_progreso and i % 300 == 0:
            print(f'  escribiendo... frame {i}')
    video.release()
    writer.release()
    print(f'Video estabilizado guardado en: {ruta_salida} ({i} frames, sin zoom ni recorte)')
    return ruta_salida
