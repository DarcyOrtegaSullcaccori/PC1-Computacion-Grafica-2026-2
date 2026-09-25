import os
import cv2
import pickle
import numpy as np
from collections import Counter

from comun import (ruta_video, recortar, zonas_de_carta, fraccion_cambio, desfase_exposicion,
                   RUTA_CARTAS_PKL, RUTA_MESA_VACIA, ANCHO_CANON, ALTO_CANON)

RUTA_SALIDA_VIDEO = 'video_procesado.mp4'
CARPETA_DETECTADAS = 'cartas_detectadas'
MAX_ANCHO_VENTANA = 1200
MAX_ALTO_VENTANA = 700
MARGEN_BORDE = 3

UMBRAL_FIGURA = 0.36
CORTE_LATERAL_VACIO = 0.30
CENTRO_SOBRE_LATERAL = 0.5
CENTRO_SOLO_MEDIO = 2.0
CENTRO_SIN_MEDIO = 0.5
CORTE_MEDIO_EXTREMOS = 0.5
CORTE_COBERTURA = 0.72
FILAS_MIN_TINTA = 3
FIGURA_MIN_CENTRO = 3.0
FIGURA_MIN_RAZON = 1.38
FIGURA_CENTRO_CON_RAZON = 2.2

UMBRAL_MOVIMIENTO = 8.0
FRACCION_CAMBIO_CARTA = 0.30
FRACCION_CAMBIO_CUADRANTE = 0.35
COBERTURA_MAX_CENTRO_SOLO = 0.85
FRACCION_CAMBIO_VACIO = 0.05
FRAMES_ESTABLE = 12
FRAMES_VACIO = 30
NIVEL_CAMBIO_PIXEL = 25.0
FRACCION_CAMBIO_HUELLA = 0.03
SEGUNDOS_CAMBIO = 1.5
ACUERDO_MINIMO = 0.7
NUM_MAZOS = 1

PATRONES = {
    (2, 0): '4', (2, 1): '5',
    (3, 0): '6', (3, 1): '7', (3, 2): '8',
    (4, 1): '9', (4, 2): '10',
}
CENTRO_VALIDO = {2: (0, 1), 3: (0, 1, 2), 4: (1, 2)}
ZONAS_CANON = zonas_de_carta((0, 0, ANCHO_CANON, ALTO_CANON))


def densidad(img_bin):
    return cv2.countNonZero(img_bin) / img_bin.size if img_bin.size else 0.0


def aplicar_pipeline(imgBN, carta, kernel):
    recorte = cv2.resize(recortar(imgBN, carta), (ANCHO_CANON, ALTO_CANON), interpolation=cv2.INTER_AREA)
    recorte = cv2.normalize(recorte, None, 0, 255, cv2.NORM_MINMAX)
    imgTH = cv2.adaptiveThreshold(recorte, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    imgDil = cv2.dilate(imgMedian, kernel)
    return imgTH, imgMedian, imgDil


def ajustar_a_carta(imgBN, carta):
    x, y, w, h = carta
    recorte = recortar(imgBN, carta)
    if recorte.size == 0:
        return carta
    _, binaria = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return carta
    cx, cy, cw, ch = cv2.boundingRect(max(contornos, key=cv2.contourArea))
    if cw * ch < 0.4 * w * h or cw < 20 or ch < 20:
        return carta
    return (x + cx, y + cy, cw, ch)


def carta_cortada(carta, ancho, alto):
    x, y, w, h = carta
    return x <= MARGEN_BORDE or y <= MARGEN_BORDE or x + w >= ancho - MARGEN_BORDE or y + h >= alto - MARGEN_BORDE


def vistas_blanco_negro(imgBN, esc, kernel):
    pequeno = cv2.resize(imgBN, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    imgTH = cv2.adaptiveThreshold(pequeno, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    imgDil = cv2.dilate(imgMedian, kernel)
    return [imgTH, imgMedian, imgDil]


def pegar_carta(vista, carta, binaria, esc):
    x, y, w, h = (int(round(v * esc)) for v in carta)
    if w > 0 and h > 0:
        vista[y:y + h, x:x + w] = cv2.resize(binaria, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(vista, (x, y), (x + w, y + h), 128, 1)


def mostrar(nombre, img):
    h, w = img.shape[:2]
    esc = min(1.0, MAX_ANCHO_VENTANA / w, MAX_ALTO_VENTANA / h)
    cv2.imshow(nombre, img if esc == 1.0 else cv2.resize(img, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA))


def valor_hilo(etiqueta):
    if etiqueta in ('2', '3', '4', '5', '6'):
        return 1
    if etiqueta in ('7', '8', '9'):
        return 0
    return -1


def columna_lateral(imgDil, nombre):
    z = ZONAS_CANON
    arriba = cv2.countNonZero(recortar(imgDil, z[f'{nombre}_arriba']))
    medio = cv2.countNonZero(recortar(imgDil, z[f'{nombre}_medio']))
    abajo = cv2.countNonZero(recortar(imgDil, z[f'{nombre}_abajo']))
    columna = recortar(imgDil, z[nombre])
    cobertura = float(((columna > 0).sum(axis=1) >= FILAS_MIN_TINTA).mean())
    razon = medio / max(1.0, (arriba + abajo) / 2)
    return arriba + medio + abajo, razon, cobertura


def clasificar(imgDil):
    z = ZONAS_CANON
    info = {'densidad': densidad(recortar(imgDil, z['interior']))}

    if info['densidad'] >= UMBRAL_FIGURA:
        info['regla'] = 'densidad total'
        return 'J/Q/K', info

    tinta_izq, razon_izq, cob_izq = columna_lateral(imgDil, 'izquierda')
    tinta_der, razon_der, cob_der = columna_lateral(imgDil, 'derecha')
    razon = (razon_izq + razon_der) / 2
    cobertura_lateral = (cob_izq + cob_der) / 2
    tinta_centro = cv2.countNonZero(recortar(imgDil, z['centro']))

    if cobertura_lateral < CORTE_LATERAL_VACIO and tinta_centro >= CENTRO_SOBRE_LATERAL * (tinta_izq + tinta_der):
        arriba = cv2.countNonZero(recortar(imgDil, z['centro_arriba']))
        medio = cv2.countNonZero(recortar(imgDil, z['centro_medio']))
        abajo = cv2.countNonZero(recortar(imgDil, z['centro_abajo']))
        columna = recortar(imgDil, z['centro'])
        cobertura = float(((columna > 0).sum(axis=1) >= FILAS_MIN_TINTA).mean())
        razon_centro = medio / max(1.0, (arriba + abajo) / 2)
        info.update({'razon_centro': razon_centro, 'cobertura': cobertura,
                     'cobertura_lateral': cobertura_lateral})
        if arriba + medio + abajo == 0 or cobertura > COBERTURA_MAX_CENTRO_SOLO:
            info['regla'] = 'columna central sin simbolos claros'
            return None, info
        info['regla'] = f'columna central, medio/extremos={razon_centro:.2f}'
        if razon_centro >= CENTRO_SOLO_MEDIO:
            return 'A', info
        if razon_centro <= CENTRO_SIN_MEDIO:
            return '2', info
        return '3', info

    if razon < CORTE_MEDIO_EXTREMOS:
        s = 2
    elif cobertura_lateral < CORTE_COBERTURA:
        s = 3
    else:
        s = 4

    c_est = tinta_centro / max(1.0, (tinta_izq + tinta_der) / 2) * s
    info.update({'s': s, 'c': c_est, 'razon': razon, 'cobertura': cobertura_lateral})

    if c_est >= FIGURA_MIN_CENTRO or (razon >= FIGURA_MIN_RAZON and c_est >= FIGURA_CENTRO_CON_RAZON):
        info['regla'] = 'cuadricula (centro y laterales llenos)'
        return 'J/Q/K', info

    c = min(CENTRO_VALIDO[s], key=lambda v: abs(v - c_est))
    info['regla'] = f's={s} c={c}'
    return PATRONES[(s, c)], info


class Ranura:
    def __init__(self, frames_cambio):
        self.frames_necesarios = frames_cambio
        self.frames_cambio = 0
        self.votos_revision = []
        self.contada = False
        self.etiqueta = None
        self.votos = []
        self.ultimo_info = None
        self.frames_vacio = 0
        self.previo = None
        self.pequeno = None
        self.etiqueta_viva = None
        self.huella = None

    def movimiento(self, recorte_gris):
        pequeno = cv2.resize(recorte_gris, (60, 80)).astype(np.float32)
        mov = 255.0 if self.previo is None else float(np.abs(pequeno - self.previo).mean())
        self.previo = pequeno
        self.pequeno = pequeno
        return mov

    def zona_cambiada(self):
        if self.huella is None or self.pequeno is None:
            return False
        diferencia = self.pequeno - self.huella
        diferencia = diferencia - np.median(diferencia)
        return float((np.abs(diferencia) > NIVEL_CAMBIO_PIXEL).mean()) >= FRACCION_CAMBIO_HUELLA

    def revisar(self, etiqueta):
        if not self.zona_cambiada():
            self.frames_cambio = 0
            self.votos_revision = []
            return None
        self.frames_cambio += 1
        self.votos_revision.append(etiqueta)
        self.votos_revision = self.votos_revision[-FRAMES_ESTABLE:]
        if self.frames_cambio < self.frames_necesarios or len(self.votos_revision) < FRAMES_ESTABLE:
            return None
        ganadora, n = Counter(self.votos_revision).most_common(1)[0]
        if ganadora is None or n / FRAMES_ESTABLE < ACUERDO_MINIMO:
            return None
        self.votos_revision = []
        self.frames_cambio = 0
        self.huella = self.pequeno.copy()
        if ganadora == self.etiqueta:
            return None
        self.etiqueta = ganadora
        return ganadora

    def registrar_vacio(self, mesa_visible):
        self.votos = []
        self.votos_revision = []
        self.frames_vacio = self.frames_vacio + 1 if mesa_visible else 0
        if self.contada and self.frames_vacio >= FRAMES_VACIO:
            self.contada = False
            self.etiqueta = None

    def votar(self, etiqueta, info):
        self.frames_vacio = 0
        if self.contada:
            return None
        self.votos.append(etiqueta)
        self.votos = self.votos[-FRAMES_ESTABLE:]
        self.ultimo_info = info
        if len(self.votos) < FRAMES_ESTABLE:
            return None
        ganadora, n = Counter(self.votos).most_common(1)[0]
        if ganadora is None or n / FRAMES_ESTABLE < ACUERDO_MINIMO:
            return None
        self.contada = True
        self.etiqueta = ganadora
        self.huella = None if self.pequeno is None else self.pequeno.copy()
        self.votos_revision = []
        return ganadora


def carta_cubre_zona(recorte, vacia_recorte, desfase):
    h, w = recorte.shape[:2]
    mh, mw = h // 2, w // 2
    cuadrantes = [(slice(0, mh), slice(0, mw)), (slice(0, mh), slice(mw, w)),
                  (slice(mh, h), slice(0, mw)), (slice(mh, h), slice(mw, w))]
    return all(fraccion_cambio(recorte[a, b], vacia_recorte[a, b], desfase) >= FRACCION_CAMBIO_CUADRANTE
               for a, b in cuadrantes)


def etiqueta_con_fondo(img, texto, x, y, color, fuente):
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, fuente, 2)
    ty = max(th + 8, y - 8)
    cv2.rectangle(img, (x, ty - th - 6), (x + tw + 8, ty + base), (0, 0, 0), -1)
    cv2.putText(img, texto, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fuente, color, 2)


def dibujar_carta(img, carta, ranura, presente, escala, cortada=False):
    x, y, w, h = carta
    if cortada:
        color, texto = (0, 220, 255), 'cortada'
    elif ranura.contada:
        v = valor_hilo(ranura.etiqueta)
        color = (0, 200, 0) if v == 1 else (0, 0, 230) if v == -1 else (220, 220, 220)
        texto = f'{ranura.etiqueta} ({v:+d})'
    elif presente:
        color = (0, 220, 255)
        texto = f'leyendo {ranura.etiqueta_viva}...' if ranura.etiqueta_viva else 'leyendo...'
    else:
        color, texto = (120, 120, 120), ''

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    z = zonas_de_carta(carta)
    for k in ('izquierda', 'centro_arriba', 'centro_medio', 'centro_abajo', 'derecha'):
        zx, zy, zw, zh = z[k]
        cv2.rectangle(img, (zx, zy), (zx + zw, zy + zh), color, 1)
    if texto:
        etiqueta_con_fondo(img, texto, x, y, color, 0.8 * escala)


def guardar_detectada(img, carta, numero, indice, etiqueta, binaria):
    os.makedirs(CARPETA_DETECTADAS, exist_ok=True)
    recorte = cv2.resize(recortar(img, carta), (ANCHO_CANON, ALTO_CANON))
    cv2.rectangle(recorte, (0, 0), (ANCHO_CANON, 36), (0, 0, 0), -1)
    cv2.putText(recorte, f'{etiqueta} ({valor_hilo(etiqueta):+d})', (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    nombre = etiqueta.replace('/', '')
    base = os.path.join(CARPETA_DETECTADAS, f'{numero:03d}_pos{indice + 1}_{nombre}')
    cv2.imwrite(f'{base}.png', recorte)
    cv2.imwrite(f'{base}_binaria.png', binaria)


def detalle_lectura(binaria):
    z = ZONAS_CANON
    tinta_izq, razon_izq, cob_izq = columna_lateral(binaria, 'izquierda')
    tinta_der, razon_der, cob_der = columna_lateral(binaria, 'derecha')
    tinta_centro = cv2.countNonZero(recortar(binaria, z['centro']))
    arriba = cv2.countNonZero(recortar(binaria, z['centro_arriba']))
    medio = cv2.countNonZero(recortar(binaria, z['centro_medio']))
    abajo = cv2.countNonZero(recortar(binaria, z['centro_abajo']))
    return (f'cob_lat={(cob_izq + cob_der) / 2:.2f} cen/lat={tinta_centro / max(1, tinta_izq + tinta_der):.2f} '
            f'lat={tinta_izq + tinta_der} cen={tinta_centro} ({arriba}/{medio}/{abajo})')


def main():
    with open(RUTA_CARTAS_PKL, 'rb') as f:
        cartas = pickle.load(f)
    if not cartas:
        raise RuntimeError(f'{RUTA_CARTAS_PKL} esta vacio. Corran obtener_cartas.py primero.')

    vacia = cv2.imread(RUTA_MESA_VACIA, cv2.IMREAD_GRAYSCALE)
    if vacia is None:
        raise RuntimeError(f'No existe {RUTA_MESA_VACIA}. Corran obtener_cartas.py primero.')

    ruta = ruta_video()
    video = cv2.VideoCapture(ruta)
    if not video.isOpened():
        raise RuntimeError(f'No se pudo abrir {ruta}.')

    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escala = max(0.6, w / 1600)
    if vacia.shape != (h, w):
        raise RuntimeError(f'{RUTA_MESA_VACIA} no tiene el tamano del video. Vuelvan a correr obtener_cartas.py.')
    print(f'Procesando {ruta} ({w}x{h}, {fps:.1f} fps) con {len(cartas)} posiciones de carta')
    print('Ajuste automatico de la zona al borde de la carta: activado')

    writer = cv2.VideoWriter(RUTA_SALIDA_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    ranuras = [Ranura(int(fps * SEGUNDOS_CAMBIO)) for _ in cartas]
    kernel = np.ones((5, 5), np.int8)
    esc_vista = min(1.0, MAX_ANCHO_VENTANA / w, MAX_ALTO_VENTANA / h)
    cortadas = [carta_cortada(c, w, h) for c in cartas]
    for i, cortada in enumerate(cortadas, start=1):
        if cortada:
            print(f'Aviso: la posicion {i} toca el borde del video; esa carta queda cortada y no se cuenta')

    conteo = 0
    cartas_vistas = 0
    frame_idx = 0

    while True:
        check, img = video.read()
        if not check:
            break

        imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        desfase = desfase_exposicion(imgBN, vacia)
        imgTH, imgMedian, imgDil = vistas_blanco_negro(imgBN, esc_vista, kernel)

        for i, (carta, ranura) in enumerate(zip(cartas, ranuras)):
            zona_carta = ajustar_a_carta(imgBN, carta)
            etapas = aplicar_pipeline(imgBN, zona_carta, kernel)
            for vista, binaria in zip((imgTH, imgMedian, imgDil), etapas):
                pegar_carta(vista, zona_carta, binaria, esc_vista)

            if cortadas[i]:
                dibujar_carta(img, carta, ranura, False, escala, cortada=True)
                continue

            recorte = recortar(imgBN, carta)
            recorte_vacio = recortar(vacia, carta)
            cambio = fraccion_cambio(recorte, recorte_vacio, desfase)
            presente = cambio >= FRACCION_CAMBIO_CARTA and carta_cubre_zona(recorte, recorte_vacio, desfase)
            quieto = ranura.movimiento(recorte) < UMBRAL_MOVIMIENTO
            nueva = None

            if presente and quieto:
                etiqueta, info = clasificar(etapas[2])
                ranura.etiqueta_viva = etiqueta
                if not ranura.contada:
                    nueva = ranura.votar(etiqueta, info)
                else:
                    ranura.ultimo_info = info
                    nueva = ranura.revisar(etiqueta)
                    if nueva is not None:
                        print(f'[{frame_idx / fps:7.1f}s] posicion {i + 1}: cambio la carta')
            elif not presente:
                ranura.registrar_vacio(quieto and cambio <= FRACCION_CAMBIO_VACIO)

            if nueva is not None:
                v = valor_hilo(nueva)
                conteo += v
                cartas_vistas += 1
                guardar_detectada(img, zona_carta, cartas_vistas, i, nueva, etapas[2])
                inf = ranura.ultimo_info
                print(f'[{frame_idx / fps:7.1f}s] posicion {i + 1}: {nueva:>5} -> {v:+d} | conteo={conteo:+d} '
                      f"| densidad={inf['densidad']:.3f} | {inf['regla']}")
                print(f'            zona={zona_carta} {detalle_lectura(etapas[2])}')

            dibujar_carta(img, zona_carta, ranura, presente, escala)

        mazos_restantes = max(0.5, (52 * NUM_MAZOS - cartas_vistas) / 52)
        cv2.rectangle(img, (0, 0), (int(520 * escala), int(125 * escala)), (0, 0, 0), -1)
        cv2.putText(img, f'Conteo Hi-Lo: {conteo:+d}', (int(12 * escala), int(45 * escala)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3 * escala, (255, 255, 255), 3)
        cv2.putText(img, f'Cartas vistas: {cartas_vistas}   Conteo verdadero: {conteo / mazos_restantes:+.2f}',
                    (int(12 * escala), int(95 * escala)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * escala, (200, 200, 200), 2)

        writer.write(img)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f'  ... {frame_idx} frames procesados')

        #mostrar('video', img)
        mostrar('video TH', imgTH)
        # mostrar('video Median', imgMedian)
        # mostrar('video Dilatada', imgDil)
        if cv2.waitKey(1) & 0xFF == 27:
            print('Detenido con ESC')
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    print('----------------------------------------')
    print(f'Video procesado guardado en: {RUTA_SALIDA_VIDEO}')
    print(f'Recortes de cada carta contada en: {CARPETA_DETECTADAS}/')
    print(f'Cartas contadas: {cartas_vistas}')
    print(f'Conteo Hi-Lo final: {conteo:+d}')


if __name__ == '__main__':
    main()