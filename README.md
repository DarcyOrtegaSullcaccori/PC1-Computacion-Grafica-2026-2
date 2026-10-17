# Sistema de Detección, Clasificación y Conteo de Cartas en Blackjack (Hi-Lo)
### Computación Gráfica (2026-2) — Práctica Calificada 1 (PC1)
**Universidad Nacional de Ingeniería (UNI)**

---

## 📸 Captura de la Ejecución en Tiempo Real

A continuación se muestra el sistema en funcionamiento, procesando el flujo de video, reconociendo las cartas en la mesa de juego y calculando el puntaje Hi-Lo y True Count de Blackjack:

<p align="center">
  <img src="WhatsApp%20Image%202026-09-22%20at%2012.33.57.jpeg" alt="Ejecución del sistema de conteo de Blackjack" width="90%"/>
</p>

*Figura: Ejecución en tiempo real mostrando el HUD superior (Conteo Hi-Lo actual: `+7`, Cartas vistas: `10`, Conteo verdadero: `+8.67`), la detección de contornos, la subdivisión en cuadrícula de análisis y la clasificación de cada carta (`4 (+1)`, `8 (+0)`, `5 (+1)`, `2 (+1)`, `9 (+0)`).*

---

## 📖 Resumen del Proyecto

Este proyecto implementa un sistema automatizado de **visión por computadora clásica** para detectar, clasificar y contar cartas de baraja en un tapete de **Blackjack**, aplicando la estrategia de conteo **Hi-Lo** en secuencias de video.

A diferencia de enfoques basados en redes neuronales profundas (deep learning), esta solución está construida completamente sobre técnicas de procesamiento digital de imágenes con **OpenCV** y **NumPy**, logrando alta eficiencia computacional, interpretabilidad geométrica y robustez ante variaciones de iluminación, vibración de cámara y oclusiones por el movimiento de las manos del repartidor o jugadores.

### Principales Capacidades
- **Estabilización de Video (Affine Motion Correction)**: Algoritmo bidireccional de flujo óptico (Lucas-Kanade) combinado con RANSAC para compensar vibraciones o desplazamientos involuntarios de la cámara sin recortar ni distorsionar artificialmente la escena.
- **Compensación Dinámica de Iluminación**: Corrección de desfase de exposición en escala de grises frente a un frame de referencia de mesa vacía.
- **Ajuste Automático de Región de Interés (ROI)**: Segmentación con umbralizado Otsu y operaciones morfológicas de apertura para ceñir el marco exactamente a la carta, tolerando leves desalineaciones en la colocación.
- **Clasificación Estructural por Densidad y Simetría**: Análisis de distribución de tinta en subzonas normalizadas (3 columnas $\times$ 3 filas) para identificar cartas numéricas (2 al 10), Ases y figuras (J, Q, K) según la presencia y simetría de sus símbolos.
- **Filtro Temporal y Consenso Multi-Frame**: Máquina de estados con búfer de votación ponderada para evitar falsos positivos durante la colocación, retiro o movimiento de cartas.
- **Estrategia Blackjack Hi-Lo**: Cálculo del *Running Count* y del *True Count* normalizado por la estimación de mazos restantes, con retroalimentación visual codificada por colores en pantalla.

---

## 🔬 Arquitectura y Metodología

El flujo de procesamiento del sistema se divide en tres fases principales:

```
[ Video Original ]
        │
        ▼
1. estabilizacion.py ──► Flujo Óptico Lucas-Kanade + RANSAC ──► [ Video Estabilizado ]
        │
        ▼
2. obtener_cartas.py ──► Calibración interactiva y ROI ──► [ cartas.pkl + mesa_vacia.png ]
        │
        ▼
3. main_blackjack.py ──► Preprocesamiento + Clasificación + Hi-Lo ──► [ video_procesado.mp4 ]
                                                                      [ cartas_detectadas/ ]
```

### 1. Estabilización de Video (`estabilizacion.py` / `estabilizar.py`)
- **Detección de Características**: Localización de puntos de esquina estables en el fondo mediante el algoritmo de Shi-Tomasi (`cv2.goodFeaturesToTrack`).
- **Seguimiento Bidireccional (Forward-Backward)**: Seguimiento temporal punto a punto con pirámide Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`), descartando puntos con error de ciclo ida-vuelta superior a 1.0 px.
- **Estimación de Matriz Afín Parcial con RANSAC**: Estimación robusta de traslación ($t_x, t_y$) y rotación ($\theta$), eliminando puntos espurios causados por objetos en movimiento (manos, fichas, cartas).
- **Suavizado Temporal**: Filtro de mediana móvil para eliminar jitter y reconstrucción de cada cuadro mediante transformación afín (`cv2.warpAffine`).

### 2. Calibración y Definición de Zonas (`obtener_cartas.py`)
- **Captura de Mesa Vacía**: Selección interactiva del fotograma donde el tapete no contiene cartas ni manos (`mesa_vacia.png`), utilizado como línea base de sustracción de fondo.
- **Sugerencia Automática de Frame con Cartas**: Análisis diferencial acumulado en el tiempo para proponer de forma autónoma el frame con mayor presencia de cartas.
- **Definición de Ranuras (ROIs)**: Mediante `cv2.selectROI`, el usuario marca las posiciones esperadas de las cartas en el tapete, guardándose en un archivo serializado `cartas.pkl`.
- **Previsualización de Cuadrícula**: Genera `zonas_preview.png` con la malla canónica $3 \times 3$ proyectada sobre cada carta.

### 3. Preprocesamiento y Segmentación de Cartas
Para cada cuadro y ranura activa:
1. **Compensación de Brillo**: Se calcula la mediana de la diferencia entre la imagen actual y la mesa vacía (`desfase_exposicion`), aplicando un offset adaptativo.
2. **Refinamiento de Borde**: Mediante binarización Otsu y apertura morfológica (`cv2.morphologyEx`), se detecta el rectángulo envolvente real de la carta (`ajustar_a_carta`).
3. **Normalización Canónica**: Redimensionamiento a $300 \times 400$ px y normalización de contraste (`cv2.normalize`).
4. **Pipeline Binarizado**: Umbralizado adaptativo gaussiano (`cv2.adaptiveThreshold`), filtro de mediana (`cv2.medianBlur`) y dilatación morfológica (`cv2.dilate`) para consolidar trazos y figuras.

### 4. Algoritmo de Clasificación por Cuadrícula
La carta normalizada se descompone en 9 celdas canónicas (`izquierda`, `centro`, `derecha` $\times$ `arriba`, `medio`, `abajo`):
- **Figuras (`J / Q / K`)**: Si la densidad de tinta en el área interior supera el umbral crítico ($\ge 0.36$) o la cobertura lateral y central es masiva, se cataloga directamente como figura con valor Hi-Lo $-1$.
- **Columna Central Dominante (`A`, `2`, `3`)**: Si las columnas laterales están prácticamente vacías y la tinta se concentra en el centro, se analiza la proporción entre el tercio medio y los extremos:
  - Ratio medio/extremos alto $\implies$ **As (`A`)**.
  - Ratio medio/extremos bajo $\implies$ **`2`**.
  - Distribución uniforme $\implies$ **`3`**.
- **Cartas Numéricas (`4` al `10`)**: Se evalúa la cantidad de símbolos laterales ($s \in \{2, 3, 4\}$) y la cantidad de símbolos en la columna central ($c \in \{0, 1, 2\}$) mediante un mapeo determinista:
  - $(s=2, c=0) \implies \mathbf{4}$
  - $(s=2, c=1) \implies \mathbf{5}$
  - $(s=3, c=0) \implies \mathbf{6}$
  - $(s=3, c=1) \implies \mathbf{7}$
  - $(s=3, c=2) \implies \mathbf{8}$
  - $(s=4, c=1) \implies \mathbf{9}$
  - $(s=4, c=2) \implies \mathbf{10}$

### 5. Máquina de Estados y Filtro Anti-Rebote (`Ranura`)
Cada posición de carta mantiene un objeto `Ranura` que:
- Detecta si la carta está quieta mediante diferencia de fotogramas sucesivos (`UMBRAL_MOVIMIENTO`).
- Exige verificación de cobertura en los 4 cuadrantes (`carta_cubre_zona`).
- Requiere un acuerdo mínimo del $70\%$ (`ACUERDO_MINIMO`) a lo largo de 12 frames consecutivos (`FRAMES_ESTABLE`) antes de consolidar el conteo.
- Detecta cuando una carta es retirada tras 30 frames continuos de mesa vacía (`FRAMES_VACIO`) para liberar la ranura.

---

## ♠️ Sistema de Conteo Hi-Lo

El sistema implementa el método clásico de conteo **Hi-Lo** (High-Low) de Blackjack:

| Categoría | Cartas | Valor Hi-Lo | Justificación Estratégica |
|:---:|:---:|:---:|:---|
| **Bajas** | `2`, `3`, `4`, `5`, `6` | **$+1$** | Favorecen al crupier; cuando salen del mazo, benefician al jugador. |
| **Neutras** | `7`, `8`, `9` | **$0$** | Poco impacto en la probabilidad relativa del mazo. |
| **Altas** | `10`, `J`, `Q`, `K`, `A` | **$-1$** | Favorecen al jugador (Blackjack y manos altas); al salir, reducen la ventaja. |

### Conteo Corriente y Conteo Verdadero
- **Running Count ($RC$)**: Suma algebraica acumulada de los valores Hi-Lo de todas las cartas reveladas:
  $$RC = \sum_{k=1}^{N} \text{valor}(carta_k)$$
- **True Count ($TC$)**: Conteo normalizado respecto a los mazos estimados por jugar:
  $$TC = \frac{RC}{\text{Mazos restantes}}$$
  Donde $\text{Mazos restantes} = \max\left(0.5, \frac{52 \times M - \text{cartas vistas}}{52}\right)$.

### Código de Colores en Pantalla
- 🟩 **Verde**: Cartas con valor $+1$ (ej. `4 (+1)`, `5 (+1)`, `2 (+1)`).
- ⬜ **Blanco**: Cartas neutras con valor $0$ (ej. `8 (+0)`, `9 (+0)`).
- 🟥 **Rojo**: Cartas con valor $-1$ (ej. `10 (-1)`, `J/Q/K (-1)`).
- 🟨 **Cyan/Amarillo**: Cartas en proceso de lectura y estabilización (`leyendo...`).

---

## 📁 Estructura del Repositorio

```text
├── comun.py                               # Constantes, funciones de geometría y análisis de color/exposición
├── estabilizacion.py                      # Motor de estabilización afín con LK y RANSAC
├── estabilizar.py                         # Script para ejecutar la estabilización del video de entrada
├── obtener_cartas.py                      # Calibración interactiva: mesa vacía y selección de ROIs
├── main_blackjack.py                      # Procesamiento principal, HUD, clasificación y conteo Hi-Lo
├── requirements.txt                       # Dependencias del proyecto (numpy, opencv-python)
├── WhatsApp Image 2026-09-22 at 12.33.57.jpeg  # Captura de demostración de la ejecución
└── README.md                              # Documentación completa del proyecto
```

---

## 🚀 Requisitos e Instalación

### Requisitos Previos
- Python 3.9 o superior.
- Gestor de paquetes `pip`.

### Instalación de Dependencias
```bash
# Clonar el repositorio
git clone https://github.com/DarcyOrtegaSullcaccori/PC1-Computacion-Grafica-2026-2.git
cd PC1-Computacion-Grafica-2026-2

# Instalar dependencias
pip install -r requirements.txt
```

*Contenido de `requirements.txt`:*
- `numpy==1.26.4`
- `opencv-python==4.9.0.80`

---

## 🎮 Guía de Ejecución

El flujo completo se ejecuta en tres pasos secuenciales:

### Paso 1: Estabilización del Video (Opcional si ya se dispone de video estable)
Asegurarse de tener el video de entrada `video_n1.MOV` en el directorio raíz y ejecutar:
```bash
python estabilizar.py
```
*Salida: genera `video_estabilizado.mp4` compensando movimientos de cámara.*

### Paso 2: Calibración de Mesa Vacía y Marcado de Cartas
Ejecutar el asistente interactivo:
```bash
python obtener_cartas.py
```
1. **Frame de Mesa Vacía**: Usar la barra deslizadora o teclas (`A`/`D` para avanzar/retroceder 1 frame, `J`/`L` para 1 segundo) hasta encontrar un cuadro con la mesa limpia. Pulsar `ENTER`.
2. **Frame con Cartas**: El sistema sugerirá el cuadro con mayor actividad. Confirmar o ajustar y pulsar `ENTER`.
3. **Selección de ROIs**: Con el mouse, dibujar un rectángulo sobre cada carta y presionar `ENTER`. Al terminar de marcar todas las posiciones, presionar `ENTER` con un cuadro vacío.
*Salida: se generan `mesa_vacia.png`, `cartas.pkl` y la imagen de control `zonas_preview.png`.*

### Paso 3: Ejecución del Conteo de Blackjack en Tiempo Real
Iniciar el procesamiento y la visualización:
```bash
python main_blackjack.py
```
- Se abrirán las ventanas con el video anotado y las etapas binarizadas (`video TH`).
- En la consola se imprimirán en tiempo real los eventos de detección, densidad de tinta, regla activada y conteo acumulado.
- Para detener la ejecución en cualquier momento, presionar la tecla `ESC`.

### Resultados Generados
- `video_procesado.mp4`: Video completo con el HUD superior, cuadros delimitadores y etiquetas Hi-Lo sobre cada carta.
- `cartas_detectadas/`: Directorio con recortes individuales (`.png`) y máscaras binarizadas de cada carta clasificada en orden cronológico.

---

## 👥 Autores y Colaboradores

- **Darcy Ortega Sullcaccori** — [`darcy.ortega.s@uni.pe`](mailto:darcy.ortega.s@uni.pe) — *Facultad de Ciencias, Universidad Nacional de Ingeniería (UNI)*

---

*Proyecto desarrollado para el curso de **Computación Gráfica (2026-2)**.*
