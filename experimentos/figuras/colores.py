"""Color, marcador y etiqueta de cada sistema en las figuras del TFG (T2.8).

Colores de las figuras, paleta Okabe-Ito:
- cada sistema tiene siempre el mismo color;
- las réplicas de un mismo sistema comparten color y se distinguen por el marcador del
  modelo (3B círculo, 7B cuadrado, 14B triángulo) y por su etiqueta, escrita junto a la marca;
- los baselines, sin modelo de lenguaje, llevan rombo;
- los controles (aleatorio y oráculo) son líneas grises discontinuas de referencia.

Los números de las figuras van con punto decimal y, a partir de cinco cifras, con un
espacio como separador de miles (miles(): 11 701, pero 6071).

La fusión va en negro, el color de Okabe-Ito que queda
con buen contraste sobre blanco (el amarillo apenas se ve).

estilo() carga experimentos/figuras/estilo.mplstyle y guardar() escribe el PDF vectorial
y un PNG de previsualización en experimentos/resultados/figuras/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

DIR_FIGURAS = Path(__file__).resolve().parent
RAIZ = DIR_FIGURAS.parent.parent
DIR_SALIDA = RAIZ / "experimentos" / "resultados" / "figuras"
ESTILO = DIR_FIGURAS / "estilo.mplstyle"

AZUL, NARANJA, VERDE, BERMELLON, ROSA, CELESTE, NEGRO = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000")
GRIS = "#888888"

# sistema -> (color, marcador, etiqueta)
SISTEMAS = {
    "c0": (AZUL, "D", "C0"),
    "bm25": (NARANJA, "D", "BM25"),
    "vecinos_comunes_B": (VERDE, "D", "Vecinos comunes (B)"),
    "adamic_adar_B": (BERMELLON, "D", "Adamic-Adar (B)"),
    "vecinos_comunes_A": (VERDE, "D", "Vecinos comunes (A)"),
    "adamic_adar_A": (BERMELLON, "D", "Adamic-Adar (A)"),
    "rrf_bm25_aa_B": (NEGRO, "D", "Fusión BM25 + AA (B)"),
    "c1": (ROSA, "o", "C1 (3B)"),
    "c1_7b": (ROSA, "s", "C1 (7B)"),
    "c1_14b": (ROSA, "^", "C1 (14B)"),
    "c2": (CELESTE, "o", "C2 (3B)"),
    "c2_7b": (CELESTE, "s", "C2 (7B)"),
    "c2_14b": (CELESTE, "^", "C2 (14B)"),
}
CONTROLES = {"aleatorio": "Aleatorio", "oraculo": "Oráculo"}
ESTILO_CONTROL = {"color": GRIS, "linestyle": "--", "linewidth": 0.8}
MARCADOR_MODELO = {"3b": "o", "7b": "s", "14b": "^"}


def color(sistema):
    return SISTEMAS[sistema][0]


def marcador(sistema):
    return SISTEMAS[sistema][1]


def etiqueta(sistema):
    return CONTROLES.get(sistema) or SISTEMAS[sistema][2]


def estilo():
    """Carga estilo.mplstyle. La fuente es la primera de font.serif que haya: Latin Modern
    Roman si está instalada, si no DejaVu Serif (la de matplotlib).

    - Latin Modern instalada después de que matplotlib guardara su caché de fuentes no sale
      en ella: se registra aquí desde los ficheros del sistema (lmroman*.otf).
    - Latin Modern es OpenType con contornos CFF, y matplotlib 3.11 la incrusta como
      TrueType (/FontFile2) con pdf.fonttype 42, un PDF mal formado: con una fuente CFF se
      guarda como Type 3. Con DejaVu Serif (TrueType) todo queda como está.
    """
    plt.style.use(ESTILO)
    if not any(f.name == "Latin Modern Roman" for f in font_manager.fontManager.ttflist):
        for ruta in font_manager.findSystemFonts():
            if Path(ruta).name.lower().startswith("lmroman"):
                font_manager.fontManager.addfont(ruta)
    elegida = font_manager.findfont(font_manager.FontProperties(family="serif"))
    with open(elegida, "rb") as f:
        if f.read(4) == b"OTTO":
            plt.rcParams["pdf.fonttype"] = 3


def miles(n):
    """Entero con espacio como separador de miles a partir de cinco cifras (11 701, pero 6071)."""
    return f"{n:,}".replace(",", " ") if abs(n) >= 10000 else str(n)


def separar(ys, minimo):
    """Posiciones para etiquetas en ys (mismo orden) separadas al menos `minimo`: de abajo arriba,
    cada una sube lo justo para no pisar la anterior."""
    orden = sorted(range(len(ys)), key=lambda i: ys[i])
    resultado = list(ys)
    for anterior, i in zip(orden, orden[1:]):
        resultado[i] = max(ys[i], resultado[anterior] + minimo)
    return resultado


def guardar(fig, nombre):
    """Escribe {nombre}.pdf (vectorial, sin fecha, para que sea reproducible) y {nombre}.png."""
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_SALIDA / f"{nombre}.pdf", format="pdf", metadata={"CreationDate": None})
    fig.savefig(DIR_SALIDA / f"{nombre}.png", format="png", dpi=200, metadata={"Software": None})
    plt.close(fig)
    print(f"Escritos {nombre}.pdf y {nombre}.png")
