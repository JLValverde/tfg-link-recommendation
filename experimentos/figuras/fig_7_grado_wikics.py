"""fig_7_grado_wikics: distribución del grado saliente y del entrante de WikiCS (Figura 7.1; T2.8).

WikiCS tal como se publicó (datos/raw/wikics/data.json, que no se modifica), sobre los
enlaces únicos sin autobucles, como la Tabla 7.1 y datos/inspeccionar_wikics.py, que
dibujaba solo el saliente (grado_saliente_wikics.png).

Para cada grado k, el número de artículos con exactamente k enlaces, sin agrupar, con los
dos ejes en escala logarítmica: el entrante tiene una cola muy larga, y con intervalos
logarítmicos el recuento de cada intervalo crecería con su anchura y deformaría la forma.
Saliente y entrante en el mismo eje, con marcador distinto y su etiqueta al lado. Los
artículos con grado 0, que no caben en un eje logarítmico, se cuentan en el texto, junto
a la mediana y el máximo.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_7_grado_wikics.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import colores as co

RUTA_DATOS = co.RAIZ / "datos" / "raw" / "wikics" / "data.json"
SERIES = (("saliente", "o", "#222222"), ("entrante", "^", co.GRIS))
K_ETIQUETA = 5


def grados():
    links = json.loads(RUTA_DATOS.read_text(encoding="utf-8"))["links"]
    unicos = {(u, v) for u in range(len(links)) for v in links[u] if u != v}
    g = {"saliente": np.zeros(len(links), dtype=int), "entrante": np.zeros(len(links), dtype=int)}
    for u, v in unicos:
        g["saliente"][u] += 1
        g["entrante"][v] += 1
    return g


def main():
    co.estilo()
    g = grados()
    fig, ax = plt.subplots(figsize=(4.98, 3.08))
    resumen = []
    for nombre, m, c in SERIES:
        k, n = np.unique(g[nombre][g[nombre] > 0], return_counts=True)
        ax.plot(k, n, linestyle="", marker=m, markersize=2.5, color=c, alpha=0.8,
                markerfacecolor=c if m == "o" else "none", markeredgewidth=0.6)
        # Etiqueta junto a la serie en k = K_ETIQUETA, donde las dos están separadas
        i = int(np.searchsorted(k, K_ETIQUETA))
        arriba = nombre == "saliente"
        ax.annotate(f"Grado {nombre}", (k[i], n[i]), xytext=(4, 5) if arriba else (-4, -5),
                    textcoords="offset points", ha="left" if arriba else "right",
                    va="bottom" if arriba else "top", fontsize=7, color=c)
        resumen.append(f"{nombre.capitalize()}: mediana {np.median(g[nombre]):g}, máximo {co.miles(int(g[nombre].max()))},\n"
                       f"{co.miles(int((g[nombre] == 0).sum()))} artículos con grado 0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.98, 0.97, "\n".join(resumen) + f"\n{co.miles(len(g['saliente']))} artículos en total", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    ax.set_xlabel("Grado k (enlaces únicos, sin autobucles)")
    ax.set_ylabel("Artículos con grado k")
    fig.tight_layout()
    co.guardar(fig, "fig_7_grado_wikics")


if __name__ == "__main__":
    main()
