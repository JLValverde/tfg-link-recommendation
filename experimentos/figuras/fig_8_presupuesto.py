"""fig_8_presupuesto: MRR de C1 y C2 en cada nivel de presupuesto de VRAM (T2.8; dos paneles en T2.8b).

Dos paneles, frío | caliente, con el mismo eje y. En cada uno, x = tope de VRAM de cada
nivel (8, 12 y 16 GB, con el modelo que elige en T2.6); y = MRR con IC 95 % de C1 (rosa) y
C2 (celeste), con el marcador de cada modelo y su valor escrito encima del bigote (C1) o
debajo (C2); las dos series, algo separadas en x para que no se pisen los bigotes. Líneas de
referencia punteadas, con el color de su sistema: C0 y BM25 en frío; C0, Adamic-Adar (B) y
la fusión de BM25 y Adamic-Adar (B) en caliente. Una leyenda por panel, en la parte baja,
que está vacía.
Modo recuperador, 1800 consultas, tarea implícita. Lee tabla_principal.json y
presupuesto_niveles.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_presupuesto.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import colores as co
import datos

# (nombre, sistema por modelo, desplazamiento en x, dónde va el valor)
SERIES = (("C1", {"3b": "c1", "7b": "c1_7b", "14b": "c1_14b"}, -0.35, "encima"),
          ("C2", {"3b": "c2", "7b": "c2_7b", "14b": "c2_14b"}, 0.35, "debajo"))
FONDO = {"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85}
REFERENCIAS = {"frio": ("c0", "bm25"), "caliente": ("c0", "adamic_adar_B", "rrf_bm25_aa_B")}


def panel(ax, esc, niveles, xs):
    handles = []
    for nombre, sistema_de, desp, lado in SERIES:
        puntos = [(x + desp, sistema_de[m], datos.metrica(sistema_de[m], esc))
                  for x, (_, _, m) in zip(xs, niveles) if datos.disponible(sistema_de[m], esc)]
        c = co.color(puntos[0][1])
        ax.plot([x for x, _, _ in puntos], [v["media"] for _, _, v in puntos], color=c, linewidth=1.1, zorder=2)
        for x, s, v in puntos:
            lo, hi = v["ic95"]
            ax.errorbar(x, v["media"], yerr=[[v["media"] - lo], [hi - v["media"]]], fmt=co.marcador(s),
                        color=c, markersize=4.5, elinewidth=0.9, capsize=1.5, zorder=3)
            y, va = (hi + 0.007, "bottom") if lado == "encima" else (lo - 0.007, "top")
            ax.text(x, y, f"{v['media']:.3f}", ha="center", va=va, fontsize=6.5, bbox=FONDO, zorder=4)
        handles.append(Line2D([], [], color=c, linewidth=1.1, label=nombre))
    for s in REFERENCIAS[esc]:
        y = datos.metrica(s, esc)["media"]
        ax.axhline(y, color=co.color(s), linestyle=":", linewidth=1, zorder=1)
        handles.append(Line2D([], [], color=co.color(s), linestyle=":", linewidth=1,
                              label=f"{co.etiqueta(s)}: {y:.3f}"))
    ax.legend(handles=handles, loc="lower center", fontsize=6.5, handlelength=1.8, labelspacing=0.3)
    ax.set_xticks(xs, [f"{x:g} GB\n{m.upper()}" for x, (_, _, m) in zip(xs, niveles)])
    ax.set_xlim(xs[0] - 2.2, xs[-1] + 2.2)
    ax.set_xlabel(f"Tope de VRAM y modelo ({datos.NOMBRE_ESC[esc]})")


def main():
    co.estilo()
    niveles = datos.niveles()
    xs = [tope / 1000 for _, tope, _ in niveles]
    fig, ejes = plt.subplots(1, 2, sharey=True, figsize=(4.98, 3.3))
    for ax, esc in zip(ejes, datos.ESCENARIOS):
        panel(ax, esc, niveles, xs)
    ejes[0].set_ylim(0, 0.6)
    ejes[0].set_ylabel("MRR (tarea implícita)")
    fig.tight_layout()
    co.guardar(fig, "fig_8_presupuesto")


if __name__ == "__main__":
    main()
