"""fig_8_completa_implicita: precision@1 en la tarea completa frente a la implícita, en frío (T2.8).

Una fila por sistema con dos marcas unidas por una línea: la tarea completa (marcador
relleno) y la implícita (hueco), cada una con su IC 95 % y su valor escrito al lado
(implícita a la izquierda, completa a la derecha). Aleatorio y oráculo, en gris y con la
línea discontinua, como referencias. Modo recuperador, 1800 consultas.
Lee tabla_principal.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_completa_implicita.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import colores as co
import datos

ESC = "frio"
METRICA = "precision@1"
SISTEMAS = ("oraculo", "c0", "bm25", "c1", "c1_7b", "c1_14b", "c2", "c2_7b", "c2_14b", "aleatorio")


def main():
    co.estilo()
    sistemas = [s for s in SISTEMAS if datos.disponible(s, ESC)]
    fig, ax = plt.subplots(figsize=(4.98, 0.6 + 0.26 * len(sistemas)))
    for y, s in enumerate(reversed(sistemas)):
        control = s in co.CONTROLES
        c = co.GRIS if control else co.color(s)
        m = "o" if control else co.marcador(s)
        comp, impl = (datos.metrica(s, ESC, METRICA, tarea=t) for t in ("completa", "implicita"))
        ax.plot([impl["media"], comp["media"]], [y, y], color=c, linewidth=0.9,
                linestyle="--" if control else "-", zorder=1)
        for v, relleno, lado in ((comp, c, "derecha"), (impl, "white", "izquierda")):
            lo, hi = v["ic95"]
            ax.errorbar(v["media"], y, xerr=[[v["media"] - lo], [hi - v["media"]]], fmt=m, color=c,
                        markerfacecolor=relleno, markersize=4.5, elinewidth=0.9, capsize=1.5, zorder=2)
            x, ha = (hi + 0.01, "left") if lado == "derecha" else (lo - 0.01, "right")
            ax.text(x, y, f"{v['media']:.3f}", va="center", ha=ha, fontsize=6.5)
    ax.set_yticks(range(len(sistemas)), [co.etiqueta(s) for s in reversed(sistemas)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(sistemas) - 0.4)
    ax.set_xlabel("Precisión@1 (frío)")
    ax.legend(handles=[Line2D([], [], marker="o", color="#333333", linestyle="", label="tarea completa"),
                       Line2D([], [], marker="o", color="#333333", markerfacecolor="white", linestyle="",
                              label="tarea implícita")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3)
    fig.tight_layout()
    co.guardar(fig, "fig_8_completa_implicita")


if __name__ == "__main__":
    main()
