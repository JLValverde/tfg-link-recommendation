"""fig_8_coste_calidad: MRR frente a latencia p50 por consulta, en frío (T2.8).

x = latencia p50 por consulta, en segundos y escala logarítmica (los baselines, en su valor
medido); y = MRR con IC 95 %. Todos los sistemas que se evalúan en frío, con su etiqueta
al lado (la de C1 (7B), con una línea guía gris: no cabe entre los bigotes vecinos).
Modo recuperador, tarea implícita, 1800 consultas.

El eje y no empieza en 0: las diferencias entre sistemas (unas centésimas) no se verían.
Por eso aleatorio y oráculo, que quedan fuera, se escriben como texto en vez de como líneas.
Lee tabla_principal.json, los {sistema}_analisis.json y los coste_*_agregado.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_coste_calidad.py
"""

import matplotlib.pyplot as plt

import colores as co
import datos

ESC = "frio"
SISTEMAS = ("c0", "bm25", "c1", "c1_7b", "c1_14b", "c2", "c2_7b", "c2_14b")
# Dónde va la etiqueta de cada punto: a la derecha, a la izquierda, bajo el bigote o, con una
# línea guía desde lo alto del bigote, más a la derecha (C1 (7B), que no cabe entre los bigotes
# de C1 (3B), C2 (7B) y C1 (14B))
LADO = {"c1": "izquierda", "c1_7b": "guia", "c1_14b": "izquierda", "c2": "abajo"}
COLOCACION = {"derecha": ((6, 0), "left", "center"), "izquierda": ((-6, 0), "right", "center"),
              "abajo": ((0, -3), "center", "top"), "guia": ((24, -6), "left", "center")}
ANCLA = {"abajo": 0, "guia": 1}  # extremo del IC donde se ancla la etiqueta (0, inferior; 1, superior)
GUIA = {"arrowstyle": "-", "color": co.GRIS, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 2}


def main():
    co.estilo()
    fig, ax = plt.subplots(figsize=(4.98, 3.08))
    extremos = []
    for s in (s for s in SISTEMAS if datos.disponible(s, ESC)):
        x = datos.latencia_p50_ms(s, ESC) / 1000
        v = datos.metrica(s, ESC)
        lo, hi = v["ic95"]
        extremos += [lo, hi]
        ax.errorbar(x, v["media"], yerr=[[v["media"] - lo], [hi - v["media"]]], fmt=co.marcador(s),
                    color=co.color(s), markersize=5, elinewidth=0.9, capsize=2)
        lado = LADO.get(s, "derecha")
        desp, ha, va = COLOCACION[lado]
        y = v["ic95"][ANCLA[lado]] if lado in ANCLA else v["media"]
        ax.annotate(f"{co.etiqueta(s)}\n{v['media']:.3f}", (x, y), xytext=desp,
                    textcoords="offset points", ha=ha, va=va, fontsize=7, linespacing=1.0,
                    arrowprops=GUIA if lado == "guia" else None)
    ax.set_xscale("log")
    margen = 0.15 * (max(extremos) - min(extremos))
    ax.set_ylim(min(extremos) - margen, max(extremos) + margen)
    ax.set_xlim(ax.get_xlim()[0] / 3, ax.get_xlim()[1] * 3)
    controles = ", ".join(f"{co.etiqueta(c).lower()} {datos.metrica(c, ESC)['media']:.3f}"
                          for c in ("aleatorio", "oraculo"))
    ax.text(0.01, 0.02, f"Fuera de escala: {controles}", transform=ax.transAxes, fontsize=7, color=co.GRIS)
    ax.set_xlabel("Latencia p50 por consulta (s, escala logarítmica)")
    ax.set_ylabel("MRR (tarea implícita, frío)")
    fig.tight_layout()
    co.guardar(fig, "fig_8_coste_calidad")


if __name__ == "__main__":
    main()
