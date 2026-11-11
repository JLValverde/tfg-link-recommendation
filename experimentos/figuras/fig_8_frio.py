"""fig_8_frio: MRR y precision@1 con IC 95 % en frío (T2.8).

Modo recuperador, 1800 consultas, tarea implícita. Un punto por sistema con su IC como
bigote y su valor escrito al lado: C0, BM25, C1 y C2 (3B, 7B, 14B; las que existan).
Aleatorio y oráculo, como líneas grises discontinuas de referencia.
Lee experimentos/resultados/tabla_principal.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_frio.py
"""

import matplotlib.pyplot as plt

import colores as co
import datos

SISTEMAS = ("c0", "bm25", "c1", "c1_7b", "c1_14b", "c2", "c2_7b", "c2_14b")
METRICAS = (("mrr", "MRR"), ("precision@1", "Precisión@1"))


def figura_escenario(escenario, sistemas, nombre):
    sistemas = [s for s in sistemas if datos.disponible(s, escenario)]
    fig, ejes = plt.subplots(1, 2, sharey=True, figsize=(4.98, 0.6 + 0.24 * len(sistemas)))
    for ax, (m, titulo) in zip(ejes, METRICAS):
        tope = datos.metrica("oraculo", escenario, m)["media"]
        for y, s in enumerate(reversed(sistemas)):
            v = datos.metrica(s, escenario, m)
            lo, hi = v["ic95"]
            ax.errorbar(v["media"], y, xerr=[[v["media"] - lo], [hi - v["media"]]], fmt=co.marcador(s),
                        color=co.color(s), markersize=4, elinewidth=0.9, capsize=1.5)
            ax.text(hi + 0.012 * tope, y, f"{v['media']:.3f}", va="center", fontsize=7)
        for control in ("aleatorio", "oraculo"):
            x = datos.metrica(control, escenario, m)["media"]
            ax.axvline(x, **co.ESTILO_CONTROL)
            ax.text(x + 0.01 * tope, len(sistemas) - 0.35, co.etiqueta(control), ha="left", va="bottom",
                    fontsize=7, color=co.GRIS)
        ax.set_xlim(0, tope * 1.08)
        ax.set_ylim(-0.6, len(sistemas) - 0.1)
        ax.set_xlabel(titulo)
    ejes[0].set_yticks(range(len(sistemas)), [co.etiqueta(s) for s in reversed(sistemas)])
    for ax in ejes:
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    co.guardar(fig, nombre)


def main():
    co.estilo()
    figura_escenario("frio", SISTEMAS, "fig_8_frio")


if __name__ == "__main__":
    main()
