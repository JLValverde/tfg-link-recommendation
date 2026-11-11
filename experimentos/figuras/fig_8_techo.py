"""fig_8_techo: techo del recuperador, recall@K según K (redacción del cap. 8).

Fracción de los enlaces de referencia que entra entre los K primeros candidatos del
recuperador (Qwen3-Embedding-0.6B), media e IC 95 % sobre las 1800 consultas, para
K = 10, 20, 50 y 100. Una línea por escenario (frío negro, caliente gris) y tarea
(implícita continua, completa discontinua). La línea vertical marca K = 50, el usado.
No son sistemas, así que no usa los colores de colores.SISTEMAS.
Lee experimentos/resultados/recall_recuperador.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_techo.py
"""

import matplotlib.pyplot as plt

import colores as co
import datos

KS = (10, 20, 50, 100)
SERIES = (("frio", "implicita", "Frío, implícita", "black", "-"),
          ("caliente", "implicita", "Caliente, implícita", co.GRIS, "-"),
          ("frio", "completa", "Frío, completa", "black", "--"),
          ("caliente", "completa", "Caliente, completa", co.GRIS, "--"))


def main():
    co.estilo()
    recall = datos.leer("recall_recuperador.json")["recall"]
    fig, ax = plt.subplots(figsize=(4.98, 2.9))
    ax.axvline(50, color=co.GRIS, linewidth=0.6, linestyle=":")
    ax.text(51.5, 0.015, "K = 50 (el usado)", fontsize=7, color=co.GRIS, va="bottom")
    finales = []
    for escenario, tarea, nombre, color, trazo in SERIES:
        vals = [recall[escenario][tarea][str(k)] for k in KS]
        medias = [v["media"] for v in vals]
        err = [[m - v["ic95"][0] for m, v in zip(medias, vals)],
               [v["ic95"][1] - m for m, v in zip(medias, vals)]]
        ax.errorbar(KS, medias, yerr=err, color=color, linestyle=trazo, marker="o", markersize=3,
                    elinewidth=0.8, capsize=1.5, linewidth=1.2)
        finales.append((medias[-1], nombre, color))
    # Etiquetas a la derecha, separadas para que no se pisen
    ys = co.separar([y for y, _, _ in finales], 0.024)
    for y, (_, nombre, color) in zip(ys, finales):
        ax.text(KS[-1] + 4, y, nombre, fontsize=8, color="black" if color == "black" else "#555555", va="center")
    ax.set_xticks(KS)
    ax.set_xlim(5, 150)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("K (candidatos que pasan a la reordenación)")
    ax.set_ylabel("Recall@K")
    fig.tight_layout()
    co.guardar(fig, "fig_8_techo")


if __name__ == "__main__":
    main()
