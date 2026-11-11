"""fig_8_pr_umbral: curva precisión-recall por umbral en frío (protocolo §3; T2.8).

Para cada umbral, precisión y recall de los candidatos con puntuación >= umbral, sobre los
pares (puntuación, acierto) de las 1800 consultas juntas (arnes/metricas.py, curva_pr): el
umbral es global, el mismo para todas las consultas. El recall es sobre los enlaces de
referencia que hay entre las 50 candidatas: lo que el recuperador deja fuera está en el
techo (tabla_techo.md). Cada sistema usa su propia puntuación como confianza (C0, la
similitud coseno; BM25, su puntuación; C1 y C2, P-cota). La curva empieza en el umbral de
la puntuación más alta (un par recomendado, o los empatados con él): no hay ningún punto
sin recomendaciones.
C0, BM25, C1 (3B, 14B) y C2 (3B, 14B), con leyenda (las curvas se juntan a recall bajo);
el aleatorio, como línea gris discontinua.
Modo recuperador, tarea implícita. Lee metricas_{sistema}_recuperador_frio.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_pr_umbral.py
"""

import matplotlib.pyplot as plt

import colores as co
import datos

ESC = "frio"
SISTEMAS = (("c0", "-"), ("bm25", "-"), ("c1", "-"), ("c1_14b", "--"), ("c2", "-"), ("c2_14b", "--"))


def main():
    co.estilo()
    fig, ax = plt.subplots(figsize=(4.98, 3.08))
    for s, linea in SISTEMAS:
        curva = datos.curva_pr(s, ESC)
        ax.plot([p["recall"] for p in curva], [p["precision"] for p in curva], linestyle=linea,
                color=co.color(s), marker=co.marcador(s), markevery=0.12, markersize=3.5, linewidth=1.2,
                label=co.etiqueta(s))
    curva = datos.curva_pr("aleatorio", ESC)
    ax.plot([p["recall"] for p in curva], [p["precision"] for p in curva], **co.ESTILO_CONTROL,
            label=co.etiqueta("aleatorio"))
    # Las curvas se juntan a recall bajo y las etiquetas directas se pisarían: leyenda con
    # marcador y tipo de línea, en la esquina vacía
    ax.legend(loc="upper right", handlelength=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (enlaces de referencia entre los 50 candidatos)")
    ax.set_ylabel("Precisión")
    fig.tight_layout()
    co.guardar(fig, "fig_8_pr_umbral")


if __name__ == "__main__":
    main()
