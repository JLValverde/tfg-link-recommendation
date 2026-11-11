"""fig_8_ablacion: diferencias pareadas de la ablación con IC 95 % (gráfico de bosque; T2.8, bloques en T2.8b).

Tres bloques de filas, en dos paneles (MRR y precision@1):
- "Efecto de la reordenación": C1 − C0 con el 3B;
- "Efecto del contexto": C2 − C1 con cada modelo (3B, 7B, 14B);
- "Efecto del tamaño": C1 (7B) − C1 (3B), C1 (14B) − C1 (7B) y C1 (14B) − C1 (3B).
Cada fila es una comparación con dos marcas, frío (rellena, arriba) y caliente (hueca,
abajo): la diferencia media por consulta con su IC por bootstrap pareado y su valor al lado,
en el color y con el marcador del primer sistema de la resta. La línea vertical es el 0.
Las comparaciones que aún no están en la tabla (falta un sistema) no se dibujan.
Modo recuperador, tarea implícita, 1800 consultas. Lee tabla_principal.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_ablacion.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import colores as co
import datos

BLOQUES = (
    ("Efecto de la reordenación", (("c1", "c0", "C1 − C0 (3B)"),)),
    ("Efecto del contexto", (("c2", "c1", "C2 − C1 (3B)"), ("c2_7b", "c1_7b", "C2 − C1 (7B)"),
                             ("c2_14b", "c1_14b", "C2 − C1 (14B)"))),
    ("Efecto del tamaño", (("c1_7b", "c1", "C1 (7B) − C1 (3B)"), ("c1_14b", "c1_7b", "C1 (14B) − C1 (7B)"),
                           ("c1_14b", "c1", "C1 (14B) − C1 (3B)"))),
)
METRICAS = (("mrr", "MRR (A − B)"), ("precision@1", "Precisión@1 (A − B)"))
# Posición de cada escenario dentro de su fila, y relleno de la marca
ESCENARIO = {"frio": (0.2, True), "caliente": (-0.2, False)}


def filas():
    """[(etiqueta, (a, b) o None si es la cabecera de un bloque)], de arriba abajo."""
    resultado = []
    for titulo, comparaciones in BLOQUES:
        presentes = [(nombre, (a, b)) for a, b, nombre in comparaciones
                     if all(datos.diferencia(a, b, esc) is not None for esc in ESCENARIO)]
        if presentes:
            resultado += [(titulo, None)] + presentes
    return resultado


def main():
    co.estilo()
    lista = filas()
    ys = list(range(len(lista)))[::-1]
    fig, ejes = plt.subplots(1, 2, sharey=True, figsize=(4.98, 0.75 + 0.38 * len(lista)))
    for ax, (m, titulo) in zip(ejes, METRICAS):
        extremos = [0.0]
        for y, (_, comp) in zip(ys, lista):
            if comp is None:
                continue
            a, b = comp
            for esc, (dy, relleno) in ESCENARIO.items():
                d = datos.diferencia(a, b, esc, m)
                lo, hi = d["ic95"]
                extremos += [lo, hi]
                ax.errorbar(d["diferencia"], y + dy, xerr=[[d["diferencia"] - lo], [hi - d["diferencia"]]],
                            fmt=co.marcador(a), color=co.color(a), markerfacecolor=co.color(a) if relleno else "white",
                            markersize=4, elinewidth=0.9, capsize=1.5)
                ax.annotate(f"{d['diferencia']:+.3f}".replace("-", "−"), (hi, y + dy), xytext=(4, 0),
                            textcoords="offset points", fontsize=7, va="center", ha="left")
        ax.axvline(0, color="#333333", linewidth=0.6, linestyle="--")
        rango = max(extremos) - min(extremos)
        ax.set_xlim(min(extremos) - 0.08 * rango, max(extremos) + 0.3 * rango)
        ax.set_xlabel(titulo)
    ejes[0].set_yticks(ys, [nombre for nombre, _ in lista])
    for etiqueta, (_, comp) in zip(ejes[0].get_yticklabels(), lista):
        if comp is None:
            etiqueta.set_fontweight("bold")
    for ax in ejes:
        ax.tick_params(axis="y", length=0)
    ejes[0].set_ylim(-0.6, len(lista) - 0.4)
    marca = {"marker": "o", "color": "#333333", "linestyle": ""}
    fig.legend(handles=[Line2D([], [], **marca, label="frío"),
                        Line2D([], [], **marca, markerfacecolor="white", label="caliente")],
               loc="upper center", ncol=2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    co.guardar(fig, "fig_8_ablacion")


if __name__ == "__main__":
    main()
