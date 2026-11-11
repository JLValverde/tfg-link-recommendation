"""fig_8_caliente: MRR y precision@1 con IC 95 % en caliente (T2.8).

Como fig_8_frio, más los baselines estructurales sobre B: vecinos comunes, Adamic-Adar y
la fusión RRF de BM25 y Adamic-Adar (B). La fusión exploratoria de C1 y Adamic-Adar (B)
(T2.8b, a posteriori) no va. Lee experimentos/resultados/tabla_principal.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/figuras/fig_8_caliente.py
"""

import colores as co
from fig_8_frio import figura_escenario

SISTEMAS = ("c0", "bm25", "vecinos_comunes_B", "adamic_adar_B", "rrf_bm25_aa_B",
            "c1", "c1_7b", "c1_14b", "c2", "c2_7b", "c2_14b")


def main():
    co.estilo()
    figura_escenario("caliente", SISTEMAS, "fig_8_caliente")


if __name__ == "__main__":
    main()
