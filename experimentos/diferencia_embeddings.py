"""Diferencia pareada de recall@50 entre Qwen3-Embedding-0.6B y bge-small-en-v1.5.

En la muestra de desarrollo (200 consultas), con los embeddings guardados en
datos/procesados/embeddings/, recall@50 del recuperador por consulta para cada
modelo y diferencia Qwen3 - bge-small con bootstrap pareado (diferencia_pareada),
en las tres tareas y los dos escenarios. Solo se reporta: no cambia la elección
de T1.6.

Escribe experimentos/resultados/diferencia_qwen3_bge.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/diferencia_embeddings.py
"""

import gc
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.candidatos import cargar_consultas  # noqa: E402
from arnes.evaluar import diferencia_pareada, ic_bootstrap  # noqa: E402
from pipeline import recuperacion  # noqa: E402
from recuperador import ESCENARIOS, TAREAS, recall_por_consulta, recuperar  # noqa: E402

MODELOS = {"qwen3": "Qwen/Qwen3-Embedding-0.6B", "bge_small": "BAAI/bge-small-en-v1.5"}
K = 50
RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "diferencia_qwen3_bge.json"


def recall_modelo(modelo, consultas):
    """{escenario: {"idx": [...], "por_consulta": {tarea: {"recall@50": [...]}}}}."""
    ind = recuperacion.IndiceEmbeddings(modelo)
    salida = {}
    for esc in ESCENARIOS:
        tops, _ = recuperar(ind, consultas[esc], k=K)
        filas = recall_por_consulta(consultas[esc], tops)
        salida[esc] = {"idx": [f["idx"] for f in filas],
                       "por_consulta": {t: {f"recall@{K}": [f["recall"][t][str(K)] for f in filas]}
                                        for t in TAREAS}}
    recuperacion.cargar_modelo.cache_clear()
    gc.collect()
    return salida


def main():
    import torch
    consultas = {e: [c for c in cargar_consultas(e) if c["es_desarrollo"]] for e in ESCENARIOS}
    res = {}
    for nombre, modelo in MODELOS.items():
        res[nombre] = recall_modelo(modelo, consultas)
        torch.cuda.empty_cache()
    salida = {"muestra": "desarrollo", "k": K, "diferencia": "qwen3 - bge_small", "resultados": {}}
    for esc in ESCENARIOS:
        salida["resultados"][esc] = {}
        for t in TAREAS:
            m = f"recall@{K}"
            salida["resultados"][esc][t] = {
                "qwen3": ic_bootstrap(res["qwen3"][esc]["por_consulta"][t][m]),
                "bge_small": ic_bootstrap(res["bge_small"][esc]["por_consulta"][t][m]),
                "diferencia_pareada": diferencia_pareada(res["qwen3"][esc], res["bge_small"][esc], t, m),
            }
    RUTA_SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for esc in ESCENARIOS:
        for t in TAREAS:
            d = salida["resultados"][esc][t]["diferencia_pareada"]
            print(f"{esc} {t}: {d['diferencia']:+.4f} [{d['ic95'][0]:+.4f}, {d['ic95'][1]:+.4f}] n={d['n']}")


if __name__ == "__main__":
    main()
