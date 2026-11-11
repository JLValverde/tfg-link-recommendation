"""Elección del modelo de embeddings del recuperador (T1.6).

Para cada modelo candidato (sentence-transformers) y escenario, sobre la muestra de
DESARROLLO (200 consultas): recall@K del recuperador, K = 10, 20, 50, 100, en las
tres tareas. Anota tiempo de carga, tiempo de indexado, VRAM y tamaño.

Regla, fijada antes de mirar: gana el de mayor recall@50 implícito en FRÍO; si dos
están a menos de 0,01, el más pequeño (menos parámetros). Un modelo que no cargue
se documenta y se sigue con los demás.

Escribe experimentos/resultados/eleccion_embeddings.json.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe experimentos\\elegir_embeddings.py
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.candidatos import cargar_consultas  # noqa: E402
from arnes.evaluar import ic_bootstrap  # noqa: E402
from arnes.medir_vram import medir_en_proceso  # noqa: E402
from arnes.vista import Vista  # noqa: E402
from pipeline import recuperacion  # noqa: E402

RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "eleccion_embeddings.json"
KS = (10, 20, 50, 100)
ESCENARIOS = ("caliente", "frio")
MARGEN_EMPATE = 0.01


def relevantes_por_tarea(c):
    return {"completa": set(c["referencia"]), "implicita": set(c["referencia_implicita"]),
            "implicita_solo_ancla": set(c["referencia"]) - set(c["explicitos_solo_ancla"])}


def recall_recuperador(indice, consultas, ks=KS):
    """{tarea: {K: ic_bootstrap de recall@K}} y la latencia por consulta."""
    valores = {t: {k: [] for k in ks} for t in ("completa", "implicita", "implicita_solo_ancla")}
    latencias = []
    for c in consultas:
        t0 = time.perf_counter()
        top = [d for d, _ in indice.recuperar(Vista(c), c["idx"], max(ks))]
        latencias.append((time.perf_counter() - t0) * 1000)
        for tarea, rel in relevantes_por_tarea(c).items():
            for k in ks:
                valores[tarea][k].append(len(rel & set(top[:k])) / len(rel))
    return ({t: {str(k): ic_bootstrap(v) for k, v in ks_.items()} for t, ks_ in valores.items()},
            {"p50": float(np.percentile(latencias, 50)), "p95": float(np.percentile(latencias, 95))})


def tamano_en_disco(modelo):
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == modelo:
                return repo.size_on_disk
    except Exception:  # noqa: BLE001
        return None
    return None


def evaluar_modelo(modelo, consultas):
    import torch
    r = {"modelo": modelo}
    t0 = time.perf_counter()
    st = recuperacion.cargar_modelo(modelo)
    r["carga_s"] = round(time.perf_counter() - t0, 1)
    r["parametros"] = int(sum(p.numel() for p in st.parameters()))
    r["dim"] = int(st.get_sentence_embedding_dimension())
    r["max_seq_length"] = int(st.max_seq_length)
    r["dtype"] = str(next(st.parameters()).dtype)
    r["tamano_disco_bytes"] = tamano_en_disco(modelo)
    indice, vram = medir_en_proceso(lambda: recuperacion.IndiceEmbeddings(modelo, recalcular=True))
    r["indexado"] = vram  # duracion_s, base/pico nvidia-smi, torch max allocated/reserved
    r["recall"] = {}
    r["latencia_recuperar_ms"] = {}
    for esc in ESCENARIOS:
        r["recall"][esc], r["latencia_recuperar_ms"][esc] = recall_recuperador(indice, consultas[esc])
    del indice, st
    recuperacion.cargar_modelo.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main():
    consultas = {e: [c for c in cargar_consultas(e) if c["es_desarrollo"]] for e in ESCENARIOS}
    resultados = []
    for modelo in recuperacion.MODELOS_CANDIDATOS:
        print(f"== {modelo}", flush=True)
        try:
            r = evaluar_modelo(modelo, consultas)
            r["ok"] = True
            print(f"   recall@50 implícito frío: {r['recall']['frio']['implicita']['50']['media']:.4f}", flush=True)
        except Exception as e:  # noqa: BLE001
            r = {"modelo": modelo, "ok": False, "error": repr(e), "traza": traceback.format_exc()}
            print(f"   NO CARGA: {e!r}", flush=True)
        resultados.append(r)

    validos = [r for r in resultados if r["ok"]]
    clave = lambda r: r["recall"]["frio"]["implicita"]["50"]["media"]  # noqa: E731
    mejor = max(clave(r) for r in validos)
    empatados = [r for r in validos if mejor - clave(r) < MARGEN_EMPATE]
    ganador = min(empatados, key=lambda r: r["parametros"])
    salida = {"regla": "mayor recall@50 implícito en frío (desarrollo); a menos de 0,01, el más pequeño",
              "ganador": ganador["modelo"], "empatados_con_el_mejor": [r["modelo"] for r in empatados],
              "modelos": resultados}
    RUTA_SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Ganador: {ganador['modelo']}")


if __name__ == "__main__":
    main()
