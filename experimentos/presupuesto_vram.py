"""T2.6: C2 congelado con un modelo de lenguaje y el de embeddings cargados a la vez.

Para medir el pico de VRAM de la configuración final con cada modelo (ADR de los tres
topes de VRAM): carga Qwen3-Embedding-0.6B en la GPU con max_seq_length 512 y, en un
hilo, codifica sin parar títulos + introducciones de la colección (lotes de 32)
mientras el modelo de lenguaje responde las llamadas de C2 congelado
(pipeline/prompts/c2.json) en N_CONSULTAS consultas de DESARROLLO por escenario (las
primeras del fichero; no toca la muestra principal), modo recuperador.

Se lanza dentro de arnes.medir_vram (proceso aislado, con --descargar-todos), que da
el pico de toda la GPU. De aquí salen además la latencia por consulta (sin las 3 de
calentamiento de cada escenario) y las horas estimadas para 2000 consultas x 2
escenarios, y los casos (i)/(ii)/(iii) del sí y el no. La latencia está medida con el
modelo de embeddings compitiendo por la GPU.

Escribe en experimentos/resultados/: presupuesto_{tag}.json, coste_c2_presupuesto_{tag}_{escenario}_dev.jsonl
y diagnostico_/puntuaciones_ (no versionados).

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe -m arnes.medir_vram --salida experimentos/resultados/vram_presupuesto_{tag}.json --descargar-todos -- .venv/Scripts/python.exe experimentos/presupuesto_vram.py --modelo M
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # el modelo de embeddings ya está en la caché local

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
from arnes.candidatos import K_RECUPERADOR, cargar_candidatos, cargar_consultas  # noqa: E402
from arnes.coste import con_coste, config_ejecucion  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS, puntuar  # noqa: E402
from arnes.scorer import SCORERS  # noqa: E402
from arnes.vista import cargar_textos, cargar_titulos  # noqa: E402
from evaluar_c1 import contar  # noqa: E402
from evaluar_c2 import SISTEMA_MODELO  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402
from pipeline.recuperacion import texto_documento  # noqa: E402

MODO = "recuperador"
ESCENARIOS = ("caliente", "frio")
N_CONSULTAS = 20
CALENTAMIENTO = 3
MODELO_EMBEDDINGS = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LENGTH = 512
LOTE = 32


class Codificador(threading.Thread):
    """Codifica textos en bucle con el modelo de embeddings hasta que se le pide parar."""

    def __init__(self, modelo, textos):
        super().__init__(daemon=True)
        self.modelo, self.textos = modelo, textos
        self.parar, self.codificados, self.error = threading.Event(), 0, None

    def run(self):
        k = 0
        try:
            while not self.parar.is_set():
                lote = [self.textos[(k + i) % len(self.textos)] for i in range(LOTE)]
                self.modelo.encode(lote, batch_size=LOTE, normalize_embeddings=True, show_progress_bar=False)
                k += LOTE
                self.codificados += LOTE
        except Exception as e:  # se informa al final: sin codificador la medida no vale
            self.error = f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=sorted(SISTEMA_MODELO), required=True)
    args = ap.parse_args()
    if ro.C2 is None:
        sys.exit("C2 no está congelado (pipeline/prompts/c2.json)")
    import torch
    from sentence_transformers import SentenceTransformer

    ro.MODELO = args.modelo
    tag = SISTEMA_MODELO[args.modelo].replace("c2_", "") if args.modelo != "qwen2.5:3b" else "3b"
    textos, titulos = cargar_textos(), cargar_titulos()
    docs = [texto_documento(titulos[d], textos[d][0]) for d in sorted(textos)]
    st = SentenceTransformer(MODELO_EMBEDDINGS, device="cuda")
    st.max_seq_length = MAX_SEQ_LENGTH
    st.encode(docs[:LOTE], batch_size=LOTE, show_progress_bar=False)  # cargado antes de empezar
    codificador = Codificador(st, docs)
    codificador.start()
    t0 = time.perf_counter()
    salida = {"modelo": args.modelo, "modelo_embeddings": MODELO_EMBEDDINGS, "max_seq_length": st.max_seq_length,
              "n_consultas_por_escenario": N_CONSULTAS, "hash_c2": ro.hash_texto(ro.texto_c2(ro.C2)),
              "num_ctx": ro.NUM_CTX, "por_escenario": {}}
    for esc in ESCENARIOS:
        consultas = [c for c in cargar_consultas(esc) if c["es_desarrollo"]][:N_CONSULTAS]
        nombre = f"c2_presupuesto_{tag}_{MODO}_{esc}_dev"
        ruta_coste = DIR_RESULTADOS / f"coste_{nombre}.jsonl"
        ruta_diag = DIR_RESULTADOS / f"diagnostico_{nombre}.jsonl"
        for r in (ruta_coste, ruta_diag):
            r.unlink(missing_ok=True)
        ro.RUTA_DIAGNOSTICO = ruta_diag
        ro.reiniciar_contadores()
        config = config_ejecucion(modelo=args.modelo, K=K_RECUPERADOR, num_ctx=ro.NUM_CTX, prompt=ro.texto_c2(ro.C2),
                                  variante="congelada", modelo_embeddings=MODELO_EMBEDDINGS,
                                  max_seq_length=MAX_SEQ_LENGTH, n_consultas=N_CONSULTAS,
                                  proposito="T2.6: pico de VRAM con el modelo de embeddings a la vez")
        scorer = con_coste(ruta_coste, sistema=f"c2_presupuesto_{tag}", modo=MODO, escenario=esc,
                           config=config)(SCORERS["c2"])
        puntuar(scorer, consultas, cargar_candidatos("recuperador", esc), DIR_RESULTADOS / f"puntuaciones_{nombre}.jsonl")
        filas = [json.loads(l) for l in ruta_coste.read_text(encoding="utf-8").splitlines()]
        lat = [f["latencia_ms"] for f in sorted(filas, key=lambda f: f["ts"])[CALENTAMIENTO:]]
        salida["por_escenario"][esc] = {
            "id_ejecucion": scorer.id_ejecucion, "consultas": [c["idx"] for c in consultas],
            "latencia_ms_media": float(np.mean(lat)), "latencia_ms_p50": float(np.percentile(lat, 50)),
            "latencia_ms_p95": float(np.percentile(lat, 95)), "n_latencias": len(lat),
            "horas_estimadas_2000_consultas": float(np.mean(lat)) * 2000 / 3.6e6,
            "diagnostico": contar(ruta_diag)}
    codificador.parar.set()
    codificador.join(timeout=120)
    salida["horas_estimadas_2000_consultas_x_2_escenarios"] = sum(
        e["horas_estimadas_2000_consultas"] for e in salida["por_escenario"].values())
    salida["textos_codificados"] = codificador.codificados
    salida["error_codificador"] = codificador.error
    salida["torch_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    salida["torch_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    salida["duracion_s"] = round(time.perf_counter() - t0, 1)
    (DIR_RESULTADOS / f"presupuesto_{tag}.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(salida, ensure_ascii=False, indent=1))
    if codificador.error:
        sys.exit(f"El codificador falló: {codificador.error}")


if __name__ == "__main__":
    main()
