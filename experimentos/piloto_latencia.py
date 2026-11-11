"""Piloto de latencia de qwen2.5:3b reordenando K candidatos (T0.9).

NO es un experimento de calidad: los candidatos son documentos al azar. Mide
cuánto tarda el modelo, para fijar K y el número de consultas.

- Consultas: 23 de desarrollo del escenario frío (random.Random(46)). Las 3
  primeras de cada configuración son de calentamiento y no cuentan.
- Candidatos: K documentos válidos al azar (random.Random(47 + idx)), sin la consulta.
- Texto: la consulta con su título y su introducción completa; cada candidato con
  su título y las primeras 60 palabras de su texto_visible.
- Configuraciones: lista con K = 20, 30, 50 (un prompt, ranking en JSON) y puntual
  con K = 20 (una llamada por candidato, sí/no, confianza por logprobs).
- num_ctx explícito en cada llamada: menor potencia de 2 que cubre el prompt más
  largo de la configuración con un 20 % de margen (tokens medidos con el propio
  modelo).

Escribe experimentos/resultados/piloto_latencia.jsonl (una línea por llamada) y
experimentos/resultados/piloto_latencia_resumen.json.

Uso (desde la raíz del repo, con el servidor de Ollama en marcha):
    .venv\\Scripts\\python.exe experimentos\\piloto_latencia.py
"""

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.vista import cargar_textos, texto_visible  # noqa: E402

RUTA_CONSULTAS = RAIZ / "datos" / "procesados" / "consultas_frio.jsonl"
RUTA_TEXTOS = RAIZ / "datos" / "procesados" / "textos.jsonl"
RUTA_LLAMADAS = RAIZ / "experimentos" / "resultados" / "piloto_latencia.jsonl"
RUTA_RESUMEN = RAIZ / "experimentos" / "resultados" / "piloto_latencia_resumen.json"
# 127.0.0.1 y no localhost: en Windows, localhost intenta antes IPv6 y añade ~2 s por conexión
OLLAMA = "http://127.0.0.1:11434/api/generate"
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
MODELO = "qwen2.5:3b"

N_CONSULTAS = 23
N_CALENTAMIENTO = 3
SEMILLA_CONSULTAS = 46
SEMILLA_CANDIDATOS = 47  # + idx
PALABRAS_CANDIDATO = 60
MARGEN_CTX = 1.2
CTX_MEDICION = 32768
CONSULTAS_TOTALES = 2000 * 2  # 2000 consultas x 2 escenarios
CONFIGURACIONES = [("lista", 20), ("lista", 30), ("lista", 50), ("puntual", 20)]
NUM_PREDICT = {"lista": 512, "puntual": 1}
TOP_LOGPROBS = 10
OPCIONES_COMUNES = {"temperature": 0, "seed": 0}
SESION = requests.Session()  # reutiliza la conexión entre llamadas


# ---------------------------------------------------------------- prompts

def bloque_consulta(titulo, intro):
    return f"Title: {titulo}\n{intro}"


def bloque_candidato(titulo, texto):
    return f"Title: {titulo}\n{' '.join(texto.split()[:PALABRAS_CANDIDATO])}"


def prompt_lista(consulta, candidatos):
    partes = ["You are helping to add hyperlinks to a Wikipedia article.", "",
              "Article:", consulta, "", "Candidate articles:"]
    for k, c in enumerate(candidatos, 1):
        partes += [f"[{k}] {c}", ""]
    partes.append(
        "Rank ALL the candidate articles from most to least likely to be linked from the "
        "article above. Answer only with JSON of the form {\"ranking\": [candidate numbers]}, "
        "using each candidate number exactly once.")
    return "\n".join(partes)


def prompt_puntual(consulta, candidato):
    return "\n".join(["Document A:", consulta, "", "Document B:", candidato, "",
                      "Should document A link to document B? Answer yes or no."])


# ---------------------------------------------------------------- llamadas

def llamar(prompt, num_ctx, forma):
    cuerpo = {"model": MODELO, "prompt": prompt, "stream": False, "keep_alive": "30m",
              "options": {**OPCIONES_COMUNES, "num_ctx": num_ctx, "num_predict": NUM_PREDICT[forma]}}
    if forma == "lista":
        cuerpo["format"] = "json"
    else:
        cuerpo["logprobs"] = True
        cuerpo["top_logprobs"] = TOP_LOGPROBS
    t0 = time.perf_counter()
    r = SESION.post(OLLAMA, json=cuerpo, timeout=600)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), ms


def tokens_prompt(prompt):
    """Tokens de entrada medidos con el propio modelo (contexto amplio, sin caché útil)."""
    d, _ = llamar(prompt, CTX_MEDICION, "puntual")
    return d["prompt_eval_count"] + d.get("prompt_eval_cached_count", 0)


def ollama_ps():
    try:
        salida = subprocess.run([str(OLLAMA_EXE), "ps"], capture_output=True, text=True,
                                timeout=30).stdout
    except OSError as e:
        return f"error: {e}"
    return salida.strip()


# ---------------------------------------------------------------- parseo

def parsear_lista(respuesta, k):
    try:
        d = json.loads(respuesta)
    except json.JSONDecodeError:
        return {"parseable": False}
    lista = d.get("ranking") if isinstance(d, dict) else d
    if isinstance(d, dict) and lista is None:
        listas = [v for v in d.values() if isinstance(v, list)]
        lista = listas[0] if listas else None
    if not isinstance(lista, list) or not all(isinstance(x, int) and not isinstance(x, bool)
                                              for x in lista):
        return {"parseable": False}
    validos = [x for x in lista if 1 <= x <= k]
    return {"parseable": True, "n_devueltos": len(lista), "n_distintos_validos": len(set(validos)),
            "repetidos": len(validos) - len(set(validos)), "inventados": len(lista) - len(validos)}


def probabilidad_si(d):
    """P(sí/yes) y P(no) en el primer token, sumando variantes de mayúsculas y espacios."""
    lp = d.get("logprobs") or []
    if not lp:
        return None, None
    p_si = p_no = 0.0
    for t in lp[0].get("top_logprobs", []):
        tok = t["token"].strip().lower()
        if tok in ("yes", "sí", "si"):
            p_si += math.exp(t["logprob"])
        elif tok == "no":
            p_no += math.exp(t["logprob"])
    return p_si, p_no


# ---------------------------------------------------------------- principal

def potencia_2(n):
    return 1 << math.ceil(math.log2(n))


def percentil(v, p):
    return float(np.percentile(np.asarray(v, dtype=float), p))


def main():
    consultas = {}
    for linea in RUTA_CONSULTAS.read_text(encoding="utf-8").splitlines():
        c = json.loads(linea)
        consultas[c["idx"]] = c
    titulos = {}
    for linea in RUTA_TEXTOS.read_text(encoding="utf-8").splitlines():
        f = json.loads(linea)
        titulos[f["idx"]] = f["titulo"].replace("_", " ")
    textos = cargar_textos()
    validos = sorted(textos)
    desarrollo = sorted(i for i, c in consultas.items() if c["es_desarrollo"])
    elegidas = random.Random(SEMILLA_CONSULTAS).sample(desarrollo, N_CONSULTAS)

    # Prompts de cada configuración
    trabajo = {}
    for forma, k in CONFIGURACIONES:
        nombre = f"{forma}_K{k}"
        trabajo[nombre] = []
        for q in elegidas:
            c = consultas[q]
            cands = random.Random(SEMILLA_CANDIDATOS + q).sample([v for v in validos if v != q], k)
            bq = bloque_consulta(titulos[q], textos[q][0])
            bc = [bloque_candidato(titulos[d], texto_visible(d, c)) for d in cands]
            if forma == "lista":
                prompts = [prompt_lista(bq, bc)]
            else:
                prompts = [prompt_puntual(bq, b) for b in bc]
            trabajo[nombre].append((q, cands, prompts))

    # num_ctx por configuración: tokens de los 3 prompts más largos (por caracteres)
    ctx = {}
    for nombre, items in trabajo.items():
        forma = nombre.split("_")[0]
        todos = sorted((p for _, _, ps in items for p in ps), key=len, reverse=True)[:3]
        max_tokens = max(tokens_prompt(p) for p in todos)
        necesario = MARGEN_CTX * max_tokens
        ctx[nombre] = {"max_tokens_prompt": max_tokens, "num_ctx": potencia_2(necesario),
                       "num_predict": NUM_PREDICT[forma],
                       "cabe_con_salida": max_tokens + NUM_PREDICT[forma] <= potencia_2(necesario)}
        print(f"{nombre}: prompt más largo {max_tokens} tokens -> num_ctx {ctx[nombre]['num_ctx']}",
              flush=True)

    RUTA_LLAMADAS.parent.mkdir(parents=True, exist_ok=True)
    vram = {}
    with RUTA_LLAMADAS.open("w", encoding="utf-8", newline="\n") as f:
        for nombre, items in trabajo.items():
            forma, k = nombre.split("_")[0], int(nombre.split("_K")[1])
            num_ctx = ctx[nombre]["num_ctx"]
            for n_consulta, (q, cands, prompts) in enumerate(items):
                for n_cand, p in enumerate(prompts):
                    d, ms = llamar(p, num_ctx, forma)
                    if nombre not in vram:  # tras la primera llamada: modelo cargado con este num_ctx
                        vram[nombre] = ollama_ps()
                    entrada = d.get("prompt_eval_count", 0) + d.get("prompt_eval_cached_count", 0)
                    fila = {
                        "configuracion": nombre, "idx": q,
                        "calentamiento": n_consulta < N_CALENTAMIENTO,
                        "candidato": cands[n_cand] if forma == "puntual" else None,
                        "latencia_ms": round(ms, 1),
                        "total_duration_ms": round(d.get("total_duration", 0) / 1e6, 1),
                        "tokens_entrada": entrada,
                        "tokens_entrada_cache": d.get("prompt_eval_cached_count", 0),
                        "tokens_salida": d.get("eval_count", 0),
                        "num_ctx": num_ctx, "truncado": entrada >= num_ctx,
                        "respuesta": d.get("response", ""),
                    }
                    if forma == "lista":
                        fila.update(parsear_lista(fila["respuesta"], k))
                    else:
                        p_si, p_no = probabilidad_si(d)
                        fila.update({"parseable": fila["respuesta"].strip().lower().startswith(("yes", "no", "sí")),
                                     "p_si": p_si, "p_no": p_no})
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                f.flush()
            print(f"{nombre}: hecho", flush=True)

    # Resumen
    llamadas = [json.loads(l) for l in RUTA_LLAMADAS.read_text(encoding="utf-8").splitlines()]
    resumen = {"modelo": MODELO, "consultas": elegidas, "n_calentamiento": N_CALENTAMIENTO,
               "configuraciones": {}}
    for nombre in trabajo:
        filas = [x for x in llamadas if x["configuracion"] == nombre and not x["calentamiento"]]
        por_consulta = {}
        for x in filas:
            por_consulta[x["idx"]] = por_consulta.get(x["idx"], 0) + x["latencia_ms"]
        lat = list(por_consulta.values())
        r = {
            "n_consultas": len(lat), "n_llamadas": len(filas),
            "latencia_consulta_ms": {"p50": round(percentil(lat, 50)), "p95": round(percentil(lat, 95)),
                                     "media": round(float(np.mean(lat)))},
            "tokens_entrada_mediana": percentil([x["tokens_entrada"] for x in filas], 50),
            "tokens_salida_mediana": percentil([x["tokens_salida"] for x in filas], 50),
            "tokens_entrada_cache_mediana": percentil([x["tokens_entrada_cache"] for x in filas], 50),
            **ctx[nombre], "truncados": sum(x["truncado"] for x in filas),
            "ollama_ps": vram.get(nombre),
            "pct_parseables": round(100 * sum(x["parseable"] for x in filas) / len(filas), 2),
            "horas_4000_consultas": {
                "con_media": round(CONSULTAS_TOTALES * float(np.mean(lat)) / 3.6e6, 2),
                "con_p50": round(CONSULTAS_TOTALES * percentil(lat, 50) / 3.6e6, 2)},
        }
        if nombre.startswith("lista"):
            ok = [x for x in filas if x["parseable"]]
            if ok:
                r["lista"] = {
                    "devueltos_media": round(float(np.mean([x["n_devueltos"] for x in ok])), 2),
                    "distintos_validos_media": round(float(np.mean([x["n_distintos_validos"] for x in ok])), 2),
                    "repetidos_media": round(float(np.mean([x["repetidos"] for x in ok])), 2),
                    "inventados_media": round(float(np.mean([x["inventados"] for x in ok])), 2),
                    "salidas_con_repetidos": sum(x["repetidos"] > 0 for x in ok),
                    "salidas_con_inventados": sum(x["inventados"] > 0 for x in ok),
                    "salidas_completas": sum(x["n_distintos_validos"] == int(nombre.split("_K")[1]) for x in ok),
                }
        else:
            ps = [x["p_si"] for x in filas if x["p_si"] is not None]
            r["puntual"] = {"llamadas_con_logprobs": len(ps),
                            "p_si_mediana": round(percentil(ps, 50), 4) if ps else None,
                            "p_si_p90": round(percentil(ps, 90), 4) if ps else None,
                            "respuestas_si": sum(x["respuesta"].strip().lower().startswith(("yes", "sí")) for x in filas)}
        resumen["configuraciones"][nombre] = r
    RUTA_RESUMEN.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_LLAMADAS.relative_to(RAIZ)} y {RUTA_RESUMEN.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
