"""Batería de comprobación del sí/no (T2.2a): ¿el sesgo hacia "No" es del modelo o de la vía?

Pasa las 40 preguntas de experimentos/bateria_si_no.json (20 "Yes" y 20 "No", con el
sufijo " Answer Yes or No.") por los tres modelos de entorno_congelado.json y por dos
vías:
1. chat: /api/chat con un único mensaje de usuario, sin mensaje de sistema propio
   (la vía del scorer C1: pipeline.reordenacion.llamar);
2. generate: /api/generate como en experimentos/congelar_entorno.py.
Temperatura 0, num_predict 1, num_ctx 2048 y top_logprobs 20 (el máximo que acepta
Ollama). La confianza es la de C1 (pipeline.reordenacion.confianza): lse(yes) - lse(no)
sobre las variantes normalizadas del top, con P-cota si falta una.

Por modelo y vía reporta:
- aciertos (sobre 40) por el token generado y por el signo de la confianza;
- la confianza media en las preguntas "Yes" y en las "No";
- cuántas veces falta del top el "yes" o el "no".

Escribe experimentos/resultados/bateria_si_no.json (resumen y detalle por pregunta).

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/bateria_si_no.py
"""

import json
import statistics
import sys
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.coste import OLLAMA_URL  # noqa: E402
from pipeline import reordenacion as ro  # noqa: E402

RUTA_PREGUNTAS = RAIZ / "experimentos" / "bateria_si_no.json"
RUTA_ENTORNO = RAIZ / "experimentos" / "entorno_congelado.json"
RUTA_SALIDA = RAIZ / "experimentos" / "resultados" / "bateria_si_no.json"
SUFIJO = " Answer Yes or No."
NUM_CTX = 2048


def por_chat(modelo, prompt):
    r = ro.llamar(prompt, modelo=modelo, num_ctx=NUM_CTX)
    return ro.top_de_respuesta(r)


def por_generate(modelo, prompt):
    d = requests.post(f"{OLLAMA_URL}/api/generate", timeout=600, json={
        "model": modelo, "prompt": prompt, "stream": False, "keep_alive": ro.KEEP_ALIVE,
        "logprobs": True, "top_logprobs": ro.TOP_LOGPROBS,
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX}}).json()
    lp = d.get("logprobs") or []
    if not lp:
        return [], None
    top = [(t["token"], t["logprob"]) for t in lp[0].get("top_logprobs", [])]
    if lp[0]["token"] not in {t for t, _ in top}:
        top.append((lp[0]["token"], lp[0]["logprob"]))
    return top, lp[0]["token"]


VIAS = {"chat": por_chat, "generate": por_generate}


def descargar(modelo):
    requests.post(f"{OLLAMA_URL}/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=60)


def main():
    preguntas = json.loads(RUTA_PREGUNTAS.read_text(encoding="utf-8"))["preguntas"]
    modelos = list(json.loads(RUTA_ENTORNO.read_text(encoding="utf-8"))["modelos"])
    detalle, resumen = [], []
    for modelo in modelos:
        for via, f in VIAS.items():
            filas = []
            for p in preguntas:
                top, token = f(modelo, p["pregunta"] + SUFIJO)
                conf = ro.confianza(top)
                filas.append({"modelo": modelo, "via": via, "pregunta": p["pregunta"], "esperada": p["esperada"],
                              "token": token, "token_norm": ro.normalizar(token or ""), **conf, "top": top})
            detalle += filas

            def media(esperada):
                v = [x["conf_cota"] for x in filas if x["esperada"] == esperada and x["conf_cota"] is not None]
                return statistics.fmean(v) if v else None
            r = {"modelo": modelo, "via": via, "n": len(filas),
                 "aciertos_token": sum(x["token_norm"] == x["esperada"] for x in filas),
                 "aciertos_signo": sum(x["conf_cota"] is not None
                                       and (x["conf_cota"] > 0) == (x["esperada"] == "yes") for x in filas),
                 "respuestas_yes": sum(x["token_norm"] == "yes" for x in filas),
                 "confianza_media_yes": media("yes"), "confianza_media_no": media("no"),
                 "falta_yes": sum(x["lse_si"] is None for x in filas),
                 "falta_no": sum(x["lse_no"] is None for x in filas),
                 "casos": {c: sum(x["caso"] == c for x in filas) for c in ro.CASOS}}
            resumen.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        descargar(modelo)
    RUTA_SALIDA.write_text(json.dumps({"num_ctx": NUM_CTX, "top_logprobs": ro.TOP_LOGPROBS, "sufijo": SUFIJO,
                                       "resumen": resumen, "detalle": detalle},
                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
