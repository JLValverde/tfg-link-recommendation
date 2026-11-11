"""Entorno congelado de la Fase 2 (T2.0): versiones, modelos y una prueba de logprobs.

Anota la versión del servidor y del cliente de Ollama, el driver de NVIDIA, la versión
CUDA que informa nvidia-smi y la GPU. De cada modelo, su digest y su cuantización
(/api/tags y /api/show), y el resultado de UNA llamada de prueba (temperatura 0,
num_predict 1, logprobs con top_logprobs) que solo comprueba que devuelve logprobs.
No mide latencia ni VRAM (eso es T2.6).

No descarga modelos: deben estar ya en Ollama (`ollama pull <etiqueta>`).
con_coste compara cada ejecución con este fichero (`entorno_coincide`).

Escribe experimentos/entorno_congelado.json.

Uso (desde la raíz del repo, con Ollama en marcha):
    .venv/Scripts/python.exe experimentos/congelar_entorno.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.coste import OLLAMA_URL  # noqa: E402

RUTA_SALIDA = RAIZ / "experimentos" / "entorno_congelado.json"
MODELOS = ("qwen2.5:3b", "qwen2.5:7b-instruct-q4_K_M", "qwen2.5:14b-instruct-q4_K_M")
PROMPT_PRUEBA = "Is Paris the capital of France? Answer Yes or No."
TOP_LOGPROBS = 5
NUM_CTX_PRUEBA = 2048


def ollama_cli():
    exe = shutil.which("ollama") or os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    salida = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=30,
                            env={**os.environ, "OLLAMA_HOST": OLLAMA_URL})
    m = re.search(r"client version is (\S+)", salida.stdout + salida.stderr) or \
        re.search(r"ollama version is (\S+)", salida.stdout + salida.stderr)
    return m.group(1) if m else None


def nvidia():
    gpu, driver, memoria = [x.strip() for x in subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip().split(",")]
    cabecera = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", cabecera)
    return {"driver": driver, "cuda": m.group(1) if m else None, "gpu": gpu, "memoria_total": memoria}


def modelo(nombre, tags):
    t = next((m for m in tags if m["name"] == nombre), None)
    if t is None:
        raise SystemExit(f"{nombre} no está en Ollama: ollama pull {nombre}")
    show = requests.post(f"{OLLAMA_URL}/api/show", json={"model": nombre}, timeout=30).json()
    d = requests.post(f"{OLLAMA_URL}/api/generate", timeout=600, json={
        "model": nombre, "prompt": PROMPT_PRUEBA, "stream": False, "keep_alive": 0,
        "logprobs": True, "top_logprobs": TOP_LOGPROBS,
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": NUM_CTX_PRUEBA}}).json()
    lp = d.get("logprobs") or []
    return {
        "digest": t["digest"], "tamano_bytes": t["size"],
        "cuantizacion": show["details"]["quantization_level"],
        "parametros": show["details"]["parameter_size"], "familia": show["details"]["family"],
        "prueba_logprobs": {
            "respuesta": d.get("response"),
            "devuelve_logprobs": bool(lp and lp[0].get("top_logprobs")),
            "n_top_logprobs": len(lp[0].get("top_logprobs", [])) if lp else 0,
            "token": lp[0].get("token") if lp else None,
            "logprob": lp[0].get("logprob") if lp else None,
            "top_logprobs": [{"token": x["token"], "logprob": x["logprob"]}
                             for x in lp[0].get("top_logprobs", [])] if lp else [],
        },
    }


def main():
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30).json()["models"]
    entorno = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ollama": {"servidor": requests.get(f"{OLLAMA_URL}/api/version", timeout=10).json()["version"],
                   "cliente_cli": ollama_cli(), "cliente_python": version("ollama")},
        "nvidia": nvidia(),
        "prueba_logprobs": {"prompt": PROMPT_PRUEBA, "temperature": 0, "num_predict": 1,
                            "num_ctx": NUM_CTX_PRUEBA, "top_logprobs": TOP_LOGPROBS},
        "modelos": {},
    }
    for nombre in MODELOS:
        entorno["modelos"][nombre] = modelo(nombre, tags)
        p = entorno["modelos"][nombre]
        print(f"{nombre}: {p['cuantizacion']}, {p['digest'][:12]}, "
              f"logprobs={p['prueba_logprobs']['devuelve_logprobs']}", flush=True)
    RUTA_SALIDA.write_text(json.dumps(entorno, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
