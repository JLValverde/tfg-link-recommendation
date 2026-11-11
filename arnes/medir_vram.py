"""A3: medición de VRAM de una fase, en un proceso aislado.

El modelo de lenguaje vive en el proceso de Ollama, así que la VRAM se mide con
nvidia-smi sobre toda la GPU: memoria usada antes de la fase (base) y pico
durante la fase, muestreando cada 200 ms. El incremento es pico - base.

Para lo que corre en el propio proceso (por ejemplo, embeddings con torch),
medir_en_proceso() añade torch.cuda.max_memory_allocated/reserved, con reset
previo de los contadores.

Uso como ejecutable (la fase es un comando aparte):
    .venv\\Scripts\\python.exe -m arnes.medir_vram --salida vram.json [--descargar-modelo qwen2.5:3b] -- comando args...

Con --descargar-modelo se descarga antes el modelo de Ollama (keep_alive 0), para
que la base no incluya el modelo ya cargado; con --descargar-todos, todos los que
estén cargados (T2.6: medir un modelo sin que otro siga en la GPU).

La salida lista en "ejecuciones" los id_ejecucion que el comando creó en
experimentos/resultados/ejecuciones/ (T2.6): arnes.agregar.picos_vram enlaza así el
pico con cada ejecución para comprobar el tope de su nivel de presupuesto.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from arnes.coste import OLLAMA_URL, RAIZ

INTERVALO_MS = 200
DIR_EJECUCIONES = RAIZ / "experimentos" / "resultados" / "ejecuciones"


def memoria_usada_mib():
    salida = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True).stdout
    return int(salida.strip().splitlines()[0])


class Muestreador:
    """Muestrea la memoria usada de la GPU cada INTERVALO_MS con nvidia-smi en bucle."""

    def __init__(self):
        self.muestras = []
        self._proc = None
        self._hilo = None

    def __enter__(self):
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-lms", str(INTERVALO_MS)], stdout=subprocess.PIPE, text=True)
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()
        return self

    def _leer(self):
        for linea in self._proc.stdout:
            linea = linea.strip()
            if linea.isdigit():
                self.muestras.append(int(linea))

    def __exit__(self, *exc):
        self._proc.terminate()
        self._proc.wait(timeout=10)
        self._hilo.join(timeout=5)

    @property
    def pico(self):
        return max(self.muestras) if self.muestras else None


def descargar_modelo(modelo, espera_max_s=30):
    """Pide a Ollama que descargue el modelo de la GPU y espera a que desaparezca de /api/ps."""
    requests.post(f"{OLLAMA_URL}/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=60)
    limite = time.time() + espera_max_s
    while time.time() < limite:
        cargados = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
        if not any(m.get("name", "").startswith(modelo) for m in cargados):
            return True
        time.sleep(0.5)
    return False


def descargar_todos():
    """Descarga todos los modelos cargados en Ollama. Devuelve {modelo: descarga_confirmada}."""
    cargados = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    return {m["name"]: descargar_modelo(m["name"]) for m in cargados}


def _ejecuciones(directorio):
    return {p.stem for p in Path(directorio).glob("*.json")} if Path(directorio).exists() else set()


def medir_comando(comando, modelo_a_descargar=None, todos=False, dir_ejecuciones=DIR_EJECUCIONES):
    descargados = descargar_todos() if todos else None
    if modelo_a_descargar:
        descargado = descargar_modelo(modelo_a_descargar)
    else:
        descargado = None
    if todos or modelo_a_descargar:
        time.sleep(1)
    antes = _ejecuciones(dir_ejecuciones)
    # En Windows, CreateProcess no encuentra un ejecutable dado como ruta relativa con
    # barras normales (.venv/Scripts/python.exe): se pasa a ruta absoluta si existe.
    if comando and Path(comando[0]).exists():
        comando = [str(Path(comando[0]).resolve()), *comando[1:]]
    base = memoria_usada_mib()
    t0 = time.perf_counter()
    with Muestreador() as m:
        rc = subprocess.run(comando).returncode
    pico = max(m.pico or base, base)
    return {"comando": comando, "codigo_salida": rc, "duracion_s": round(time.perf_counter() - t0, 2),
            "base_mib": base, "pico_mib": pico, "incremento_mib": pico - base,
            "n_muestras": len(m.muestras), "intervalo_ms": INTERVALO_MS,
            "modelo_descargado_antes": modelo_a_descargar, "descarga_confirmada": descargado,
            "descargados_todos": descargados,
            "ejecuciones": sorted(_ejecuciones(dir_ejecuciones) - antes)}


def medir_en_proceso(funcion, *args, **kwargs):
    """Ejecuta funcion en este proceso midiendo nvidia-smi y, si hay CUDA, los contadores de torch.

    Devuelve (resultado, medidas).
    """
    import torch

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    base = memoria_usada_mib()
    t0 = time.perf_counter()
    with Muestreador() as m:
        resultado = funcion(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
    pico = max(m.pico or base, base)
    medidas = {"duracion_s": round(time.perf_counter() - t0, 2), "base_mib": base, "pico_mib": pico,
               "incremento_mib": pico - base, "n_muestras": len(m.muestras)}
    if cuda:
        medidas["torch_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        medidas["torch_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    return resultado, medidas


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        sys.exit("Uso: python -m arnes.medir_vram [--salida F] [--descargar-modelo M] [--descargar-todos] -- comando ...")
    corte = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=None)
    ap.add_argument("--descargar-modelo", default=None)
    ap.add_argument("--descargar-todos", action="store_true")
    args = ap.parse_args(argv[:corte])
    resultado = medir_comando(argv[corte + 1:], args.descargar_modelo, args.descargar_todos)
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.salida:
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
    print(texto)
    return resultado


if __name__ == "__main__":
    main()
