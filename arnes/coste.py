"""A3: decorador que registra el coste de un scorer por consulta en JSONL.

Uso:
    scorer_medido = con_coste(ruta, sistema="mi_sistema", modo="aleatorio", semilla=1)(scorer)

El scorer envuelto conserva la firma puntuar(consulta, candidatos, vista) y el
resultado; el arnés no sabe que el decorador existe. Por cada llamada (una por
consulta) se añade una línea al JSONL con: ts, id_ejecucion, sistema, modo,
escenario, semilla, consulta, latencia_ms, llamadas, tokens_in, tokens_out,
num_ctx, modelo y nivel de presupuesto (None si la ejecución no tiene; T2.6).

Cada decorador creado es una ejecución. Su fichero <directorio del JSONL>/ejecuciones/
<id_ejecucion>.json se escribe una sola vez, en la primera llamada, con: inicio,
sistema, modo, escenario, semilla, fichero de coste, `config` (lo que pasa el script:
modelo, digest, K, num_ctx, hash_prompt; ver config_ejecucion), `git` (hash y cambios
sin commitear), hardware, versiones y `entorno_coincide` frente a
experimentos/entorno_congelado.json (T2.0). Si el entorno no coincide, avisa sin fallar.

Los scorers informan de sus llamadas al modelo con anotar(...). Si no lo usan, o
si no hay decorador activo, llamadas y tokens quedan a 0 (anotar no hace nada).
"""

import contextvars
import functools
import hashlib
import json
import platform
import subprocess
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import requests

# Siempre 127.0.0.1: en Windows, localhost intenta antes IPv6 y añade ~2 s por conexión (T0.9)
OLLAMA_URL = "http://127.0.0.1:11434"

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ENTORNO_CONGELADO = RAIZ / "experimentos" / "entorno_congelado.json"
# Las salidas de los experimentos no cuentan como cambios del código sin commitear
EXCLUIR_DE_CAMBIOS = ("experimentos/resultados/",)
# Niveles de presupuesto: tope de memoria de GPU en MiB, N x 1000 (ADR "El presupuesto de
# recursos son tres topes de VRAM de 8, 12 y 16 GB"). arnes.agregar falla si el pico de una
# ejecución con nivel supera su tope.
NIVELES = {"bajo": 8000, "medio": 12000, "alto": 16000}

LIBRERIAS = ["numpy", "torch", "sentence-transformers", "transformers", "ollama",
             "requests", "mwparserfromhell", "pyahocorasick", "bm25s"]


@dataclass
class ContextoCoste:
    llamadas: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    num_ctx: int | None = None
    modelo: str | None = None


_CONTEXTO = contextvars.ContextVar("contexto_coste", default=None)


def anotar(llamadas=1, tokens_in=0, tokens_out=0, num_ctx=None, modelo=None):
    """El scorer informa de una o varias llamadas al modelo. Sin decorador activo, no hace nada."""
    ctx = _CONTEXTO.get()
    if ctx is None:
        return
    ctx.llamadas += llamadas
    ctx.tokens_in += tokens_in
    ctx.tokens_out += tokens_out
    if num_ctx is not None:
        ctx.num_ctx = max(ctx.num_ctx or 0, num_ctx)
    if modelo is not None:
        ctx.modelo = modelo


def _version(paquete):
    try:
        return version(paquete)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def entorno():
    """Hardware y versiones; se calcula una vez por proceso."""
    hw = {"so": platform.platform(), "cpu": platform.processor() or platform.machine()}
    driver = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        hw["gpu"] = gpu or None
        driver = gpu.split(",")[1].strip() if gpu.count(",") >= 2 else None
    except (OSError, subprocess.SubprocessError):
        hw["gpu"] = None
    versiones = {"python": platform.python_version(), "driver_nvidia": driver}
    versiones.update({lib: _version(lib) for lib in LIBRERIAS})
    try:
        import torch
        versiones["torch_cuda"] = torch.version.cuda
    except ImportError:
        versiones["torch_cuda"] = None
    try:
        versiones["ollama_servidor"] = requests.get(f"{OLLAMA_URL}/api/version", timeout=2).json()["version"]
    except (requests.RequestException, ValueError, KeyError):
        versiones["ollama_servidor"] = None
    return {"hw": hw, "versiones": versiones}


def estado_git():
    """Hash de HEAD y cambios sin commitear (`git status --porcelain`), sin las salidas de
    experimentos/resultados/. None en todo si git no está disponible."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True,
                                  timeout=10, check=True).stdout
        cambios = [l for l in git("status", "--porcelain", "--untracked-files=all").splitlines()
                   if not l[3:].strip('"').startswith(EXCLUIR_DE_CAMBIOS)]
        return {"hash": git("rev-parse", "HEAD").strip(), "hay_cambios": bool(cambios),
                "cambios_sin_commitear": cambios}
    except (OSError, subprocess.SubprocessError):
        return {"hash": None, "hay_cambios": None, "cambios_sin_commitear": None}


def comparar_entorno(versiones, ruta=RUTA_ENTORNO_CONGELADO):
    """Compara las versiones de Ollama (servidor) y del driver con las congeladas.

    Devuelve (coincide, diferencias): True si las dos coinciden; False si alguna difiere;
    None si no se puede comprobar (falta el fichero o alguna versión actual).
    """
    ruta = Path(ruta)
    if not ruta.exists():
        return None, [f"no existe {ruta.name}"]
    congelado = json.loads(ruta.read_text(encoding="utf-8"))
    pares = {"ollama_servidor": congelado["ollama"]["servidor"],
             "driver_nvidia": congelado["nvidia"]["driver"]}
    diferencias = [{"campo": k, "congelado": v, "actual": versiones.get(k)}
                   for k, v in pares.items() if versiones.get(k) != v]
    if any(d["actual"] is not None for d in diferencias):
        return False, diferencias
    return (None if diferencias else True), diferencias


def digest_modelo(nombre):
    """Digest del modelo en Ollama (GET /api/tags); None si no está o Ollama no responde."""
    try:
        modelos = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()["models"]
    except (requests.RequestException, ValueError, KeyError):
        return None
    return next((m["digest"] for m in modelos if m["name"] == nombre), None)


def config_ejecucion(modelo=None, K=None, num_ctx=None, prompt=None, **extra):
    """El diccionario `config` que pasa el script a con_coste.

    El digest se pide a Ollama si hay modelo; el prompt se guarda como su SHA-256.
    """
    return {"modelo": modelo, "digest": digest_modelo(modelo) if modelo else None, "K": K,
            "num_ctx": num_ctx,
            "hash_prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None,
            **extra}


def con_coste(ruta, sistema, modo, semilla=None, escenario=None, config=None, nivel=None):
    """Decorador: registra en `ruta` (JSONL, en modo añadir) el coste de cada llamada al scorer.

    escenario: si no se da, se toma de la primera consulta. config: ver config_ejecucion.
    nivel: nivel de presupuesto (clave de NIVELES) al que pertenece la ejecución; va en
    cada registro y en el fichero de la ejecución, con su tope.
    """
    if nivel is not None and nivel not in NIVELES:
        raise ValueError(f"Nivel desconocido: {nivel!r} (válidos: {sorted(NIVELES)})")
    ruta = Path(ruta)
    inicio = datetime.now(timezone.utc)
    id_ejecucion = (f"{inicio.strftime('%Y%m%dT%H%M%S')}_{sistema}_{modo}"
                    + (f"_s{semilla}" if semilla is not None else "") + f"_{uuid.uuid4().hex[:6]}")
    registrada = False

    def registrar_ejecucion(escenario_consulta):
        dir_ej = ruta.parent / "ejecuciones"
        dir_ej.mkdir(parents=True, exist_ok=True)
        env = entorno()
        coincide, diferencias = comparar_entorno(env["versiones"])
        if coincide is not True:
            warnings.warn(f"{id_ejecucion}: el entorno no coincide con entorno_congelado.json "
                          f"(o no se puede comprobar): {diferencias}", stacklevel=3)
        (dir_ej / f"{id_ejecucion}.json").write_text(json.dumps(
            {"id_ejecucion": id_ejecucion, "inicio": inicio.isoformat(timespec="seconds"),
             "sistema": sistema, "modo": modo, "escenario": escenario or escenario_consulta,
             "semilla": semilla, "fichero_coste": ruta.name, "config": config,
             "nivel": nivel, "tope_mib": NIVELES.get(nivel),
             "git": estado_git(), **env,
             "entorno_coincide": coincide, "entorno_diferencias": diferencias},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def decorador(scorer):
        @functools.wraps(scorer)
        def envuelto(consulta, candidatos, vista):
            nonlocal registrada
            if not registrada:
                registrar_ejecucion(consulta.get("escenario"))
                registrada = True
            ctx = ContextoCoste()
            token = _CONTEXTO.set(ctx)
            t0 = time.perf_counter()
            try:
                resultado = scorer(consulta, candidatos, vista)
            finally:
                latencia_ms = (time.perf_counter() - t0) * 1000
                _CONTEXTO.reset(token)
            fila = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "id_ejecucion": id_ejecucion, "sistema": sistema, "modo": modo,
                "escenario": consulta.get("escenario"), "semilla": semilla, "consulta": consulta["idx"],
                "latencia_ms": round(latencia_ms, 3),
                "llamadas": ctx.llamadas, "tokens_in": ctx.tokens_in, "tokens_out": ctx.tokens_out,
                "num_ctx": ctx.num_ctx, "modelo": ctx.modelo, "nivel": nivel,
            }
            ruta.parent.mkdir(parents=True, exist_ok=True)
            with ruta.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            return resultado
        envuelto.id_ejecucion = id_ejecucion
        return envuelto
    return decorador
