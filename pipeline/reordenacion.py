"""Capa 3 del pipeline: reordenación con un modelo pequeño local vía prompt (Ollama).

C1 (T2.2): una llamada por candidato a /api/chat, con un único mensaje de usuario y
sin mensaje de sistema propio. El texto del prompt está congelado en
pipeline/prompts/c1.txt (la variante 2, elegida en T2.2b con la regla del roadmap;
las variantes están en experimentos/variantes_c1.py): ese fichero no se toca.
La forma está decidida en el ADR "La
reordenación puntúa cada candidato por separado":
- la consulta va al principio del prompt, y las llamadas de una consulta van
  seguidas para que Ollama reutilice la caché del prefijo;
- temperatura 0, num_predict 1, num_ctx explícito y keep_alive para que el modelo
  no se descargue entre consultas;
- el texto es siempre el texto_visible de la Vista: título + introducción, con la
  introducción recortada por palabras (400 en la consulta y 200 en el candidato).

Confianza del primer token: cada token del top_logprobs se normaliza (solo letras, en
minúsculas), y las variantes de "yes" y de "no" se agregan por logsumexp.
confianza = lse(yes) - lse(no). Hay tres casos, y se cuentan por separado:
  (i) están los dos; (ii) falta uno; (iii) faltan los dos.
Con cada llamada se calculan dos políticas:
- P-none: en (ii) y en (iii), None (la peor puntuación);
- P-cota: en (ii), el que falta toma como logprob el menor del top (es una cota
  superior); en (iii), None.
El scorer devuelve P-cota. Las dos se guardan en el JSONL de diagnóstico
(RUTA_DIAGNOSTICO, que fija el script), junto con el token generado y el
top_logprobs completo. Un error de la llamada o una respuesta vacía dan None con
las dos políticas, y se cuentan.

CONTADORES acumula los casos, los errores, las respuestas que no son "yes"/"no",
los recortes y los truncados (prompt de num_ctx tokens o más). El script lo pone a
cero antes de cada ejecución.

C2 (T2.4): la receta de C1 más el contexto estructural de pipeline/contexto.py (listas
de vecinos visibles en B y de vecinos comunes). Su especificación (una plantilla por
escenario y los parámetros de las listas) está congelada en pipeline/prompts/c2.json;
mientras no existe, C2 es None y el script de desarrollo fija la variante. Mismas
llamadas, confianza, política y diagnóstico que C1.

MODELO se lee en cada llamada: las réplicas (T2.7) solo lo cambian a nivel de módulo.
"""

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import ollama

from arnes.coste import OLLAMA_URL, anotar
from arnes.scorer import registrar
from pipeline.contexto import Contexto

MODELO = "qwen2.5:3b"
TOP_LOGPROBS = 20  # el máximo que acepta Ollama 0.40.2 (T2.2a)
MAX_PALABRAS_CONSULTA = 400
MAX_PALABRAS_CANDIDATO = 200
# Menor potencia de 2 >= 1,2 x el prompt más largo posible, medido con qwen2.5:3b
# (experimentos/num_ctx_c1.py -> resultados/num_ctx_c1.json)
NUM_CTX = 4096
KEEP_ALIVE = "30m"
# Prompt congelado (T2.2b). Los saltos de línea se normalizan por si git lo saca con CRLF.
RUTA_PROMPT = Path(__file__).resolve().parent / "prompts" / "c1.txt"
PLANTILLA = RUTA_PROMPT.read_bytes().decode("utf-8").replace("\r\n", "\n")
RUTA_C2 = RUTA_PROMPT.with_name("c2.json")


def texto_c2(espec):
    """Texto canónico de una especificación de C2: es lo que se congela y lo que se hashea."""
    return json.dumps(espec, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def cargar_c2(ruta=RUTA_C2):
    """Especificación congelada de C2, o None si todavía no existe."""
    if not ruta.exists():
        return None
    return json.loads(ruta.read_bytes().decode("utf-8").replace("\r\n", "\n"))


C2 = cargar_c2()

RUTA_DIAGNOSTICO = None
CASOS = ("ambos", "falta_uno", "faltan_ambos")
CONTADORES = {}
_PALABRA = re.compile(r"\S+")


def reiniciar_contadores():
    CONTADORES.clear()
    CONTADORES.update({**{c: 0 for c in CASOS}, "error": 0, "respuesta_no_si_no": 0,
                       "llamadas": 0, "consultas_recortadas": 0, "candidatos_recortados": 0,
                       "truncados": 0})


reiniciar_contadores()


@lru_cache(maxsize=1)
def cliente():
    return ollama.Client(host=OLLAMA_URL)


# ---------------------------------------------------------------- texto

def recortar(texto, max_palabras):
    """Primeras max_palabras palabras (separadas por espacio en blanco), sin tocar el
    formato de lo que queda. Devuelve (texto, si_se_recortó)."""
    for k, m in enumerate(_PALABRA.finditer(texto), start=1):
        if k == max_palabras:
            fin = m.end()
            return texto[:fin], bool(texto[fin:].strip())
    return texto, False


def construir_prompt(titulo_consulta, texto_consulta, titulo_candidato, texto_candidato):
    return PLANTILLA.format(titulo_consulta=titulo_consulta, texto_consulta=texto_consulta,
                            titulo_candidato=titulo_candidato, texto_candidato=texto_candidato)


# ---------------------------------------------------------------- confianza

def normalizar(token):
    """Solo letras, en minúsculas: " Yes" -> "yes", "-No" -> "no", "_yes" -> "yes"."""
    return "".join(ch for ch in token if ch.isalpha()).lower()


def logsumexp(valores):
    m = max(valores)
    return m + math.log(sum(math.exp(v - m) for v in valores))


def confianza(top):
    """top: [(token, logprob)] del primer token generado.

    Devuelve {caso, lse_si, lse_no, conf_cota, conf_none}. lse_* es None si la
    variante no está en el top.
    """
    si = [lp for t, lp in top if normalizar(t) == "yes"]
    no = [lp for t, lp in top if normalizar(t) == "no"]
    lse_si = logsumexp(si) if si else None
    lse_no = logsumexp(no) if no else None
    if si and no:
        conf = lse_si - lse_no
        return {"caso": "ambos", "lse_si": lse_si, "lse_no": lse_no, "conf_cota": conf, "conf_none": conf}
    if not si and not no:
        return {"caso": "faltan_ambos", "lse_si": None, "lse_no": None, "conf_cota": None, "conf_none": None}
    cota = min(lp for _, lp in top)
    conf = (lse_si if si else cota) - (lse_no if no else cota)
    return {"caso": "falta_uno", "lse_si": lse_si, "lse_no": lse_no, "conf_cota": conf, "conf_none": None}


def top_de_respuesta(r):
    """[(token, logprob)] del top del primer token de una respuesta de Ollama, más el token
    generado si no estuviera en el top. ([], None) si no hay logprobs."""
    lp = r.logprobs or []
    if not lp:
        return [], None
    primero = lp[0]
    top = [(t.token, t.logprob) for t in (primero.top_logprobs or [])]
    if primero.token not in {t for t, _ in top}:
        top.append((primero.token, primero.logprob))
    return top, primero.token


# ---------------------------------------------------------------- scorer

def llamar(prompt, modelo=None, num_ctx=None):
    return cliente().chat(model=modelo or MODELO, messages=[{"role": "user", "content": prompt}],
                          logprobs=True, top_logprobs=TOP_LOGPROBS, keep_alive=KEEP_ALIVE,
                          options={"temperature": 0, "num_predict": 1, "num_ctx": num_ctx or NUM_CTX})


def hash_texto(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def prompts_c2(espec, consulta, vista):
    """Constructor de prompts de C2 para una consulta: f(c, texto_q, texto_c) -> (prompt, extra).

    extra va al diagnóstico: cuántos vecinos tienen la consulta y el candidato y cuántos comunes.
    """
    q = consulta["idx"]
    plantilla = espec["plantillas"][consulta["escenario"]]
    ctx = Contexto(vista, grafo=espec["grafo"], max_vecinos=espec["max_vecinos"],
                   separador=espec["separador"], vacio=espec["vacio"])
    lista_q = ctx.lista(q) if "{vecinos_consulta}" in plantilla else None
    con_comunes = "{comunes}" in plantilla

    def construir(c, texto_q, texto_c):
        prompt = plantilla.format(
            titulo_consulta=vista.titulo(q), texto_consulta=texto_q,
            titulo_candidato=vista.titulo(c), texto_candidato=texto_c,
            vecinos_consulta=lista_q, vecinos_candidato=ctx.lista(c),
            comunes=ctx.comunes(q, c) if con_comunes else None)
        return prompt, {"n_vecinos_consulta": ctx.grado(q), "n_vecinos_candidato": ctx.grado(c),
                        "n_comunes": len(ctx.vecinos(q) & ctx.vecinos(c))}
    return construir


@registrar("c1")
def c1(consulta, candidatos, vista):
    q = consulta["idx"]
    return _puntuar(consulta, candidatos, vista,
                    lambda c, tq, tc: (construir_prompt(vista.titulo(q), tq, vista.titulo(c), tc), {}))


@registrar("c2")
def c2(consulta, candidatos, vista):
    if C2 is None:
        raise RuntimeError("C2 no está congelado (pipeline/prompts/c2.json) ni lo ha fijado el script")
    return _puntuar(consulta, candidatos, vista, prompts_c2(C2, consulta, vista))


def _puntuar(consulta, candidatos, vista, construir):
    """Una llamada por candidato; construir(c, texto_q, texto_c) -> (prompt, extra del diagnóstico)."""
    q = consulta["idx"]
    texto_q, recortada = recortar(vista.texto(q), MAX_PALABRAS_CONSULTA)
    CONTADORES["consultas_recortadas"] += recortada
    puntuaciones, filas = {}, []
    for c in candidatos:
        texto_c, recortado = recortar(vista.texto(c), MAX_PALABRAS_CANDIDATO)
        CONTADORES["candidatos_recortados"] += recortado
        CONTADORES["llamadas"] += 1
        prompt, extra = construir(c, texto_q, texto_c)
        fila = {"consulta": q, "escenario": consulta["escenario"], "candidato": c,
                "recorte_consulta": recortada, "recorte_candidato": recortado,
                "hash_prompt": hash_texto(prompt)[:16], **extra}
        try:
            r = llamar(prompt)
            top, token = top_de_respuesta(r)
            if not top:
                raise ValueError("respuesta sin logprobs")
        except Exception as e:  # error de la llamada o respuesta vacía: None con las dos políticas
            CONTADORES["error"] += 1
            anotar(llamadas=1, num_ctx=NUM_CTX, modelo=MODELO)
            puntuaciones[c] = None
            filas.append({**fila, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        conf = confianza(top)
        CONTADORES[conf["caso"]] += 1
        CONTADORES["respuesta_no_si_no"] += normalizar(token) not in ("yes", "no")
        CONTADORES["truncados"] += r.prompt_eval_count >= NUM_CTX
        anotar(llamadas=1, tokens_in=r.prompt_eval_count, tokens_out=r.eval_count,
               num_ctx=NUM_CTX, modelo=MODELO)
        puntuaciones[c] = conf["conf_cota"]
        filas.append({**fila, "token": token, **conf, "top": top,
                      "prompt_eval_count": r.prompt_eval_count, "eval_count": r.eval_count,
                      "prompt_eval_ms": r.prompt_eval_duration / 1e6 if r.prompt_eval_duration else None,
                      "total_ms": r.total_duration / 1e6 if r.total_duration else None})
    if RUTA_DIAGNOSTICO is not None:
        with open(RUTA_DIAGNOSTICO, "a", encoding="utf-8", newline="\n") as f:
            for fila in filas:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    return puntuaciones
