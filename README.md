# Recomendación de enlaces en grafos documentales con modelos de lenguaje pequeños y locales

Código, datos procesados y resultados del Trabajo Fin de Grado «Recomendación de enlaces en grafos documentales mediante modelos de lenguaje pequeños y desplegables en local» (Grado en Ingeniería Informática, Universidad de Granada, 2026), de Jose Luis Valverde Azañón.

El trabajo estudia cuánto aporta un modelo de lenguaje pequeño, ejecutado en local y sin entrenamiento, al recomendar enlaces entre los documentos de una colección, dentro de un presupuesto fijo de memoria de GPU (8, 12 y 16 GB). El sistema recupera 50 candidatos con un modelo de *embeddings* y los reordena con un modelo de lenguaje, y se evalúa sobre WikiCS frente a BM25 y a heurísticas estructurales.

## Qué hay en el repositorio

| Carpeta | Contenido |
| --- | --- |
| `arnes/` | Arnés de evaluación: vista del grafo y del texto que ve cada sistema, candidatos, métricas, intervalos de confianza y medida del coste |
| `pipeline/` | El sistema evaluado, en cuatro capas (recuperación, contexto, reordenación y filtro), y las líneas base. `pipeline/prompts/` guarda los *prompts* congelados |
| `datos/` | Scripts que construyen el conjunto de evaluación y, en `datos/procesados/`, el conjunto ya construido |
| `experimentos/` | Un script por experimento, los scripts de tablas y figuras (`figuras/`) y los resultados (`resultados/`) |
| `tests/` | Tests, incluidos los que comprueban que ningún sistema ve los enlaces que debe encontrar |

## Requisitos

- Python 3.14 y las dependencias de `requirements.txt` (las versiones exactas de todo el entorno están en `requirements.lock`). PyTorch con CUDA se instala desde su índice:

  ```
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
  ```

- Para repetir los experimentos con modelos: una GPU NVIDIA y [Ollama](https://ollama.com) con los modelos `qwen2.5:3b`, `qwen2.5:7b-instruct-q4_K_M` y `qwen2.5:14b-instruct-q4_K_M`. El modelo de *embeddings*, `Qwen/Qwen3-Embedding-0.6B`, se descarga solo la primera vez.
- Las versiones con las que se obtuvieron los resultados (Ollama, controlador, GPU e identificador de cada modelo) están en `experimentos/entorno_congelado.json`. El equipo fue un PC con Windows 11 y una RTX 5060 Ti de 16 GB.

Todos los comandos se lanzan desde la raíz del repositorio.

## Repetir las tablas y las figuras (sin GPU)

Las tablas y figuras de la memoria salen de los resultados guardados, sin volver a ejecutar ningún modelo:

```
python experimentos/tablas_y_figuras.py
```

Escribe las tablas en `experimentos/resultados/tabla_*.md` y las figuras en `experimentos/resultados/figuras/`. Solo la figura de la distribución de grados necesita además el fichero original de WikiCS (ver «Datos»).

## Datos

**Incluidos** en `datos/procesados/`:

| Fichero | Qué es |
| --- | --- |
| `textos.jsonl` | Título e introducción de cada artículo, con las anclas de sus enlaces y la revisión de Wikipedia de la que salen |
| `grafo.json` | Los dos conjuntos de enlaces: todos los de WikiCS (A) y los escritos en el artículo (B) |
| `explicitos.jsonl`, `nombres_destinos.json` | Qué destinos nombra el texto de cada artículo |
| `consultas_{caliente,frio}.jsonl` | Las 2000 consultas de cada escenario, con sus enlaces ocultos y de referencia |
| `candidatos_recuperador_*.jsonl`, `candidatos_aleatorio_*.jsonl` | Las listas de 50 candidatos de cada consulta |

**No incluidos**, porque se pueden volver a obtener:

- **WikiCS original.** Se descarga del [repositorio de sus autores](https://github.com/pmernyei/wiki-cs-dataset), fijado a un commit, en `datos/raw/wikics/`:

  ```
  C=f5207315d649377f936edb66d7d93f5342f01d81
  mkdir -p datos/raw/wikics && cd datos/raw/wikics
  curl -sSfLO https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/$C/dataset/data.json
  curl -sSfLO https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/$C/dataset/metadata.json
  ```

  SHA-256: `9bf8cb3ef8eeae81b25e6ccbe0ea195600c205d7edf63ce04f2ec8d9c7dcb3d8` (`data.json`) y `fd9e8ea7b384a72183ca04d03db50d22e78b6f18735df494acb570098969fd2f` (`metadata.json`).
- **Wikitexto de Wikipedia.** `datos/descargar_textos.py` descarga, para cada artículo, su última revisión anterior al 20 de agosto de 2019 (unos 70 minutos). Solo hace falta para reconstruir `textos.jsonl` desde cero.
- **Embeddings.** `experimentos/elegir_embeddings.py` los calcula en `datos/procesados/embeddings/` (un par de minutos con GPU).

## Repetir los experimentos

Cada paso lee lo que dejan los anteriores. Los pasos 1 a 5 ya están hechos en `datos/procesados/`; repetirlos solo sirve para comprobar que dan lo mismo.

| Paso | Comando | Produce |
| --- | --- | --- |
| 1 | `python datos/descargar_textos.py` | `datos/raw/wikitext/`, `textos.jsonl` |
| 2 | `python datos/detectar_explicitos.py` | `explicitos.jsonl`, `nombres_destinos.json` |
| 3 | `python datos/construir_grafo.py` | `grafo.json` |
| 4 | `python datos/generar_consultas.py` | `consultas_{caliente,frio}.jsonl` |
| 5 | `python -m arnes.candidatos` | Candidatos del modo aleatorio |
| 6 | `python experimentos/elegir_embeddings.py` | Embeddings y elección del modelo |
| 7 | `python experimentos/recuperador.py` | Candidatos del recuperador y su *recall* |
| 8 | `python experimentos/evaluar_controles.py`, `evaluar_baselines.py`, `evaluar_estructurales.py`, `evaluar_fusion.py` | Controles y líneas base |
| 9 | `python experimentos/evaluar_c1.py` y `evaluar_c2.py` (`--modelo` y `--nivel` para cada tamaño) | Reordenación con modelo de lenguaje: varias horas por configuración |
| 10 | `python experimentos/tablas_y_figuras.py` | Tablas y figuras |

Tests: `python -m pytest tests -q`. Los de `tests/test_entorno.py` comprueban la GPU y Ollama, y fallan si no están; los que necesitan los embeddings se saltan.

## Resultados guardados

En `experimentos/resultados/`:

- `*_resumen.json`, `*_analisis.json` y `tabla_*.md`: las cifras de la memoria.
- `metricas_*.json`: las métricas de cada sistema, consulta a consulta.
- `crudas/<id_ejecucion>/`: las puntuaciones que dio cada modelo de lenguaje a cada candidato en la muestra principal, comprimidas.
- `ejecuciones/<id_ejecucion>.json`: la configuración, el hardware y las versiones de cada ejecución.
- `figuras/`: las figuras en PDF y PNG.

Dos avisos sobre estos ficheros:

- Las rutas absolutas del equipo en que se ejecutaron se han sustituido por rutas relativas. Es el único cambio respecto a los ficheros originales.
- El campo `git` de cada ejecución apunta a un commit del repositorio de desarrollo, que es privado. Este repositorio contiene el estado final de ese desarrollo, sin su historial.

## Licencias y atribución

- **Código:** licencia MIT (`LICENSE`).
- **WikiCS:** de Péter Mernyei y Cătălina Cangea, [distribuido con licencia MIT](https://github.com/pmernyei/wiki-cs-dataset).
- **Texto de Wikipedia:** los ficheros de `datos/procesados/` y los de `experimentos/resultados/` que contienen títulos o fragmentos de artículos derivan de la Wikipedia en inglés, cuyo texto escriben sus colaboradores y se publica con licencia [Creative Commons Atribución-CompartirIgual](https://creativecommons.org/licenses/by-sa/4.0/deed.es). Esos ficheros se distribuyen con la misma licencia. Cada artículo de `textos.jsonl` lleva su identificador de página (`pageid`) y de revisión (`revid`): la revisión original y su historial de autores están en `https://en.wikipedia.org/w/index.php?oldid=<revid>`. El texto se ha modificado: se conserva solo la introducción, sin el formato del wikitexto.
- **Modelos:** no se distribuyen. Qwen2.5 de 7B y 14B y Qwen3-Embedding-0.6B tienen licencia Apache 2.0; Qwen2.5 de 3B, una licencia de investigación no comercial (Qwen Research License).
