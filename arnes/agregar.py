"""A3: agrega los JSONL de coste en tablas (p50 y p95).

Agrupa por (sistema, modo, escenario, nivel). Antes, en cada ejecución (id_ejecucion y
escenario) se descartan las 3 primeras consultas, en orden de ts, como
calentamiento.

Presupuesto (T2.6, confirmación del ADR de los tres topes de VRAM): si algún registro
lleva nivel, la agregación necesita el pico de VRAM de su ejecución (picos_vram(), de
los ficheros de arnes.medir_vram) y falla si falta o si supera el tope del nivel.

Uso (desde la raíz del repo):
    .venv\\Scripts\\python.exe -m arnes.agregar experimentos\\resultados\\coste_*.jsonl [--vram vram_*.json] [--salida tabla.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from arnes.coste import NIVELES

DESCARTAR = 3
CAMPOS = ("latencia_ms", "llamadas", "tokens_in", "tokens_out")


class PresupuestoSuperado(ValueError):
    """Una ejecución supera el tope de VRAM de su nivel, o no hay pico medido para comprobarlo."""


def leer(rutas):
    filas = []
    for ruta in rutas:
        filas += [json.loads(l) for l in Path(ruta).read_text(encoding="utf-8").splitlines() if l]
    return filas


def picos_vram(rutas):
    """{id_ejecucion: pico_mib} de los ficheros de arnes.medir_vram (campo "ejecuciones")."""
    picos = {}
    for ruta in rutas:
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        for id_ejecucion in d.get("ejecuciones") or []:
            picos[id_ejecucion] = max(picos.get(id_ejecucion, 0), d["pico_mib"])
    return picos


def comprobar_presupuesto(filas, picos=None):
    """Falla si una ejecución con nivel no tiene pico medido o su pico supera el tope del nivel.

    Devuelve {id_ejecucion: {nivel, tope_mib, pico_mib}} de las ejecuciones con nivel.
    """
    niveles = defaultdict(set)
    for f in filas:
        if f.get("nivel") is not None:
            niveles[f["id_ejecucion"]].add(f["nivel"])
    comprobadas = {}
    for id_ejecucion, ns in sorted(niveles.items()):
        if len(ns) > 1:
            raise PresupuestoSuperado(f"{id_ejecucion}: registros con niveles distintos {sorted(ns)}")
        nivel = ns.pop()
        if nivel not in NIVELES:
            raise PresupuestoSuperado(f"{id_ejecucion}: nivel desconocido {nivel!r}")
        pico = (picos or {}).get(id_ejecucion)
        if pico is None:
            raise PresupuestoSuperado(f"{id_ejecucion}: nivel {nivel} sin pico de VRAM medido")
        if pico > NIVELES[nivel]:
            raise PresupuestoSuperado(f"{id_ejecucion}: pico {pico} MiB > tope {NIVELES[nivel]} MiB "
                                      f"del nivel {nivel}")
        comprobadas[id_ejecucion] = {"nivel": nivel, "tope_mib": NIVELES[nivel], "pico_mib": pico}
    return comprobadas


def agregar(filas, descartar=DESCARTAR, picos=None):
    comprobar_presupuesto(filas, picos)
    por_ejecucion = defaultdict(list)
    for f in filas:
        por_ejecucion[(f["sistema"], f["modo"], f["escenario"], f.get("nivel"), f["id_ejecucion"])].append(f)
    grupos = defaultdict(list)
    descartadas = defaultdict(int)
    for (*grupo, _), fs in sorted(por_ejecucion.items(), key=lambda x: str(x[0])):
        fs = sorted(fs, key=lambda f: f["ts"])
        grupos[tuple(grupo)] += fs[descartar:]
        descartadas[tuple(grupo)] += min(descartar, len(fs))
    tabla = []
    for (sistema, modo, escenario, nivel), fs in sorted(grupos.items(),
                                                        key=lambda x: tuple(v or "" for v in x[0])):
        fila = {"sistema": sistema, "modo": modo, "escenario": escenario,
                **({"nivel": nivel} if nivel is not None else {}),
                "n_consultas": len(fs), "descartadas": descartadas[(sistema, modo, escenario, nivel)]}
        for campo in CAMPOS:
            v = np.asarray([f[campo] for f in fs], dtype=float)
            fila[campo] = ({"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
                           if len(v) else None)
        tabla.append(fila)
    return tabla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rutas", nargs="+")
    ap.add_argument("--vram", nargs="*", default=[], help="ficheros de arnes.medir_vram (picos por ejecución)")
    ap.add_argument("--salida", default=None)
    args = ap.parse_args()
    tabla = agregar(leer(args.rutas), picos=picos_vram(args.vram))
    texto = json.dumps(tabla, ensure_ascii=False, indent=2)
    if args.salida:
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
