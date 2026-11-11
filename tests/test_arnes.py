"""Tests del arnés de evaluación (T1.1-T1.5) con los scorers de control.

Necesitan los ficheros de datos/procesados/ (consultas y candidatos aleatorios).
"""

import filecmp
import json
import shutil
import sys

import pytest

from arnes import evaluar as ev
from arnes.agregar import agregar
from arnes.candidatos import (MAX_REFERENCIAS, N_CANDIDATOS, SEMILLAS, cargar_candidatos,
                              cargar_consultas, muestrear_referencias, ruta_candidatos)
from arnes.coste import anotar, con_coste
from arnes.metricas import MRR, curva_pr, hits_en, precision_en, precision_normalizada_en
from arnes.rangos import ordenar, rankings_por_tarea
from arnes.scorer import SCORERS, registrar
from arnes.vista import cargar_adyacencia

ESCENARIOS = ("caliente", "frio")

if not ruta_candidatos("aleatorio", "frio", 1).exists():
    pytest.skip("faltan los candidatos aleatorios (python -m arnes.candidatos)", allow_module_level=True)


@pytest.fixture(scope="module")
def consultas():
    return {e: {c["idx"]: c for c in cargar_consultas(e)} for e in ESCENARIOS}


@pytest.fixture(scope="module")
def resultados(tmp_path_factory):
    """Los tres controles, en los dos escenarios, semilla 1, muestra principal."""
    d = tmp_path_factory.mktemp("controles")
    return {(s, e): ev.evaluar(s, "aleatorio", e, 1, dir_resultados=d)
            for s in ("aleatorio", "siempre_none", "oraculo") for e in ESCENARIOS}


# ---------------------------------------------------------------- candidatos

@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_candidatos_aleatorios(consultas, escenario):
    A = cargar_adyacencia("A")
    entrantes = {}
    for u, vec in A.items():
        for v in vec:
            entrantes.setdefault(v, set()).add(u)
    for s in SEMILLAS:
        cands = cargar_candidatos("aleatorio", escenario, s)
        assert sorted(cands) == sorted(consultas[escenario])
        for q, lista in cands.items():
            c = consultas[escenario][q]
            assert len(lista) == N_CANDIDATOS == len(set(lista))
            refs = set(lista) & set(c["referencia"])
            assert len(refs) == min(MAX_REFERENCIAS, len(c["referencia"]))
            assert refs & set(c["referencia_implicita"])  # nunca sin implícitos (T1.2)
            negativos = set(lista) - refs
            assert q not in lista
            assert not negativos & (A[q] | entrantes.get(q, set()))


def test_muestreo_estratificado():
    import random
    # 14 referencias, 3 implícitas: 10 * 3/14 = 2,1 -> 2 implícitas y 8 explícitas
    ref, impl = list(range(14)), [0, 1, 2]
    m = muestrear_referencias(ref, impl, random.Random(0))
    assert len(m) == 10 and len(set(m) & set(impl)) == 2
    # 20 referencias, 1 implícita: la proporción daría 0,5 -> 1 (al menos 1)
    m = muestrear_referencias(list(range(20)), [7], random.Random(0))
    assert len(m) == 10 and 7 in m
    # sin implícitas
    m = muestrear_referencias(list(range(20)), [], random.Random(0))
    assert len(m) == 10


# ---------------------------------------------------------------- 1. control negativo

@pytest.mark.parametrize("escenario", ESCENARIOS)
@pytest.mark.parametrize("tarea", ["completa", "implicita", "implicita_solo_ancla"])
def test_control_negativo_aleatorio(resultados, consultas, escenario, tarea):
    """El aleatorio da precisión@1 y @5 dentro del IC del valor esperado |R_lista|/N_lista."""
    cands = cargar_candidatos("aleatorio", escenario, 1)
    esperado = []
    for q, c in consultas[escenario].items():
        lista = set(cands[q]) - set(c["neutros_plantilla"])
        rel = set(c["referencia"])
        if tarea == "implicita":
            lista -= set(c["explicitos"])
            rel = set(c["referencia_implicita"])
        elif tarea == "implicita_solo_ancla":
            lista -= set(c["explicitos_solo_ancla"])
            rel = set(c["referencia"]) - set(c["explicitos_solo_ancla"])
        esperado.append(len(lista & rel) / len(lista))
    e = sum(esperado) / len(esperado)
    agg = resultados[("aleatorio", escenario)]["agregado"][tarea]
    for m in ("precision@1", "precision@5"):
        lo, hi = agg[m]["ic95"]
        assert lo <= e <= hi, (m, e, lo, hi)


# ---------------------------------------------------------------- 2. siempre_none

def test_malformadas_por_tipo(consultas, tmp_path):
    """None devueltos, ausentes y NaN se cuentan por separado; los ficheros antiguos dan null."""
    def mixto(consulta, candidatos, vista):
        c = sorted(candidatos)
        return {c[0]: None, c[1]: None, c[2]: float("nan"), **{x: 0.5 for x in c[3:-4]}}  # 4 ausentes

    res = ev.evaluar("mixto", "aleatorio", "frio", 1, solo_desarrollo=True, scorer=mixto,
                     dir_resultados=tmp_path)
    m = res["malformadas"]
    assert (m["none"], m["nan"], m["ausentes"]) == (2 * 200, 200, 4 * 200)
    assert m["consultas_con_alguna"] == 200
    assert m["tasa_none"] == 400 / res["n_puntuaciones"]
    assert res["tasa_none"] == 7 * 200 / res["n_puntuaciones"]  # todo acaba como None
    linea = json.loads((tmp_path / "puntuaciones_mixto_aleatorio_frio_s1_dev.jsonl").open().readline())
    assert linea["malformadas"] == {"none": 2, "ausentes": 4, "nan": 1}

    # Fichero anterior al recuento (sin el campo): los totales quedan en null
    ruta = tmp_path / "antiguo.jsonl"
    ruta.write_text("".join(json.dumps({k: v for k, v in json.loads(l).items() if k != "malformadas"}) + "\n"
                            for l in (tmp_path / "puntuaciones_mixto_aleatorio_frio_s1_dev.jsonl").open()))
    cs = [c for c in consultas["frio"].values() if c["es_desarrollo"]]
    viejo = ev.calcular_metricas(ruta, cs, [precision_en(1)])
    assert all(v is None for v in viejo["malformadas"].values())


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_agregado_sin_desarrollo(resultados, consultas, escenario, tmp_path):
    """El agregado de reporte usa solo las 1800 consultas que no son de desarrollo."""
    res = resultados[("aleatorio", escenario)]
    sin_dev = [k for k, q in enumerate(res["idx"]) if not consultas[escenario][q]["es_desarrollo"]]
    assert res["n_consultas_sin_desarrollo"] == len(sin_dev) == 1800
    for tarea in ("completa", "implicita", "implicita_solo_ancla"):
        v = res["por_consulta"][tarea]["mrr"]
        assert res["agregado_sin_desarrollo"][tarea]["mrr"]["media"] == pytest.approx(
            sum(v[k] for k in sin_dev) / 1800)
        assert res["agregado_sin_desarrollo"][tarea]["mrr"]["n"] == 1800
        assert res["agregado"][tarea]["mrr"]["n"] == 2000
    # Con --desarrollo no hay consultas de reporte
    dev = ev.evaluar("aleatorio", "aleatorio", escenario, 1, solo_desarrollo=True, dir_resultados=tmp_path)
    assert dev["n_consultas_sin_desarrollo"] == 0 and dev["agregado_sin_desarrollo"] is None


def test_estructurales_caso_a_mano():
    """Vecinos comunes y Adamic-Adar sobre la vista no dirigida, en un grafo pequeño.

    Enlaces: 1->2, 3->1, 1->4 (oculto, con su recíproco 4->1 también oculto),
    2->5, 5->3, 6->3, 6->2, 5->7, 3->7, 7->8.
    N(1) = {2, 3} (el 4 no se ve). Candidatos: 5, 6, 7, 4.
    """
    import math
    from arnes.vista import GrafoVisible
    from pipeline.estructurales import _puntuar

    ady = {1: {2, 4}, 2: {5}, 3: {1, 7}, 4: {1}, 5: {3, 7}, 6: {3, 2}, 7: {8}, 8: set()}
    nd = {u: set(v) for u, v in ady.items()}
    for u, vs in ady.items():
        for v in vs:
            nd[v].add(u)
    g = GrafoVisible({u: frozenset(v) for u, v in ady.items()}, [(1, 4), (4, 1)],
                     {u: frozenset(v) for u, v in nd.items()})

    class VistaFalsa:
        def grafo(self, nombre):
            return g

    assert g.vecinos_no_dirigidos(1) == {2, 3}
    assert g.vecinos_no_dirigidos(4) == set()
    cn = _puntuar(VistaFalsa(), 1, [5, 6, 7, 4], "B", "vecinos_comunes")
    assert cn == {5: 2.0, 6: 2.0, 7: 1.0, 4: 0.0}  # N(5)={2,3,7}, N(6)={2,3}, N(7)={3,5,8}
    aa = _puntuar(VistaFalsa(), 1, [5, 6, 7, 4], "B", "adamic_adar")
    # |N(2)| = |{1, 5, 6}| = 3 ; |N(3)| = |{1, 5, 6, 7}| = 4
    assert aa[5] == pytest.approx(1 / math.log(3) + 1 / math.log(4))
    assert aa[6] == pytest.approx(1 / math.log(3) + 1 / math.log(4))
    assert aa[7] == pytest.approx(1 / math.log(4))
    assert aa[4] == 0.0


def test_none_va_al_final():
    p = {1: 0.2, 2: None, 3: 0.9, 4: float("nan"), 5: -3.0}
    r = ordenar(p, idx_consulta=7)
    assert r[:3] == [3, 1, 5] and set(r[3:]) == {2, 4}


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_siempre_none_no_rompe(resultados, consultas, escenario, tmp_path):
    res = resultados[("siempre_none", escenario)]
    assert res["tasa_none"] == 1.0
    assert res["n_consultas"] == 2000
    for tarea in ("completa", "implicita", "implicita_solo_ancla"):
        for v in res["agregado"][tarea].values():
            assert v["media"] is not None
        assert res["curva_pr"][tarea] == []
    # Peor rango posible: todos los candidatos empatan en el grupo final
    c = next(iter(consultas[escenario].values()))
    punt = {x: None for x in cargar_candidatos("aleatorio", escenario, 1)[c["idx"]]}
    for ranking, _ in rankings_por_tarea(c, punt).values():
        assert all(punt[x] is None for x in ranking)


# ---------------------------------------------------------------- 3. oráculo

@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_oraculo_perfecto(resultados, escenario):
    res = resultados[("oraculo", escenario)]
    for tarea in ("completa", "implicita", "implicita_solo_ancla"):
        assert all(v == 1.0 for v in res["por_consulta"][tarea]["precision@1"])
        assert all(v == 1.0 for v in res["por_consulta"][tarea]["mrr"])


# ---------------------------------------------------------------- 4. caso a mano

def test_caso_calculado_a_mano():
    consulta = {"idx": 1, "referencia": [11, 12], "referencia_implicita": [12],
                "explicitos": [11, 20, 14], "explicitos_solo_ancla": [20], "neutros_plantilla": [30]}
    # 11 y 12 empatan (0,5) y los dos son relevantes; 13 no se pudo puntuar; 30 es neutro
    punt = {10: 0.9, 11: 0.5, 12: 0.5, 13: None, 14: 0.2, 30: 5.0}
    tareas = rankings_por_tarea(consulta, punt)

    ranking, rel = tareas["completa"]
    assert ranking[0] == 10 and set(ranking[1:3]) == {11, 12} and ranking[3:] == [14, 13]
    assert precision_en(1)(ranking, rel) == 0.0
    assert precision_en(5)(ranking, rel) == 2 / 5
    assert precision_normalizada_en(1)(ranking, rel) == 0.0
    assert precision_normalizada_en(5)(ranking, rel) == 1.0  # 2 / min(5, 2)
    assert MRR(ranking, rel) == 0.5
    assert hits_en(1)(ranking, rel) == 0.0 and hits_en(5)(ranking, rel) == 1.0
    assert precision_en(3)(ranking, rel) == 2 / 3  # métrica inyectada distinta de las de por defecto

    ranking, rel = tareas["implicita"]  # sin 11 ni 14 (explícitos) ni 30 (neutro)
    assert ranking == [10, 12, 13]
    assert precision_en(1)(ranking, rel) == 0.0
    assert precision_en(5)(ranking, rel) == 1 / 5
    assert precision_normalizada_en(5)(ranking, rel) == 1.0  # 1 / min(5, 1)
    assert MRR(ranking, rel) == 0.5
    assert hits_en(5)(ranking, rel) == 1.0

    # precision_norm@k normaliza por |R| completo, aunque haya relevantes fuera de la lista (T1.8)
    assert precision_normalizada_en(5)([10, 12, 14], {12, 99}) == 1 / 2
    assert precision_normalizada_en(5)([10, 14], {99}) == 0.0
    assert MRR([10, 14], {99}) == 0.0

    # Solo ancla: solo 20 es explícito; 11 y 12 vuelven a ser relevantes implícitos
    ranking, rel = tareas["implicita_solo_ancla"]
    assert ranking == tareas["completa"][0] and rel == {11, 12}
    assert MRR(ranking, rel) == 0.5

    # El desempate es fijo: mismo orden en las dos tareas y en cada ejecución
    assert rankings_por_tarea(consulta, punt)["completa"][0] == tareas["completa"][0]

    # Curva PR: umbrales 0,9 -> (0/1, 0/2), 0,5 -> (2/3, 2/2), 0,2 -> (2/4, 2/2); el None no se predice
    pares = [(0.9, False), (0.5, True), (0.5, True), (0.2, False), (None, False)]
    assert curva_pr(pares) == [{"umbral": 0.9, "precision": 0.0, "recall": 0.0},
                               {"umbral": 0.5, "precision": 2 / 3, "recall": 1.0},
                               {"umbral": 0.2, "precision": 0.5, "recall": 1.0}]


def test_metricas_inyectadas(tmp_path):
    solo_p3 = [precision_en(3)]
    res = ev.evaluar("aleatorio", "aleatorio", "frio", 1, solo_desarrollo=True,
                     metricas=solo_p3, dir_resultados=tmp_path)
    assert res["metricas"] == ["precision@3"]
    assert set(res["agregado"]["implicita"]) == {"precision@3"}


# ---------------------------------------------------------------- 5. filtros de tarea

def test_filtros_de_tarea(consultas, tmp_path):
    """Implícita sin explícitos; neutros fuera de las dos tareas y fuera de lo que ve el scorer."""
    recibidos = {}

    def espia(consulta, candidatos, vista):
        recibidos[consulta["idx"]] = list(candidatos)
        return {x: float(x) for x in candidatos}

    # Candidatos forzados que incluyen neutros y explícitos
    cs = [c for c in consultas["caliente"].values() if c["es_desarrollo"]]
    base = cargar_candidatos("aleatorio", "caliente", 1)
    cands = {c["idx"]: base[c["idx"]] + c["neutros_plantilla"][:5] + c["explicitos"][:5] for c in cs}
    assert any(c["neutros_plantilla"] for c in cs) and any(c["explicitos"] for c in cs)
    ev.evaluar("espia", "aleatorio", "caliente", 1, solo_desarrollo=True, scorer=espia,
               dir_resultados=tmp_path, candidatos=cands)
    ruta = tmp_path / "puntuaciones_espia_aleatorio_caliente_s1_dev.jsonl"
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        d = json.loads(linea)
        c = consultas["caliente"][d["idx"]]
        assert not set(recibidos[d["idx"]]) & set(c["neutros_plantilla"])
        tareas = rankings_por_tarea(c, {x: p for x, p in d["puntuaciones"]})
        assert not set(tareas["implicita"][0]) & set(c["explicitos"])
        assert not set(tareas["implicita_solo_ancla"][0]) & set(c["explicitos_solo_ancla"])
        for ranking, _ in tareas.values():
            assert not set(ranking) & set(c["neutros_plantilla"])


# ---------------------------------------------------------------- 6. determinismo

def test_determinismo(tmp_path):
    for k in (1, 2):
        for e in ESCENARIOS:
            for s in ("aleatorio", "oraculo", "siempre_none"):
                ev.evaluar(s, "aleatorio", e, 2, solo_desarrollo=True, dir_resultados=tmp_path / str(k))
    nombres = sorted(p.name for p in (tmp_path / "1").iterdir())
    assert len(nombres) == 12
    for n in nombres:
        assert filecmp.cmp(tmp_path / "1" / n, tmp_path / "2" / n, shallow=False), n


# ---------------------------------------------------------------- 7. registro de scorers

def test_scorers_registrados_y_nuevo(tmp_path):
    assert {"aleatorio", "siempre_none", "oraculo"} <= set(SCORERS)

    @registrar("prueba_constante")
    def constante(consulta, candidatos, vista):
        return {x: 1.0 for x in candidatos}

    try:
        res = ev.evaluar("prueba_constante", "aleatorio", "frio", 1, solo_desarrollo=True,
                         dir_resultados=tmp_path)
        assert res["n_consultas"] == 200 and res["tasa_none"] == 0.0
        with pytest.raises(ValueError):
            registrar("prueba_constante")(constante)
    finally:
        del SCORERS["prueba_constante"]


def test_scorer_que_usa_la_vista(tmp_path):
    """Un scorer de una función que usa la vista (grafo y texto) sin tocar otras capas."""
    @registrar("prueba_vista")
    def por_grado(consulta, candidatos, vista):
        g = vista.grafo("B")
        return {x: len(g.vecinos(x)) + len(vista.texto(x)) * 1e-6 for x in candidatos}

    try:
        res = ev.evaluar("prueba_vista", "aleatorio", "caliente", 1, solo_desarrollo=True,
                         dir_resultados=tmp_path)
        assert res["n_consultas"] == 200
    finally:
        del SCORERS["prueba_vista"]


# ---------------------------------------------------------------- 8. decorador de coste

def test_decorador_no_cambia_metricas(tmp_path):
    ruta_coste = tmp_path / "coste.jsonl"

    def con_llamadas(consulta, candidatos, vista):
        anotar(llamadas=2, tokens_in=10, tokens_out=3, num_ctx=2048, modelo="falso")
        return SCORERS["aleatorio"](consulta, candidatos, vista)

    envuelto = con_coste(ruta_coste, sistema="aleatorio", modo="aleatorio", semilla=1)(con_llamadas)
    ev.evaluar("aleatorio", "aleatorio", "frio", 1, solo_desarrollo=True, scorer=envuelto,
               dir_resultados=tmp_path / "con")
    ev.evaluar("aleatorio", "aleatorio", "frio", 1, solo_desarrollo=True, scorer=con_llamadas,
               dir_resultados=tmp_path / "sin")
    n = "puntuaciones_aleatorio_aleatorio_frio_s1_dev.jsonl"
    assert filecmp.cmp(tmp_path / "con" / n, tmp_path / "sin" / n, shallow=False)
    # Las métricas solo difieren en el id_ejecucion, que únicamente tiene la versión con decorador
    n = "metricas_aleatorio_aleatorio_frio_s1_dev.json"
    con, sin = (json.loads((tmp_path / d / n).read_text(encoding="utf-8")) for d in ("con", "sin"))
    assert con.pop("id_ejecucion") == envuelto.id_ejecucion and sin.pop("id_ejecucion") is None
    assert con == sin
    filas = [json.loads(l) for l in ruta_coste.read_text(encoding="utf-8").splitlines()]
    assert len(filas) == 200
    campos = {"ts", "id_ejecucion", "sistema", "modo", "escenario", "semilla", "consulta", "latencia_ms",
              "llamadas", "tokens_in", "tokens_out", "num_ctx", "modelo", "nivel"}  # nivel: T2.6
    assert set(filas[0]) == campos  # sin hw ni versiones en cada fila
    assert filas[0]["nivel"] is None
    assert filas[0]["llamadas"] == 2 and filas[0]["tokens_in"] == 10 and filas[0]["num_ctx"] == 2048
    assert filas[0]["escenario"] == "frio"
    assert {f["id_ejecucion"] for f in filas} == {envuelto.id_ejecucion}
    ejecucion = json.loads((tmp_path / "ejecuciones" / f"{envuelto.id_ejecucion}.json").read_text(encoding="utf-8"))
    assert ejecucion["versiones"]["python"] and "gpu" in ejecucion["hw"]
    assert ejecucion["escenario"] == "frio" and ejecucion["config"] is None
    assert set(ejecucion["git"]) == {"hash", "hay_cambios", "cambios_sin_commitear"}
    assert ejecucion["entorno_coincide"] in (True, False, None)
    assert len(list((tmp_path / "ejecuciones").iterdir())) == 1
    # Sin decorador, anotar no hace nada (no falla)
    anotar(llamadas=1)


def test_comparar_entorno(tmp_path):
    from arnes.coste import comparar_entorno

    ruta = tmp_path / "entorno_congelado.json"
    assert comparar_entorno({}, ruta)[0] is None  # sin fichero no se puede comprobar
    ruta.write_text(json.dumps({"ollama": {"servidor": "1.0"}, "nvidia": {"driver": "600.1"}}))
    assert comparar_entorno({"ollama_servidor": "1.0", "driver_nvidia": "600.1"}, ruta) == (True, [])
    coincide, dif = comparar_entorno({"ollama_servidor": "1.1", "driver_nvidia": "600.1"}, ruta)
    assert coincide is False and [d["campo"] for d in dif] == ["ollama_servidor"]
    # Ollama apagado: no difiere, pero tampoco se puede confirmar
    assert comparar_entorno({"ollama_servidor": None, "driver_nvidia": "600.1"}, ruta)[0] is None


def test_agregar_descarta_calentamiento():
    filas = [{"ts": f"2026-01-01T00:00:{k:02d}", "sistema": "x", "modo": "m", "escenario": "e",
              "semilla": s, "id_ejecucion": f"ej{s}", "latencia_ms": 1000.0 if k < 3 else 1.0, "llamadas": 0, "tokens_in": 0,
              "tokens_out": 0} for s in (1, 2) for k in range(10)]
    (t,) = agregar(filas)
    assert t["n_consultas"] == 14 and t["descartadas"] == 6
    assert t["latencia_ms"] == {"p50": 1.0, "p95": 1.0}


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="sin nvidia-smi")
def test_medir_vram_comando():
    from arnes.medir_vram import medir_comando
    r = medir_comando([sys.executable, "-c", "import time; time.sleep(1)"])
    assert r["codigo_salida"] == 0 and r["n_muestras"] >= 2
    assert r["pico_mib"] >= r["base_mib"] > 0


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_y_diferencia_pareada(resultados):
    a, o = resultados[("aleatorio", "frio")], resultados[("oraculo", "frio")]
    d = ev.diferencia_pareada(o, a, "implicita", "mrr")
    assert d["n"] == 2000 and d["ic95"][0] > 0
    ic = ev.ic_bootstrap([0.0, 1.0] * 50)
    assert ic["media"] == 0.5 and ic["ic95"][0] < 0.5 < ic["ic95"][1]
