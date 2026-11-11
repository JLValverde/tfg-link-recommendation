"""Tests de fuga de las consultas (T0.8).

Comprueban, sobre las consultas generadas en datos/procesados/, que lo que debe
estar oculto no es visible ni en el grafo ni en el texto, y que la generación es
determinista. Necesitan los ficheros de datos/procesados/ (no versionados).
"""

import filecmp
import json
import math
import random
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "datos"))

from arnes.vista import cargar_adyacencia, cargar_textos, grafo_visible, texto_visible  # noqa: E402

DIR_PROCESADOS = RAIZ / "datos" / "procesados"
RUTA_GRAFO = DIR_PROCESADOS / "grafo.json"
ESCENARIOS = ("caliente", "frio")
N_FUGA_INVERSA = 50
SEMILLA_FUGA_INVERSA = 45

if not RUTA_GRAFO.exists():
    pytest.skip("faltan los datos procesados (grafo.json)", allow_module_level=True)


def cargar_consultas(escenario):
    ruta = DIR_PROCESADOS / f"consultas_{escenario}.jsonl"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def consultas():
    return {e: cargar_consultas(e) for e in ESCENARIOS}


@pytest.fixture(scope="module")
def nodos():
    g = json.loads(RUTA_GRAFO.read_text(encoding="utf-8"))
    return {int(i): d for i, d in g["nodos"].items()}


@pytest.fixture(scope="module")
def predecesores():
    pred = {}
    for grafo in ("A", "B"):
        p = {}
        for u, vec in cargar_adyacencia(grafo).items():
            for v in vec:
                p.setdefault(v, set()).add(u)
        pred[grafo] = p
    return pred


# ---------------------------------------------------------------- estructura

def test_grafo_sin_indices_invalidos(nodos):
    validos = set(nodos)
    for i, d in nodos.items():
        for clave in ("A", "B", "explicitos", "explicitos_solo_ancla"):
            assert set(d[clave]) <= validos, (i, clave)
            assert i not in d[clave], (i, clave)
        assert set(d["B"]) <= set(d["A"])


def test_mismas_consultas_en_los_dos_escenarios(consultas):
    idx = {e: [c["idx"] for c in consultas[e]] for e in ESCENARIOS}
    assert idx["caliente"] == idx["frio"] == sorted(idx["caliente"])
    assert len(idx["caliente"]) == 2000
    assert sum(c["es_desarrollo"] for c in consultas["caliente"]) == 200
    assert ([c["es_desarrollo"] for c in consultas["caliente"]]
            == [c["es_desarrollo"] for c in consultas["frio"]])


# ---------------------------------------------------------------- grafo visible

@pytest.mark.parametrize("grafo", ["B", "A"])
def test_frio_consulta_aislada(consultas, predecesores, grafo):
    """Frío: ningún enlace con la consulta en algún extremo es visible."""
    for c in consultas["frio"]:
        q = c["idx"]
        vista = grafo_visible(c, grafo)
        assert not vista.vecinos(q), q
        for u in predecesores[grafo].get(q, ()):
            assert q not in vista.vecinos(u), (u, q)
            assert not vista.contiene(u, q), (u, q)


@pytest.mark.parametrize("grafo", ["B", "A"])
def test_caliente_ocultos_invisibles(consultas, grafo):
    """Caliente: ningún enlace oculto es visible, ni su recíproco."""
    for c in consultas["caliente"]:
        vista = grafo_visible(c, grafo)
        for u, v in c[f"ocultos_{grafo}"]:
            assert not vista.contiene(u, v), (c["idx"], u, v)
            assert not vista.contiene(v, u), (c["idx"], v, u)
            assert v not in vista.vecinos(u)


def test_caliente_ocultos_incluyen_reciprocos(consultas, nodos):
    for c in consultas["caliente"]:
        q = c["idx"]
        for grafo in ("A", "B"):
            ocultos = {tuple(p) for p in c[f"ocultos_{grafo}"]}
            for j in c["referencia"]:
                assert (q, j) in ocultos
                if q in nodos[j][grafo]:
                    assert (j, q) in ocultos


# ---------------------------------------------------------------- referencias

@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_referencias_coherentes(consultas, nodos, escenario):
    for c in consultas[escenario]:
        B = set(nodos[c["idx"]]["B"])
        ref, ref_impl = set(c["referencia"]), set(c["referencia_implicita"])
        assert ref <= B
        assert ref_impl <= ref
        assert not ref_impl & set(c["explicitos"])
        assert not set(c["neutros_plantilla"]) & ref
        assert len(ref_impl) >= 1


def test_caliente_ocultacion_estratificada(consultas):
    """Caliente: se ocultan exactamente ceil(I/2) implícitos y floor(E/2) explícitos
    (ADR "En caliente se oculta al azar la mitad de cada tipo de enlace por separado").
    I y E salen del registro de frío del mismo idx, cuya referencia es B entera.
    Que quede al menos un implícito oculto ya lo comprueba test_referencias_coherentes."""
    frio = {c["idx"]: c for c in consultas["frio"]}
    for c in consultas["caliente"]:
        f = frio[c["idx"]]
        impl_B = set(f["referencia_implicita"])
        expl_B = set(f["referencia"]) - impl_B
        ocultos_impl = set(c["referencia_implicita"])
        ocultos_expl = set(c["referencia"]) - ocultos_impl
        assert ocultos_impl <= impl_B and ocultos_expl <= expl_B, c["idx"]
        assert len(ocultos_impl) == math.ceil(len(impl_B) / 2), c["idx"]
        assert len(ocultos_expl) == len(expl_B) // 2, c["idx"]


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_candidatos_recuperador_sin_sucesores_visibles(consultas, predecesores, escenario):
    """Modo recuperador: nunca son candidatos la consulta ni sus sucesores visibles en B;
    los predecesores visibles sí pueden serlo (Protocolo de evaluación v2.0, §1)."""
    from arnes.candidatos import cargar_candidatos, ruta_candidatos

    if not ruta_candidatos("recuperador", escenario).exists():
        pytest.skip("faltan las listas del modo recuperador")
    listas = cargar_candidatos("recuperador", escenario)
    con_predecesores = 0
    for c in consultas[escenario]:
        q, lista = c["idx"], set(listas[c["idx"]])
        vista = grafo_visible(c, "B")
        assert q not in lista
        assert not lista & set(vista.sucesores(q)), q
        pred_visibles = {u for u in predecesores["B"].get(q, ()) if vista.contiene(u, q)}
        # Un predecesor visible nunca es un enlace de referencia: si q -> u estuviera
        # oculto, u -> q se habría ocultado también como recíproco
        assert not pred_visibles & set(c["referencia"]), q
        con_predecesores += bool(lista & pred_visibles)
    if escenario == "caliente":
        assert con_predecesores > 0, "ninguna lista trae predecesores visibles: ¿se están excluyendo?"
    else:
        assert con_predecesores == 0  # en frío no hay ningún enlace visible de la consulta


@pytest.mark.parametrize("grafo", ["B", "A"])
def test_vista_no_dirigida_sin_fugas(consultas, grafo):
    """La vista no dirigida (heurísticas estructurales) no deja ver ningún enlace oculto."""
    for c in consultas["frio"]:
        assert not grafo_visible(c, grafo).vecinos_no_dirigidos(c["idx"]), c["idx"]
    for c in consultas["caliente"]:
        q, vista = c["idx"], grafo_visible(c, grafo)
        nq = vista.vecinos_no_dirigidos(q)
        assert not nq & set(c["referencia"]), q
        for u, v in c[f"ocultos_{grafo}"]:
            assert v not in vista.vecinos_no_dirigidos(u) and u not in vista.vecinos_no_dirigidos(v)


def test_vista_no_dirigida_por_definicion(consultas):
    """En una muestra, N(u) coincide con {v : u -> v o v -> u visible}, calculado a mano."""
    A_nd = {}
    for grafo in ("B", "A"):
        for u, vec in cargar_adyacencia(grafo).items():
            for v in vec:
                A_nd.setdefault((grafo, u), set()).add(v)
                A_nd.setdefault((grafo, v), set()).add(u)
    for esc in ESCENARIOS:
        for c in random.Random(SEMILLA_FUGA_INVERSA).sample(consultas[esc], 30):
            for grafo in ("B", "A"):
                vista = grafo_visible(c, grafo)
                nodos = {c["idx"]} | set(A_nd.get((grafo, c["idx"]), ()))
                for u in list(nodos)[:40]:
                    esperado = {v for v in A_nd.get((grafo, u), ())
                                if vista.contiene(u, v) or vista.contiene(v, u)}
                    assert vista.vecinos_no_dirigidos(u) == esperado, (esc, grafo, c["idx"], u)


def test_caliente_queda_algun_enlace_visible(consultas):
    for c in consultas["caliente"]:
        assert grafo_visible(c, "B").vecinos(c["idx"]), c["idx"]


# ---------------------------------------------------------------- fuga inversa

@pytest.fixture(scope="module")
def anclas_hacia():
    """destino -> [(doc, ancla)] de todas las anclas de introducción."""
    indice = {}
    for doc, (_, anclas) in cargar_textos().items():
        for a in anclas:
            for q in {a["destino_idx"], a["destino_idx_redir"]} - {None}:
                indice.setdefault(q, []).append((doc, a))
    return indice


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_fuga_inversa(consultas, anclas_hacia, escenario):
    """Ningún texto visible de otro documento conserva un ancla a la consulta con el enlace oculto."""
    textos = cargar_textos()
    muestra = random.Random(SEMILLA_FUGA_INVERSA).sample(consultas[escenario], N_FUGA_INVERSA)
    comprobadas = 0
    for c in muestra:
        q = c["idx"]
        ocultos = {tuple(p) for p in c["ocultos_B"]}
        por_doc = {}
        for doc, a in anclas_hacia.get(q, []):
            if doc != q:
                por_doc.setdefault(doc, []).append(a)
        for doc, anclas in por_doc.items():
            original = textos[doc][0]
            visible = texto_visible(doc, c)
            # En frío no puede quedar ninguna ancla a la consulta; en caliente, solo
            # se tapan las de enlaces ocultos
            if escenario == "caliente" and (doc, q) not in ocultos:
                assert visible == original
                continue
            comprobadas += 1
            # Cada ancla a q pasa a ser "[...]" y el resto del texto queda intacto
            esperado, previo = [], 0
            for a in sorted(anclas, key=lambda a: a["inicio"]):
                assert original[a["inicio"]:a["fin"]] == a["texto"]
                esperado += [original[previo:a["inicio"]], "[...]"]
                previo = a["fin"]
            assert visible == "".join(esperado + [original[previo:]]), (q, doc)
            assert visible.count("[...]") >= len(anclas)
        assert texto_visible(q, c) == textos[q][0]
    assert comprobadas > 0, "la muestra no ha comprobado ningún ancla oculta"


# ---------------------------------------------------------------- determinismo

def test_generacion_determinista(tmp_path):
    import generar_consultas

    for k in (1, 2):
        d = tmp_path / str(k)
        d.mkdir()
        generar_consultas.main(dir_salida=d)
    for nombre in ("consultas_caliente.jsonl", "consultas_frio.jsonl"):
        assert filecmp.cmp(tmp_path / "1" / nombre, tmp_path / "2" / nombre, shallow=False)
        assert filecmp.cmp(tmp_path / "1" / nombre, DIR_PROCESADOS / nombre, shallow=False)
