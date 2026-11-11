"""Tests unitarios de C1 (T2.2): agregación de variantes, los tres casos, recorte, scorer
y prompt congelado.

No necesitan Ollama: el modelo se sustituye por respuestas fijas.
"""

import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from arnes.scorer import SCORERS
from pipeline import reordenacion as ro

# SHA-256 del prompt congelado (variante 2, T2.2b): si cambia, el prompt se ha tocado
HASH_PROMPT_CONGELADO = "34b9e892d5facf186b3fab2b1fd4b6d7c9caaede4ab41a3d6d24c9b1f73a658c"
PLANTILLA_V1 = ("Article A:\nTitle: {titulo_consulta}\n{texto_consulta}\n\n"
                "Article B:\nTitle: {titulo_candidato}\n{texto_candidato}\n\n"
                "Should article A link to article B? Answer Yes or No.")


def test_prompt_congelado_no_ha_cambiado():
    assert hashlib.sha256(ro.PLANTILLA.encode("utf-8")).hexdigest() == HASH_PROMPT_CONGELADO
    assert ro.PLANTILLA.endswith("Is article B closely related to the topic of article A? Answer Yes or No.")
    for campo in ("titulo_consulta", "texto_consulta", "titulo_candidato", "texto_candidato"):
        assert ro.PLANTILLA.count("{" + campo + "}") == 1


def lse(*v):
    return math.log(sum(math.exp(x) for x in v))


def test_normalizar_variantes():
    assert {ro.normalizar(t) for t in ("Yes", " yes", "YES", "_yes", ".Yes", ",Yes", "Yes!")} == {"yes"}
    assert {ro.normalizar(t) for t in ("No", " No", "NO", "-No", "_No")} == {"no"}
    assert ro.normalizar("Not") == "not" and ro.normalizar(" ") == ""


def test_caso_ambos_agrega_variantes_por_logsumexp():
    top = [("Yes", -0.1), (" yes", -3.0), ("No", -2.5), ("NO", -4.0), ("Maybe", -6.0)]
    c = ro.confianza(top)
    assert c["caso"] == "ambos"
    assert c["lse_si"] == pytest.approx(lse(-0.1, -3.0)) and c["lse_no"] == pytest.approx(lse(-2.5, -4.0))
    assert c["conf_cota"] == c["conf_none"] == pytest.approx(lse(-0.1, -3.0) - lse(-2.5, -4.0))


def test_caso_falta_uno():
    # Falta el "yes": toma como logprob el menor del top (-9), solo con P-cota
    c = ro.confianza([("No", -0.01), ("-No", -5.0), ("The", -9.0)])
    assert c["caso"] == "falta_uno" and c["lse_si"] is None
    assert c["conf_cota"] == pytest.approx(-9.0 - lse(-0.01, -5.0))
    assert c["conf_none"] is None
    # Falta el "no"
    c = ro.confianza([("Yes", -0.2), ("Sure", -2.0), (" yes", -7.5)])
    assert c["caso"] == "falta_uno" and c["lse_no"] is None
    assert c["conf_cota"] == pytest.approx(lse(-0.2, -7.5) - (-7.5))
    assert c["conf_none"] is None


def test_caso_faltan_ambos():
    c = ro.confianza([("The", -0.5), ("I", -1.0)])
    assert c["caso"] == "faltan_ambos" and c["conf_cota"] is None and c["conf_none"] is None


def test_recortar_por_palabras_sin_tocar_el_formato():
    texto = "uno dos\ntres  cuatro\n\ncinco"
    assert ro.recortar(texto, 3) == ("uno dos\ntres", True)
    assert ro.recortar(texto, 5) == (texto, False)
    assert ro.recortar(texto, 10) == (texto, False)
    assert ro.recortar("a b c   ", 3) == ("a b c", False)  # solo queda espacio: no es recorte


class VistaFalsa:
    def texto(self, doc):
        return {1: "consulta " * 500, 2: "corto", 3: "largo " * 300, 4: "otro"}[doc].strip()

    def titulo(self, doc):
        return f"T{doc}"


def respuesta(top, prompt_eval_count=100):
    lp = SimpleNamespace(token=top[0][0], logprob=top[0][1],
                         top_logprobs=[SimpleNamespace(token=t, logprob=l) for t, l in top])
    return SimpleNamespace(logprobs=[lp], prompt_eval_count=prompt_eval_count, eval_count=1,
                           prompt_eval_duration=1_000_000, total_duration=2_000_000)


def test_scorer_c1_con_respuestas_fijas(monkeypatch, tmp_path):
    prompts = []
    respuestas = {2: respuesta([("Yes", -0.1), ("No", -2.4)]),
                  3: respuesta([("No", -0.01), ("The", -9.0)]),
                  4: None}  # error de la llamada

    def llamar(prompt, **kw):
        prompts.append(prompt)
        cand = int(prompt.split("Article B:\nTitle: T")[1].split("\n")[0])
        if respuestas[cand] is None:
            raise ConnectionError("caída")
        return respuestas[cand]

    monkeypatch.setattr(ro, "llamar", llamar)
    monkeypatch.setattr(ro, "PLANTILLA", PLANTILLA_V1)  # la mecánica no depende del texto
    monkeypatch.setattr(ro, "RUTA_DIAGNOSTICO", tmp_path / "diag.jsonl")
    ro.reiniciar_contadores()
    p = SCORERS["c1"]({"idx": 1, "escenario": "frio"}, [2, 3, 4], VistaFalsa())
    assert p[2] == pytest.approx(-0.1 + 2.4)
    assert p[3] == pytest.approx(-9.0 + 0.01)  # P-cota
    assert p[4] is None
    assert all(pr.startswith("Article A:\nTitle: T1\n") for pr in prompts)
    # Consulta recortada a 400 palabras, más "Article A:" y "Title: T1" (4 palabras)
    assert len(prompts[0].split("Article B:")[0].split()) == 400 + 4
    assert ro.CONTADORES["ambos"] == 1 and ro.CONTADORES["falta_uno"] == 1 and ro.CONTADORES["error"] == 1
    assert ro.CONTADORES["consultas_recortadas"] == 1 and ro.CONTADORES["candidatos_recortados"] == 1
    filas = (tmp_path / "diag.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(filas) == 3


# ---------------------------------------------------------------- C2 (T2.4)

class GrafoFalso:
    # 1-2, 1-3, 1-4, 2-4, 2-5, 4-5 (vista no dirigida)
    N = {1: {2, 3, 4}, 2: {1, 4, 5}, 3: {1}, 4: {1, 2, 5}, 5: {2, 4}}

    def vecinos_no_dirigidos(self, u):
        return frozenset(self.N[u])


class VistaGrafoFalsa(VistaFalsa):
    def grafo(self, nombre="B"):
        assert nombre == "B"
        return GrafoFalso()

    def titulo(self, doc):
        return {1: "Uno", 2: "Beta", 3: "Alpha", 4: "Aardvark", 5: "Cinco"}[doc]

    def texto(self, doc):
        return "texto corto" if doc == 5 else super().texto(doc)


def test_listas_de_vecinos_y_comunes():
    from pipeline.contexto import Contexto
    ctx = Contexto(VistaGrafoFalsa())
    # Grado descendente (2 y 4 tienen 3; 3 tiene 1); empate por título
    assert ctx.lista(1) == "Aardvark; Beta; Alpha"
    # Comunes de 1 y 5: {2, 4}, grado ascendente (empatan a 3) y título
    assert ctx.comunes(1, 5) == "Aardvark; Beta"
    assert ctx.comunes(3, 5) == "none"
    assert Contexto(VistaGrafoFalsa(), max_vecinos=2).lista(1) == "Aardvark; Beta"


def test_comunes_por_grado_ascendente():
    from pipeline.contexto import Contexto
    ctx = Contexto(VistaGrafoFalsa())
    # Comunes de 2 y 4: {1, 5}; grado 2 (Cinco) antes que grado 3 (Uno)
    assert ctx.comunes(2, 4) == "Cinco; Uno"
    assert ctx.lista(5) == "Aardvark; Beta"


def variantes_c2():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experimentos"))
    import variantes_c2
    return variantes_c2


def test_variantes_c2_solo_anaden_lineas_a_c1():
    v = variantes_c2()
    for n, espec in v.VARIANTES.items():
        for esc, plantilla in espec["plantillas"].items():
            quitado = plantilla.replace(v.LINEA_A, "").replace(v.LINEA_B, "")
            quitado = quitado.replace("\n" + v.LINEA_COMUNES, "")
            assert quitado == ro.PLANTILLA, (n, esc)
        assert espec["plantillas"]["frio"] == v.VARIANTES[1]["plantillas"]["frio"]
        assert "{vecinos_consulta}" not in espec["plantillas"]["frio"]
    assert "{comunes}" not in v.VARIANTES[1]["plantillas"]["caliente"]
    assert "{vecinos_consulta}" not in v.VARIANTES[3]["plantillas"]["caliente"]


def test_scorer_c2_con_respuestas_fijas(monkeypatch, tmp_path):
    v = variantes_c2()
    prompts = []

    def llamar(prompt, **kw):
        prompts.append(prompt)
        return respuesta([("No", -0.1), ("Yes", -2.4)])

    monkeypatch.setattr(ro, "llamar", llamar)
    monkeypatch.setattr(ro, "C2", v.VARIANTES[2])
    monkeypatch.setattr(ro, "RUTA_DIAGNOSTICO", tmp_path / "diag.jsonl")
    ro.reiniciar_contadores()
    p = SCORERS["c2"]({"idx": 1, "escenario": "caliente"}, [5], VistaGrafoFalsa())
    assert p[5] == pytest.approx(-2.4 + 0.1)
    assert "Title: Uno\n" in prompts[0] and "Connected articles: Aardvark; Beta; Alpha\n" in prompts[0]
    assert "Connected articles: Aardvark; Beta\n\nArticles connected to both A and B: Aardvark; Beta\n\nIs" in prompts[0]
    fila = json.loads((tmp_path / "diag.jsonl").read_text(encoding="utf-8"))
    assert (fila["n_vecinos_consulta"], fila["n_vecinos_candidato"], fila["n_comunes"]) == (3, 2, 2)
    assert fila["hash_prompt"] == ro.hash_texto(prompts[0])[:16]
    # En frío no hay lista de A ni comunes
    ro.c2({"idx": 1, "escenario": "frio"}, [5], VistaGrafoFalsa())
    assert "Uno\n" in prompts[1] and prompts[1].count("Connected articles") == 1
    assert "both A and B" not in prompts[1]


# SHA-256 del contexto de C2 congelado (variante 1, T2.4): si cambia, se ha tocado
HASH_C2_CONGELADO = "05e5d4b230b8020061824793d8bcf10761b6e366b266069ce2c903a8764ff07c"


def test_contexto_c2_congelado_no_ha_cambiado():
    assert ro.C2 is not None and ro.C2["variante"] == 1
    assert ro.hash_texto(ro.RUTA_C2.read_bytes().decode("utf-8")) == HASH_C2_CONGELADO
    assert ro.hash_texto(ro.texto_c2(ro.C2)) == HASH_C2_CONGELADO
    # Es el prompt congelado de C1 más las líneas de vecinos
    v = variantes_c2()
    assert ro.C2 == v.especificacion(1, c1=ro.PLANTILLA)
    assert (ro.C2["grafo"], ro.C2["max_vecinos"], ro.C2["separador"], ro.C2["vacio"]) == ("B", 20, "; ", "none")
