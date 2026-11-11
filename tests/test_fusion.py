"""Test unitario de la fusión RRF de BM25 y Adamic-Adar (B) (T2.1), con un caso de mano;
también la fusión exploratoria de C1 y Adamic-Adar (B) (T2.8b).

Consulta 7, candidatos 10, 20, 30, 40. Claves de desempate del arnés
(arnes.rangos._desempate(7, c)): 10 -> 0,254; 30 -> 0,658; 40 -> 0,885; 20 -> 0,962.
En un empate va antes la clave menor.

- BM25: 10 y 20 empatan en cabeza (2,0), 30 = 1,0 y 40 = None (peor rango).
  Rangos: 10 -> 1, 20 -> 2, 30 -> 3, 40 -> 4.
- Adamic-Adar (B): 20 = 0,5 y 10, 30, 40 empatan a 0.
  Rangos: 20 -> 1, 10 -> 2, 30 -> 3, 40 -> 4.
"""

import pytest

from arnes.rangos import _desempate
from arnes.scorer import SCORERS
from pipeline import fusion

Q, CANDIDATOS = 7, [10, 20, 30, 40]
BM25 = {10: 2.0, 20: 2.0, 30: 1.0, 40: None}
AA = {10: 0.0, 20: 0.5, 30: 0.0, 40: 0.0}
RANGOS_BM25 = {10: 1, 20: 2, 30: 3, 40: 4}
RANGOS_AA = {20: 1, 10: 2, 30: 3, 40: 4}


def test_claves_de_desempate_del_caso():
    claves = {c: _desempate(Q, c) for c in CANDIDATOS}
    assert sorted(CANDIDATOS, key=claves.get) == [10, 30, 40, 20]


def test_rangos_con_empates_y_none():
    assert fusion.rangos(BM25, CANDIDATOS, Q) == RANGOS_BM25
    assert fusion.rangos(AA, CANDIDATOS, Q) == RANGOS_AA
    # Un candidato ausente de la salida cuenta como None: mismo rango que con None explícito
    assert fusion.rangos({c: p for c, p in BM25.items() if c != 40}, CANDIDATOS, Q) == RANGOS_BM25


def test_formula_rrf():
    esperado = {10: 1 / 61 + 1 / 62, 20: 1 / 62 + 1 / 61, 30: 1 / 63 + 1 / 63, 40: 1 / 64 + 1 / 64}
    obtenido = fusion.rrf([RANGOS_BM25, RANGOS_AA], CANDIDATOS)
    assert obtenido == pytest.approx(esperado, abs=0, rel=1e-15)
    assert fusion.K_RRF == 60


def test_scorer_registrado_con_componentes_de_mano(monkeypatch):
    monkeypatch.setitem(SCORERS, "bm25", lambda consulta, candidatos, vista: dict(BM25))
    monkeypatch.setitem(SCORERS, "adamic_adar_B", lambda consulta, candidatos, vista: dict(AA))
    monkeypatch.setattr(fusion, "NONE_POR_COMPONENTE", {c: 0 for c in fusion.COMPONENTES})
    p = SCORERS["rrf_bm25_aa_B"]({"idx": Q, "escenario": "caliente"}, CANDIDATOS, vista=None)
    assert p == fusion.rrf([RANGOS_BM25, RANGOS_AA], CANDIDATOS)
    assert fusion.NONE_POR_COMPONENTE == {"bm25": 1, "adamic_adar_B": 0}


def test_fusion_exploratoria_con_puntuaciones_guardadas_de_c1(monkeypatch):
    """T2.8b: con las puntuaciones de BM25 del caso haciendo de C1 guardado, la misma RRF."""
    monkeypatch.setitem(SCORERS, "adamic_adar_B", lambda consulta, candidatos, vista: dict(AA))
    publica = {"idx": Q, "escenario": "caliente"}
    scorer = fusion.rrf_c1_aa_B({Q: dict(BM25)})
    assert scorer(publica, CANDIDATOS, vista=None) == fusion.rrf([RANGOS_BM25, RANGOS_AA], CANDIDATOS)
    assert scorer.none_por_componente == {"c1": 1, "adamic_adar_B": 0}
    # Si las puntuaciones guardadas no son de la lista del arnés, falla en vez de rellenar con None
    with pytest.raises(ValueError):
        fusion.rrf_c1_aa_B({Q: {10: 1.0, 20: 0.5}})(publica, CANDIDATOS, vista=None)
