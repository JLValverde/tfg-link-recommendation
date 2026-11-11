"""Longitud en tokens del texto que codifica el recuperador (T2.8d). No cambia nada del pipeline.

Con el tokenizador de Qwen/Qwen3-Embedding-0.6B (el del SentenceTransformer, cargado en CPU,
con sus tokens especiales: lo que ve el modelo), la longitud de título + introducción
(recuperacion.texto_documento) de:
- cada documento de la colección (el texto original, el de los embeddings precalculados);
- las 1800 consultas de la muestra principal que no son de desarrollo, por escenario (su
  texto nunca se tapa: el original);
- sus K = 50 candidatos del modo recuperador, por pareja (consulta, candidato): con el texto
  visible para esa consulta si el candidato está en documentos_alterados() (es el que se
  recodifica), y el original si no.
Las listas se generaron sin max_seq_length (32768 por defecto); el pico de VRAM de T2.6 se
midió con 512. Cuenta cuántos superan 512 tokens, el máximo y los percentiles 50, 90 y 99.

Escribe experimentos/resultados/longitud_tokens.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/longitud_tokens.py
"""

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.candidatos import cargar_candidatos, cargar_consultas  # noqa: E402
from arnes.vista import Vista, cargar_textos, cargar_titulos  # noqa: E402
from pipeline.recuperacion import texto_documento  # noqa: E402

MODELO = "Qwen/Qwen3-Embedding-0.6B"
UMBRAL = 512
ESCENARIOS = ("frio", "caliente")
TAM_LOTE = 256


def longitudes(st, textos):
    """Tokens de cada texto, con los especiales, como en SentenceTransformer.tokenize (sin truncar)."""
    salida = []
    for i in range(0, len(textos), TAM_LOTE):
        salida += [len(ids) for ids in st.tokenizer(textos[i:i + TAM_LOTE], add_special_tokens=True)["input_ids"]]
    return salida


def resumen(v):
    v = np.asarray(v)
    return {"n": int(len(v)), "mas_de_512": int((v > UMBRAL).sum()), "pct_mas_de_512": float(100 * (v > UMBRAL).mean()),
            "max": int(v.max()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99))}


def main():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODELO, device="cpu")
    textos, titulos = cargar_textos(), cargar_titulos()
    ids = sorted(textos)
    base = dict(zip(ids, longitudes(st, [texto_documento(titulos[d], textos[d][0]) for d in ids])))

    # Comprobación: mismas longitudes que SentenceTransformer.tokenize (máscara de atención)
    muestra = ids[:64]
    mascara = st.tokenize([texto_documento(titulos[d], textos[d][0]) for d in muestra])["attention_mask"].sum(1)
    assert [base[d] for d in muestra] == mascara.tolist()

    salida = {"modelo": MODELO, "umbral": UMBRAL, "max_seq_length_por_defecto": st.max_seq_length,
              "tokens_especiales": base[ids[0]] - len(st.tokenizer(texto_documento(titulos[ids[0]], textos[ids[0]][0]),
                                                                   add_special_tokens=False)["input_ids"]),
              "coleccion": resumen(list(base.values())), "consultas": {}, "candidatos": {}}
    for esc in ESCENARIOS:
        candidatos = cargar_candidatos("recuperador", esc)
        consultas = [c for c in cargar_consultas(esc) if not c["es_desarrollo"]]
        assert len(consultas) == 1800
        largos, alterados = [], []
        listas_con_alguno = 0
        for c in consultas:
            vista = Vista(c)
            alt = set(candidatos[c["idx"]]) & set(vista.documentos_alterados())
            visibles = dict(zip(sorted(alt), longitudes(st, [texto_documento(vista.titulo(d), vista.texto(d))
                                                             for d in sorted(alt)]))) if alt else {}
            lista = [visibles.get(d, base[d]) for d in candidatos[c["idx"]]]
            largos += lista
            alterados += visibles.values()
            listas_con_alguno += any(x > UMBRAL for x in lista)
        salida["consultas"][esc] = resumen([base[c["idx"]] for c in consultas])
        salida["candidatos"][esc] = {**resumen(largos), "listas_con_alguno_de_mas_de_512": listas_con_alguno,
                                     "parejas_con_texto_visible": len(alterados),
                                     "documentos_distintos": len({d for c in consultas for d in candidatos[c["idx"]]})}
        print(esc, salida["consultas"][esc], salida["candidatos"][esc], flush=True)
    print("coleccion", salida["coleccion"])
    (RAIZ / "experimentos" / "resultados" / "longitud_tokens.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
