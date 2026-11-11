"""Variantes del prompt de C1 que se prueban en desarrollo (T2.2; como mucho 3).

Solo cambia el texto del prompt; el resto de la receta (vía, topes, num_ctx,
top_logprobs, política de confianza) es el de pipeline/reordenacion.py.
- V1: la pregunta de enlazar (T2.2a).
- V2: V1 cambiando solo la última línea por una pregunta de relación temática.
- V3: V2 con una instrucción al principio, separada por una línea en blanco.

La ganadora se congela en pipeline/prompts/c1.txt (experimentos/seleccionar_c1.py);
desde ahí, el scorer lee ese fichero y estas variantes solo sirven para reproducir
las ejecuciones de desarrollo (experimentos/evaluar_c1.py --desarrollo --variante N).
"""

CUERPO = ("Article A:\n"
          "Title: {titulo_consulta}\n"
          "{texto_consulta}\n"
          "\n"
          "Article B:\n"
          "Title: {titulo_candidato}\n"
          "{texto_candidato}\n"
          "\n")
PREGUNTA_V1 = "Should article A link to article B? Answer Yes or No."
PREGUNTA_V2 = "Is article B closely related to the topic of article A? Answer Yes or No."
INSTRUCCION_V3 = ("The following are two Wikipedia articles. Wikipedia editors link from an article to another "
                  "article when a reader of the first would benefit from consulting the second, because it covers "
                  "a concept, entity or topic closely related to it.")

VARIANTES = {
    1: CUERPO + PREGUNTA_V1,
    2: CUERPO + PREGUNTA_V2,
    3: INSTRUCCION_V3 + "\n\n" + CUERPO + PREGUNTA_V2,
}


def sistema(variante):
    """Nombre del sistema en los ficheros de desarrollo: V1 conserva el de T2.2a ("c1")."""
    return "c1" if variante == 1 else f"c1_v{variante}"
