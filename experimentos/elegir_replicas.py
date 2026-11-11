"""T2.6: modelo de cada nivel de presupuesto y regla del 14B, con las reglas fijadas antes de medir.

Lee, para cada modelo, vram_presupuesto_{tag}.json (pico de toda la GPU, de
arnes.medir_vram) y presupuesto_{tag}.json (latencia y casos), de
experimentos/presupuesto_vram.py.

- Regla del ADR "El presupuesto de recursos son tres topes de VRAM de 8, 12 y 16 GB":
  para cada tope (8000, 12000 y 16000 MiB), el mayor modelo cuyo pico no lo supera. Si
  dos niveles eligen el mismo modelo, se ejecuta una vez; su nivel en el coste es el
  más bajo de los que lo eligen (el tope más estricto).
- Si ni el 3B con el modelo de embeddings cabe en 8000 MiB, para (se reabre el ADR del
  modelo de embeddings).
- Regla del 14B (Roadmap Fase 2, T2.6): si su estimación para 2000 consultas x 2
  escenarios supera 10 h, en T2.7 se ejecuta solo en frío.
- Decisión tomada en T2.7 ("el 14B en los dos escenarios"):
  la estimación de la regla se midió con la latencia inflada por el codificador de
  embeddings, así que el 14B va en los dos escenarios, frío primero. El resultado de la
  regla se conserva en regla_14b, con "aplicada": false.

Escribe experimentos/resultados/presupuesto_niveles.json.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/elegir_replicas.py
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from arnes.coste import NIVELES  # noqa: E402
from arnes.evaluar import DIR_RESULTADOS  # noqa: E402

# De menor a mayor
MODELOS = (("3b", "qwen2.5:3b"), ("7b", "qwen2.5:7b-instruct-q4_K_M"), ("14b", "qwen2.5:14b-instruct-q4_K_M"))
HORAS_MAX_14B = 10
# Decisión de Jose (T2.7): sustituye a la regla del 14B; en este orden
ESCENARIOS_14B = ["frio", "caliente"]
DECISION_14B = ("Decisión de Jose (T2.7): la estimación se midió con la latencia inflada por el "
                "codificador de embeddings de T2.6; el 14B se ejecuta en los dos escenarios, frío primero")


def main():
    medidas = {}
    for tag, modelo in MODELOS:
        vram = json.loads((DIR_RESULTADOS / f"vram_presupuesto_{tag}.json").read_text(encoding="utf-8"))
        pres = json.loads((DIR_RESULTADOS / f"presupuesto_{tag}.json").read_text(encoding="utf-8"))
        assert vram["codigo_salida"] == 0 and pres["error_codificador"] is None, tag
        medidas[tag] = {"modelo": modelo, "pico_mib": vram["pico_mib"], "base_mib": vram["base_mib"],
                        "incremento_mib": vram["incremento_mib"],
                        "horas_estimadas_2000_x_2": pres["horas_estimadas_2000_consultas_x_2_escenarios"],
                        "casos": {e: {c: d["diagnostico"][c] for c in ("ambos", "falta_uno", "faltan_ambos")}
                                  for e, d in pres["por_escenario"].items()}}
    if medidas["3b"]["pico_mib"] > NIVELES["bajo"]:
        sys.exit(f"PARA: el 3B con el modelo de embeddings no cabe en {NIVELES['bajo']} MiB "
                 f"(pico {medidas['3b']['pico_mib']}): se reabre el ADR del modelo de embeddings")
    niveles = {}
    for nivel, tope in NIVELES.items():
        caben = [tag for tag, _ in MODELOS if medidas[tag]["pico_mib"] <= tope]
        niveles[nivel] = {"tope_mib": tope, "modelo": caben[-1] if caben else None}
    replicas = {}
    for nivel in sorted(NIVELES, key=NIVELES.get):  # el nivel más bajo que elige cada modelo
        tag = niveles[nivel]["modelo"]
        if tag and tag != "3b" and tag not in replicas:
            replicas[tag] = {"modelo": medidas[tag]["modelo"], "nivel": nivel,
                             "niveles": [n for n in NIVELES if niveles[n]["modelo"] == tag]}
    solo_frio_14b = medidas["14b"]["horas_estimadas_2000_x_2"] > HORAS_MAX_14B
    if "14b" in replicas:
        replicas["14b"]["escenarios"] = ESCENARIOS_14B
    for tag in replicas:
        replicas[tag].setdefault("escenarios", ["caliente", "frio"])
    salida = {"medidas": medidas, "niveles": niveles, "replicas_t27": replicas,
              "orden_t27": [t for t, _ in MODELOS if t in replicas],
              "regla_14b": {"horas_estimadas": medidas["14b"]["horas_estimadas_2000_x_2"],
                            "umbral_h": HORAS_MAX_14B, "solo_frio": solo_frio_14b,
                            "aplicada": False, "decision": DECISION_14B},
              "nivel_3b": min((n for n in NIVELES if medidas["3b"]["pico_mib"] <= NIVELES[n]), key=NIVELES.get)}
    (DIR_RESULTADOS / "presupuesto_niveles.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(salida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
