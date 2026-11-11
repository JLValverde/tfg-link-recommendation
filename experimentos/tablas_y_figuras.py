"""Regenera todas las tablas y figuras de resultados (T2.8) con un solo comando.

En este orden, cada uno en su proceso:
1. experimentos/tabla_principal.py (tabla_principal.{md,json}, de la que leen los demás);
2. experimentos/tablas_finales.py (tabla_resumen, tabla_ablacion, tabla_ablacion_subconjunto,
   tabla_techo, tabla_A_vs_B, tabla_malformadas y tabla_secundaria);
3. cada script de experimentos/figuras/fig_*.py (PDF y PNG en resultados/figuras/).
No puntúa ni llama a Ollama: solo lee resultados ya calculados.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe experimentos/tablas_y_figuras.py
"""

import subprocess
import sys
from pathlib import Path

DIR_EXPERIMENTOS = Path(__file__).resolve().parent
RAIZ = DIR_EXPERIMENTOS.parent


def main():
    scripts = [DIR_EXPERIMENTOS / "tabla_principal.py", DIR_EXPERIMENTOS / "tablas_finales.py",
               *sorted((DIR_EXPERIMENTOS / "figuras").glob("fig_*.py"))]
    for script in scripts:
        print(f"== {script.relative_to(RAIZ).as_posix()}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=RAIZ, check=True)


if __name__ == "__main__":
    main()
