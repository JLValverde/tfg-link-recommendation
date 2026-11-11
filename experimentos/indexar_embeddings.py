"""Indexa la colección con un modelo de embeddings en el directorio indicado.

Pensado para medir la VRAM del indexado en un proceso aislado con arnes/medir_vram:
carga el modelo, codifica título + introducción de los 10989 documentos válidos
(lotes de 32, como en T1.6) y guarda el .npz en --dir-salida, sin tocar
datos/procesados/embeddings/.

Uso (desde la raíz del repo):
    .venv/Scripts/python.exe -m arnes.medir_vram --salida F --descargar-modelo qwen2.5:3b -- \\
        .venv/Scripts/python.exe experimentos/indexar_embeddings.py --modelo M --dir-salida D
"""

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from pipeline import recuperacion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--dir-salida", required=True)
    args = ap.parse_args()
    if Path(args.dir_salida).resolve() == recuperacion.DIR_EMBEDDINGS.resolve():
        sys.exit("--dir-salida no puede ser datos/procesados/embeddings/")
    t0 = time.perf_counter()
    recuperacion.cargar_modelo(args.modelo)
    t1 = time.perf_counter()
    recuperacion.IndiceEmbeddings(args.modelo, recalcular=True, directorio=args.dir_salida)
    t2 = time.perf_counter()
    print(f"carga del modelo: {t1 - t0:.1f} s; indexado: {t2 - t1:.1f} s; lote {recuperacion.TAM_LOTE}")


if __name__ == "__main__":
    main()
