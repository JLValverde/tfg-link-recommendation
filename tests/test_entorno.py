"""Comprobación del entorno: librerías, GPU, paquetes del proyecto y Ollama."""

import importlib

import pytest

LIBRERIAS = [
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "networkx",
    "matplotlib",
    "requests",
    "ollama",
    "sentence_transformers",
    "tqdm",
    "pytest",
    "bm25s",
]


@pytest.mark.parametrize("nombre", LIBRERIAS)
def test_importa_libreria(nombre):
    importlib.import_module(nombre)


def test_gpu_cuda():
    import torch

    assert torch.cuda.is_available()
    assert "5060" in torch.cuda.get_device_name(0)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    esperado = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.equal((a @ b).cpu(), esperado)


def test_importa_paquetes_proyecto():
    import arnes  # noqa: F401
    import pipeline  # noqa: F401


def test_ollama_responde():
    import ollama

    from arnes.coste import OLLAMA_URL

    r = ollama.Client(host=OLLAMA_URL).generate(
        model="qwen2.5:3b",
        prompt="Responde solo 'ok'",
        options={"num_ctx": 4096},
    )
    assert "ok" in r["response"].lower()
