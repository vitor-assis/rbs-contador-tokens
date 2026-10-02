"""Catálogo de modelos e preços. O custo da conversa é calculado em core/conversation.py."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# MODELS_FILE permite trocar o catálogo (ex.: catalogo/todos_os_provedores.json); padrão: models.json
MODELS_PATH = ROOT / os.getenv("MODELS_FILE", "models.json")

DEFAULT_CPT_EN = 4.0
DEFAULT_CPT_PT = 3.3


def load_models(path: str | Path = MODELS_PATH) -> list[dict]:
    """Lê models.json e aplica os padrões de chars/token."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    models = data["models"] if isinstance(data, dict) else data
    for m in models:
        m.setdefault("chars_per_token_en", DEFAULT_CPT_EN)
        m.setdefault("chars_per_token_pt", DEFAULT_CPT_PT)
        m.setdefault("exact_method", "none")
        m.setdefault("local_method", None)
        m.setdefault("local_method_is_exact", False)
        m.setdefault("price_cache_read_per_1m", m.get("price_input_per_1m", 0.0))
        m.setdefault("price_cache_write_per_1m", m.get("price_input_per_1m", 0.0))
        m.setdefault("price_output_per_1m", None)
        m.setdefault("cache_min_tokens", 0)
        m.setdefault("notes", "")
    return models


@dataclass
class Prices:
    input: float
    cache_read: float
    cache_write: float
    output: float | None = None
    long_context: bool = False


def prices_for(model: dict, input_tokens: int) -> Prices:
    """Preços por 1M tokens para uma requisição com ``input_tokens`` de entrada.

    Quando o modelo tem faixa de contexto longo, ela vale para a requisição inteira
    (entrada e saída) se a entrada passar do limiar, como o Google e a OpenAI cobram.
    """
    lc = model.get("long_context")
    if lc and input_tokens > lc.get("threshold", float("inf")):
        return Prices(
            input=lc.get("price_input_per_1m", model["price_input_per_1m"]),
            cache_read=lc.get("price_cache_read_per_1m", model["price_cache_read_per_1m"]),
            cache_write=lc.get("price_cache_write_per_1m", model["price_cache_write_per_1m"]),
            output=lc.get("price_output_per_1m", model.get("price_output_per_1m")),
            long_context=True,
        )
    return Prices(
        input=model["price_input_per_1m"],
        cache_read=model["price_cache_read_per_1m"],
        cache_write=model["price_cache_write_per_1m"],
        output=model.get("price_output_per_1m"),
    )


def format_usd(value: float) -> str:
    """Formata em dólares no padrão brasileiro, com mais casas para valores pequenos."""
    if value == 0:
        return "US$ 0,00"
    decimals = 2 if value >= 1 else (4 if value >= 0.0001 else 6)
    s = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"US$ {s}"
