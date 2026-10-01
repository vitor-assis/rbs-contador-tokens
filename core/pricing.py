"""Catálogo de modelos e cálculo de custo do contexto inicial."""
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
        m.setdefault("notes", "")
    return models


@dataclass
class Prices:
    input: float
    cache_read: float
    cache_write: float
    long_context: bool = False


def prices_for(model: dict, tokens: int) -> Prices:
    """Preços por 1M tokens, considerando a faixa de contexto longo, se houver."""
    lc = model.get("long_context")
    if lc and tokens > lc.get("threshold", float("inf")):
        return Prices(
            input=lc.get("price_input_per_1m", model["price_input_per_1m"]),
            cache_read=lc.get("price_cache_read_per_1m", model["price_cache_read_per_1m"]),
            cache_write=lc.get("price_cache_write_per_1m", model["price_cache_write_per_1m"]),
            long_context=True,
        )
    return Prices(
        input=model["price_input_per_1m"],
        cache_read=model["price_cache_read_per_1m"],
        cache_write=model["price_cache_write_per_1m"],
    )


@dataclass
class CostEstimate:
    """Custo da entrada do contexto inicial. Ele é reenviado a cada interação (requisição) da conversa."""
    interactions_per_conversation: int
    tokens_per_conversation: int          # contexto inicial x interações
    tokens_per_month: int                 # x conversas
    per_interaction_no_cache: float
    per_interaction_cache_write: float
    per_interaction_cache_read: float
    per_conversation_no_cache: float
    per_conversation_cache_cold: float    # 1ª interação escreve o cache, as demais leem
    per_conversation_cache_warm: float    # cache já válido: todas as interações leem
    monthly_no_cache: float
    monthly_with_cache: float
    cache_enabled: bool
    long_context_pricing: bool

    @property
    def per_conversation(self) -> float:
        return self.per_conversation_cache_cold if self.cache_enabled else self.per_conversation_no_cache

    @property
    def monthly(self) -> float:
        return self.monthly_with_cache if self.cache_enabled else self.monthly_no_cache


def estimate_cost(
    model: dict,
    tokens: int,
    conversations_per_month: int,
    cache_enabled: bool,
    cache_hit_rate: float = 0.95,
    interactions_per_conversation: int = 1,
) -> CostEstimate:
    """Custo só da ENTRADA do contexto inicial (não inclui mensagens, respostas nem histórico).

    Cada interação de uma conversa é uma requisição que reenvia o contexto inicial, então o
    volume mensal é ``conversas x interações`` requisições. Com cache, cada requisição é leitura
    (acerto) ou escrita (erro/expiração): ``cache_hit_rate`` é a fração de acertos, e a primeira
    requisição do mês é sempre escrita.
    """
    p = prices_for(model, tokens)
    per_m = tokens / 1_000_000
    no_cache = per_m * p.input
    write = per_m * p.cache_write
    read = per_m * p.cache_read

    turns = max(1, int(interactions_per_conversation))
    convs = max(0, int(conversations_per_month))
    requests = convs * turns
    monthly_no_cache = no_cache * requests
    if requests == 0:
        monthly_cache = 0.0
    else:
        hit = min(max(cache_hit_rate, 0.0), 1.0)
        reads = (requests - 1) * hit
        writes = requests - reads
        monthly_cache = writes * write + reads * read

    return CostEstimate(
        interactions_per_conversation=turns,
        tokens_per_conversation=tokens * turns,
        tokens_per_month=tokens * requests,
        per_interaction_no_cache=no_cache,
        per_interaction_cache_write=write,
        per_interaction_cache_read=read,
        per_conversation_no_cache=no_cache * turns,
        per_conversation_cache_cold=write + read * (turns - 1),
        per_conversation_cache_warm=read * turns,
        monthly_no_cache=monthly_no_cache,
        monthly_with_cache=monthly_cache,
        cache_enabled=cache_enabled,
        long_context_pricing=p.long_context,
    )


def format_usd(value: float) -> str:
    """Formata em dólares no padrão brasileiro, com mais casas para valores pequenos."""
    if value == 0:
        return "US$ 0,00"
    decimals = 2 if value >= 1 else (4 if value >= 0.0001 else 6)
    s = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"US$ {s}"
