import pytest

from core.counters import parse_method
from core.pricing import MODELS_PATH, ROOT, format_usd, load_models, prices_for

MODEL = {
    "price_input_per_1m": 2.0, "price_cache_read_per_1m": 0.2, "price_cache_write_per_1m": 2.5,
    "long_context": {"threshold": 200_000, "price_input_per_1m": 4.0,
                     "price_cache_read_per_1m": 0.4, "price_cache_write_per_1m": 5.0},
}
FULL_CATALOG = ROOT / "catalogo" / "todos_os_provedores.json"


def test_faixa_de_contexto_longo():
    assert prices_for(MODEL, 200_000).input == 2.0
    assert prices_for(MODEL, 200_001).input == 4.0


def test_todos_os_modelos_tem_preco_de_saida():
    for m in load_models(FULL_CATALOG):
        assert m["price_output_per_1m"] and m["price_output_per_1m"] > m["price_input_per_1m"], m["id"]
    for m in load_models(MODELS_PATH):
        assert m["cache_min_tokens"] in (2048, 4096), m["id"]


def test_format_usd():
    assert format_usd(1234.5) == "US$ 1.234,50"
    assert format_usd(0.0123) == "US$ 0,0123"


def _valida_catalogo(models):
    ids = [m["id"] for m in models]
    assert len(ids) == len(set(ids)), "IDs duplicados"
    for m in models:
        for key in ("id", "label", "provider", "context_window", "chars_per_token_en", "chars_per_token_pt",
                    "exact_method", "price_input_per_1m", "price_cache_read_per_1m",
                    "price_cache_write_per_1m", "price_checked_at"):
            assert key in m, f"{m['id']}: falta {key}"
        kind, _ = parse_method(m["exact_method"])
        assert kind in {"tiktoken", "hf", "api", "none"}
        assert m["context_window"] > 0


def test_models_json_publicado_so_google():
    models = load_models(MODELS_PATH)
    _valida_catalogo(models)
    assert {m["provider"] for m in models} == {"Google"}
    assert {m["id"] for m in models} >= {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.5-flash",
                                          "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"}


def test_catalogo_completo():
    models = load_models(FULL_CATALOG)
    _valida_catalogo(models)
    assert len({m["provider"] for m in models}) >= 6
    published = {m["id"]: m for m in load_models(MODELS_PATH)}
    for m in models:  # os modelos do Google são idênticos nos dois catálogos
        if m["provider"] == "Google":
            assert m == published[m["id"]]
