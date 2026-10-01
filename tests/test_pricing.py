import pytest

from core.counters import parse_method
from core.pricing import MODELS_PATH, ROOT, estimate_cost, format_usd, load_models, prices_for

MODEL = {
    "price_input_per_1m": 2.0, "price_cache_read_per_1m": 0.2, "price_cache_write_per_1m": 2.5,
    "long_context": {"threshold": 200_000, "price_input_per_1m": 4.0,
                     "price_cache_read_per_1m": 0.4, "price_cache_write_per_1m": 5.0},
}
FULL_CATALOG = ROOT / "catalogo" / "todos_os_provedores.json"


def test_custo_sem_cache():
    c = estimate_cost(MODEL, 100_000, 10, cache_enabled=False)
    assert c.per_conversation_no_cache == pytest.approx(0.2)
    assert c.monthly == pytest.approx(2.0)


def test_custo_com_cache_primeira_escrita_depois_leituras():
    c = estimate_cost(MODEL, 100_000, 10, cache_enabled=True, cache_hit_rate=1.0)
    assert c.per_interaction_cache_write == pytest.approx(0.25)
    assert c.per_interaction_cache_read == pytest.approx(0.02)
    assert c.monthly == pytest.approx(0.25 + 9 * 0.02)


def test_taxa_de_acerto_parcial():
    c = estimate_cost(MODEL, 1_000_000, 11, cache_enabled=True, cache_hit_rate=0.5)
    # 1ª escrita + 10 restantes (5 leituras, 5 escritas)  -> faixa longa (> 200k)
    assert c.long_context_pricing
    assert c.monthly == pytest.approx(6 * 5.0 + 5 * 0.4)


def test_interacoes_por_conversa_multiplicam_o_contexto():
    # 100k tokens, 10 conversas x 4 interações = 40 requisições
    c = estimate_cost(MODEL, 100_000, 10, cache_enabled=False, interactions_per_conversation=4)
    assert c.tokens_per_conversation == 400_000
    assert c.tokens_per_month == 4_000_000
    assert c.per_interaction_no_cache == pytest.approx(0.2)
    assert c.per_conversation_no_cache == pytest.approx(0.8)
    assert c.monthly == pytest.approx(8.0)


def test_interacoes_com_cache():
    c = estimate_cost(MODEL, 100_000, 10, cache_enabled=True, cache_hit_rate=1.0,
                      interactions_per_conversation=4)
    assert c.per_conversation_cache_cold == pytest.approx(0.25 + 3 * 0.02)   # escreve 1x, lê 3x
    assert c.per_conversation_cache_warm == pytest.approx(4 * 0.02)
    assert c.monthly == pytest.approx(0.25 + 39 * 0.02)                      # 40 requisições no mês
    assert c.per_conversation == c.per_conversation_cache_cold


def test_interacoes_minimo_um():
    c = estimate_cost(MODEL, 100_000, 10, cache_enabled=False, interactions_per_conversation=0)
    assert c.interactions_per_conversation == 1


def test_faixa_de_contexto_longo():
    assert prices_for(MODEL, 200_000).input == 2.0
    assert prices_for(MODEL, 200_001).input == 4.0


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
