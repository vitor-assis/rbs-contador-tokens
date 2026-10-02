"""Simulação da conversa: valores conferidos à mão."""
import pytest

from core.conversation import ConversationParams, simulate_conversation
from core.counters import CHAT_SAMPLES, count_words, tokens_per_word

MODEL = {
    "id": "teste", "context_window": 1_000_000, "cache_min_tokens": 0,
    "price_input_per_1m": 1.0, "price_cache_read_per_1m": 0.1, "price_cache_write_per_1m": 1.0,
    "price_output_per_1m": 10.0,
    "chars_per_token_en": 4.0, "chars_per_token_pt": 3.3, "exact_method": "none", "local_method": None,
}
BASE = dict(turns=3, user_tokens=10, response_tokens=100, thinking_tokens=50,
            conversations_per_month=1000, cache_enabled=True, cache_hit_rate=1.0)


def test_entrada_cresce_com_o_historico_e_prefixo_vem_do_cache():
    c = simulate_conversation(MODEL, 1000, ConversationParams(**BASE))
    # turno 1: contexto + msg; turno 2: + (msg + resposta) do turno 1; turno 3: + mais uma troca
    assert [t.input_tokens for t in c.turns] == [1010, 1120, 1230]
    # cache: contexto no 1º turno; depois, a entrada inteira do turno anterior
    assert [t.cached_tokens for t in c.turns] == [1000, 1010, 1120]
    assert [t.new_input_tokens for t in c.turns] == [10, 110, 110]
    assert c.input_total == 3360 and c.cached_total == 3130
    assert c.output_total == 3 * 150 and c.thinking_total == 150
    assert c.cost_input == pytest.approx((110 + 211 + 222) / 1e6)
    assert c.cost_output == pytest.approx(4500 / 1e6)
    assert c.monthly_cost == pytest.approx(c.cost * 1000)


def test_sem_cache_toda_entrada_paga_preco_cheio():
    c = simulate_conversation(MODEL, 1000, ConversationParams(**{**BASE, "cache_enabled": False}))
    assert c.cached_total == 0
    assert c.cost_input == pytest.approx(3360 / 1e6)


def test_minimo_do_cache_implicito():
    model = {**MODEL, "cache_min_tokens": 2000}
    c = simulate_conversation(model, 1000, ConversationParams(**BASE))
    assert c.cached_total == 0   # nenhuma requisição anterior chega a 2.000 tokens


def test_raciocinio_no_historico_pior_caso():
    c = simulate_conversation(MODEL, 1000, ConversationParams(**{**BASE, "thoughts_in_history": True}))
    assert [t.input_tokens for t in c.turns] == [1010, 1170, 1330]


def test_taxa_de_acerto_parcial():
    c = simulate_conversation(MODEL, 1000, ConversationParams(**{**BASE, "cache_hit_rate": 0.5}))
    assert c.turns[0].cached_tokens == 500 and c.turns[1].cached_tokens == 505


def test_faixa_de_contexto_longo_vale_para_entrada_e_saida():
    model = {**MODEL, "long_context": {"threshold": 1100, "price_input_per_1m": 2.0,
                                       "price_cache_read_per_1m": 0.2, "price_cache_write_per_1m": 2.0,
                                       "price_output_per_1m": 20.0}}
    c = simulate_conversation(model, 1000, ConversationParams(**BASE))
    assert [t.long_context for t in c.turns] == [False, True, True]
    assert c.turns[1].cost_output == pytest.approx(150 * 20 / 1e6)


def test_pico_da_janela_e_no_ultimo_turno():
    c = simulate_conversation({**MODEL, "context_window": 10_000}, 1000, ConversationParams(**BASE))
    assert c.peak_window_pct == pytest.approx((1230 + 150) / 10_000 * 100)


def test_sem_preco_de_saida_avisa():
    model = {k: v for k, v in MODEL.items() if k != "price_output_per_1m"}
    c = simulate_conversation(model, 1000, ConversationParams(**BASE))
    assert c.output_price_missing and c.cost_output == 0


def test_tokens_por_palavra_heuristica():
    tpw, label = tokens_per_word(MODEL, "pt")
    words = count_words(CHAT_SAMPLES["pt"])
    assert label == "heurística"
    assert tpw == pytest.approx(-(-len(CHAT_SAMPLES["pt"]) // 3.3) / words, rel=0.01)
    assert 1.0 < tpw < 3.0
    misto, _ = tokens_per_word(MODEL, "misto", pt_share=0.5)
    en, _ = tokens_per_word(MODEL, "en")
    assert misto == pytest.approx((tpw + en) / 2)


def test_tokens_por_palavra_com_tokenizador_do_gemini():
    pytest.importorskip("sentencepiece")
    model = {**MODEL, "local_method": "genai:gemini-2.5-flash", "local_method_is_exact": True}
    try:
        tpw, label = tokens_per_word(model, "pt")
    except Exception as e:  # sem rede no primeiro download
        pytest.skip(str(e))
    if label == "heurística":
        pytest.skip("tokenizador indisponível")
    assert "exato" in label and 1.0 < tpw < 2.5
