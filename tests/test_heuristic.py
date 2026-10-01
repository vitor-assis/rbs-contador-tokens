import math

import pytest

from core.counters import (
    effective_chars_per_token,
    heuristic_tokens,
    parse_method,
    local_method_for,
    tokens_per_char,
)


def test_texto_vazio_da_zero():
    assert heuristic_tokens("", 4.0, 3.3, "pt") == 0


def test_ingles_usa_chars_per_token_en():
    assert heuristic_tokens("a" * 400, 4.0, 3.3, "en") == 100


def test_portugues_usa_chars_per_token_pt():
    assert heuristic_tokens("a" * 330, 4.0, 3.3, "pt") == 100


def test_arredonda_para_cima():
    assert heuristic_tokens("abcde", 4.0, 3.3, "en") == 2
    assert heuristic_tokens("a", 4.0, 3.3, "en") == 1


def test_misto_e_media_ponderada_de_tokens_por_char():
    tpc = tokens_per_char(4.0, 3.3, "misto", pt_share=0.5)
    assert tpc == pytest.approx(0.5 / 3.3 + 0.5 / 4.0)
    # 60% PT: fica entre os dois extremos, mais perto do PT
    n = 10_000
    pt = heuristic_tokens("x" * n, 4.0, 3.3, "pt")
    en = heuristic_tokens("x" * n, 4.0, 3.3, "en")
    mix = heuristic_tokens("x" * n, 4.0, 3.3, "misto", pt_share=0.6)
    assert en < mix < pt
    assert mix == math.ceil(n * (0.6 / 3.3 + 0.4 / 4.0))


def test_misto_extremos_equivalem_aos_idiomas_puros():
    assert tokens_per_char(4.0, 3.3, "misto", 1.0) == pytest.approx(tokens_per_char(4.0, 3.3, "pt"))
    assert tokens_per_char(4.0, 3.3, "misto", 0.0) == pytest.approx(tokens_per_char(4.0, 3.3, "en"))


def test_chars_per_token_efetivo():
    assert effective_chars_per_token(4.0, 3.3, "en") == pytest.approx(4.0)
    assert effective_chars_per_token(4.0, 3.3, "pt") == pytest.approx(3.3)


def test_parse_method():
    assert parse_method("tiktoken:o200k_base") == ("tiktoken", "o200k_base")
    assert parse_method("hf:deepseek-ai/DeepSeek-V4.1-Flash") == ("hf", "deepseek-ai/DeepSeek-V4.1-Flash")
    assert parse_method("api:anthropic") == ("api", "anthropic")
    assert parse_method("none") == ("none", "")
    assert parse_method(None) == ("none", "")


def test_local_method_prioriza_exact_local():
    assert local_method_for({"exact_method": "tiktoken:o200k_base"}) == ("tiktoken:o200k_base", True)
    assert local_method_for({"exact_method": "api:openai", "local_method": "tiktoken:o200k_base",
                             "local_method_is_exact": False}) == ("tiktoken:o200k_base", False)
    assert local_method_for({"exact_method": "api:anthropic", "local_method": None}) == (None, False)


def test_metodo_genai_e_local():
    from core.counters import is_local_method, method_label
    assert is_local_method("genai:gemini-2.5-pro")
    assert local_method_for({"exact_method": "api:gemini", "local_method": "genai:gemini-2.5-pro",
                             "local_method_is_exact": True}) == ("genai:gemini-2.5-pro", True)
    assert "google-genai" in method_label("genai:gemini-2.5-pro")


def test_tokenizador_local_oficial_do_gemini():
    pytest.importorskip("sentencepiece")
    from core.counters import LocalTokenizerError, count_local
    try:
        n = count_local("genai:gemini-2.5-flash", "Olá, mundo! Tudo bem?")
    except LocalTokenizerError as e:  # sem rede no primeiro download
        pytest.skip(str(e))
    assert 3 <= n <= 15
