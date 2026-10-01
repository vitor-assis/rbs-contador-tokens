"""Lógica de contagem via API, sem rede (as chamadas ao provedor são simuladas)."""
import pytest

import core.counters as counters
from core.context import Block, assemble

TOOLS = '[{"name":"buscar","description":"Busca","input_schema":{"type":"object","properties":{}}}]'


def _model(provider):
    return {"id": f"m-{provider}", "exact_method": f"api:{provider}"}


def test_sem_chave_levanta_erro_amigavel(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = assemble([Block("S", "system", "oi")])
    with pytest.raises(counters.MissingKeyError, match="ANTHROPIC_API_KEY"):
        counters.count_via_api(_model("anthropic"), ctx)


def test_modelo_sem_api():
    ctx = assemble([Block("S", "system", "oi")])
    with pytest.raises(counters.ApiCountError):
        counters.count_via_api({"id": "x", "exact_method": "none"}, ctx)


def test_blocos_descontam_linha_de_base_quando_api_conta_requisicao(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "teste")
    calls = []

    def fake(provider, model_id, key, system_text, tools, full_text):
        calls.append((system_text, tools))
        return 10 + len(system_text) + (100 if tools else 0)  # 10 = linha de base

    monkeypatch.setattr(counters, "_api_count_request", fake)
    ctx = assemble([Block("S", "system", "abcde"), Block("T", "tools", TOOLS)])
    res = counters.count_via_api(_model("anthropic"), ctx, per_block=True)
    assert res["total"] == 10 + 5 + 100
    assert res["blocks"] == [5, 100]
    assert res["includes_wrappers"] is True
    # total com tools estruturadas, linha de base sem nada
    assert calls[0] == ("abcde", ctx.tools)
    assert calls[1] == ("", None)


def test_api_de_texto_nao_desconta_linha_de_base(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "teste")
    monkeypatch.setattr(counters, "_api_count_request",
                        lambda p, m, k, s, t, f: len(f))
    ctx = assemble([Block("S", "system", "abc"), Block("E", "interno_texto", "defg")])
    res = counters.count_via_api(_model("xai"), ctx, per_block=True)
    assert res["total"] == len(ctx.full_text)
    assert res["blocks"] == [3, 4]
    assert res["includes_wrappers"] is False


def test_fingerprint_muda_com_o_conteudo():
    a = assemble([Block("S", "system", "abc")])
    b = assemble([Block("S", "system", "abd")])
    assert counters.context_fingerprint("m", a, True) != counters.context_fingerprint("m", b, True)
    assert counters.context_fingerprint("m", a, True) == counters.context_fingerprint("m", a, True)
    assert counters.context_fingerprint("m", a, True) != counters.context_fingerprint("m", a, False)
