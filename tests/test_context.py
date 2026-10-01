import json

import pytest

from core.context import (
    SEPARATOR,
    Block,
    assemble,
    minify_json,
    parse_tools,
    validate_json,
)
from core.counters import Settings, estimate, heuristic_tokens
from core.vision import VisualItem

TOOLS_ANTHROPIC = json.dumps(
    [{"name": "buscar", "description": "Busca produtos", "input_schema": {"type": "object", "properties": {}}}]
)
TOOLS_OPENAI = json.dumps(
    [{"type": "function", "function": {"name": "clima", "description": "Clima", "parameters": {"type": "object"}}}]
)

MODEL = {
    "id": "teste", "label": "Teste", "provider": "X", "context_window": 1000,
    "chars_per_token_en": 4.0, "chars_per_token_pt": 4.0,
    "exact_method": "none", "local_method": None, "local_method_is_exact": False,
    "price_input_per_1m": 1.0, "price_cache_read_per_1m": 0.1, "price_cache_write_per_1m": 1.25,
}


def test_montagem_respeita_ordem_e_ignora_vazios():
    blocks = [
        Block("Sys", "system", "Você é um assistente."),
        Block("Vazio", "instrucao", "   "),
        Block("Persona", "persona", "Fale como pirata."),
    ]
    ctx = assemble(blocks)
    assert [b.name for b in ctx.blocks] == ["Sys", "Persona"]
    assert ctx.full_text == "Você é um assistente." + SEPARATOR + "Fale como pirata."
    assert ctx.system_text == ctx.full_text
    assert ctx.tools is None


def test_arquivo_envolvido_em_documento():
    ctx = assemble([Block("faq.md", "arquivo", "conteúdo")], wrap_files=True)
    assert ctx.full_text == '<documento nome="faq.md">\nconteúdo\n</documento>'
    ctx2 = assemble([Block("faq.md", "arquivo", "conteúdo")], wrap_files=False)
    assert ctx2.full_text == "conteúdo"


def test_tools_vao_para_campo_tools_e_para_o_texto():
    ctx = assemble([Block("Sys", "system", "Olá"), Block("Tools", "tools", TOOLS_ANTHROPIC)])
    assert ctx.tools == [{"name": "buscar", "description": "Busca produtos",
                          "parameters": {"type": "object", "properties": {}}}]
    assert TOOLS_ANTHROPIC not in ctx.system_text
    assert TOOLS_ANTHROPIC in ctx.full_text


def test_tools_formato_openai_normalizado():
    tools = parse_tools(TOOLS_OPENAI)
    assert tools == [{"name": "clima", "description": "Clima", "parameters": {"type": "object"}}]


def test_tools_invalidas_viram_texto_com_aviso():
    ctx = assemble([Block("Tools", "tools", "função buscar(x): busca x")])
    assert ctx.tools is None
    assert "função buscar" in ctx.system_text
    assert ctx.warnings


def test_validacao_json_amigavel():
    ok, msg = validate_json('{"a": 1}')
    assert ok and msg is None
    ok, msg = validate_json('{"a": 1,}')
    assert not ok and "linha 1" in msg
    ok, msg = validate_json("")
    assert not ok


def test_minificar_json():
    assert minify_json('{\n  "a": 1,\n  "b": [1, 2]\n}') == '{"a":1,"b":[1,2]}'
    assert minify_json('{"nome": "João"}') == '{"nome":"João"}'
    with pytest.raises(ValueError):
        minify_json("{quebrado")


def test_total_e_soma_das_partes_diferem():
    """O contexto montado inclui separadores, então total heurístico != soma isolada."""
    blocks = [Block("A", "system", "a" * 10), Block("B", "instrucao", "b" * 10)]
    ctx = assemble(blocks)
    est = estimate(MODEL, ctx, Settings(lang="en"))
    # 10 chars -> 3 tokens cada (ceil 2.5); montado: 22 chars -> 6 tokens
    assert est.heuristic_sum == 6
    assert est.heuristic_total == heuristic_tokens("a" * 10 + SEPARATOR + "b" * 10, 4.0, 4.0, "en") == 6
    blocks = [Block("A", "system", "a" * 9), Block("B", "instrucao", "b" * 9)]
    est = estimate(MODEL, assemble(blocks), Settings(lang="en"))
    assert est.heuristic_sum == 6          # 3 + 3
    assert est.heuristic_total == 5        # ceil(20 / 4)


def test_overhead_e_paginas_pdf_somados():
    ctx = assemble([Block("A", "system", "a" * 40)])
    model = {**MODEL, "vision": {"image": {"method": "fixed", "tokens": 100}, "pdf": {"method": "fixed", "tokens": 25}}}
    pdf = VisualItem("manual.pdf", "pdf", [(612, 792), (612, 792)])
    est = estimate(model, ctx, Settings(lang="en", overhead_tokens=100, visual_items=[pdf]))
    assert est.heuristic_total == 10 + 100 + 50
    names = [r.name for r in est.rows]
    assert "Overhead fixo" in names and "manual.pdf" in names
    assert est.best_label.startswith("estimado")


def test_resultado_api_tem_prioridade_e_nao_soma_overhead_quando_inclui_wrappers():
    ctx = assemble([Block("A", "system", "a" * 40)])
    api = {"total": 42, "blocks": [30], "label": "exato (API anthropic)", "includes_wrappers": True}
    est = estimate(MODEL, ctx, Settings(lang="en", overhead_tokens=100), api_result=api)
    assert est.best_total == 42
    assert est.exact_total == 42
    assert est.api_sum == 30
    assert est.calibration["observed_chars_per_token"] == pytest.approx(40 / 30)


def test_pct_da_janela():
    ctx = assemble([Block("A", "system", "a" * 2000)])  # 500 tokens em 1000
    est = estimate(MODEL, ctx, Settings(lang="en"))
    assert est.window_pct == pytest.approx(50.0)
