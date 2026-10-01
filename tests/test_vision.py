"""Fórmulas visuais conferidas contra os exemplos publicados pelos provedores."""
import io
from pathlib import Path

import pytest

from core.context import Block, assemble
from core.counters import Settings, estimate
from core.extractors import extract
from core.pricing import load_models
from core.vision import (
    DETAIL_LOW,
    VisualItem,
    claude_image_tokens,
    claude_resized_size,
    image_tokens,
    openai_image_tokens,
    pdf_page_tokens,
    qwen_image_tokens,
    visual_item_tokens,
)

FULL_CATALOG = Path(__file__).resolve().parent.parent / "catalogo" / "todos_os_provedores.json"
MODELS = {m["id"]: m for m in load_models(FULL_CATALOG)}


# Tabela de https://platform.claude.com/docs/en/build-with-claude/vision (Resolution and token cost)
@pytest.mark.parametrize("w,h,std,hi", [
    (200, 200, 64, 64),
    (1000, 1000, 1296, 1296),
    (1092, 1092, 1521, 1521),
    (1920, 1080, 1560, 2691),
    (2000, 1500, 1564, 3888),
    (3840, 2160, 1560, 4784),
])
def test_claude_tabela_da_doc(w, h, std, hi):
    assert claude_image_tokens(w, h, 1568, 1568) == std
    assert claude_image_tokens(w, h, 2576, 4784) == hi


def test_claude_redimensionamento_exemplos_da_doc():
    assert claude_resized_size(1075, 1520) == (924, 1307)       # A4 a 130 DPI
    assert claude_resized_size(1920, 1080) == (1456, 819)
    assert claude_resized_size(3840, 2160, 2576, 4784) == (2576, 1449)


def test_openai_exemplo_da_doc_high():
    # "2048 × 2048 ... 50 × 50 = 2500 patches ... ceil(2500 × 1.2) = 3000"
    tokens, note = openai_image_tokens(2048, 2048, 1.2, "high")
    assert tokens == 3000 and note is None


def test_openai_auto_nao_reduz_e_low_limita_a_512():
    assert openai_image_tokens(2048, 2048, 1.2, "auto")[0] == 4916  # ceil(64 * 64 * 1.2)
    assert openai_image_tokens(2048, 1024, 1.2, "low")[0] == 154    # 512x256 -> 16 * 8 = 128 patches
    tokens, note = openai_image_tokens(8000, 8000, 1.2, "auto")
    assert note  # > 30.000 patches


def test_qwen_smart_resize():
    assert qwen_image_tokens(1024, 1024) == 32 * 32 + 2
    assert qwen_image_tokens(2048, 2048) == 35 * 35 + 2        # reduzida para caber em 1280 tokens
    assert qwen_image_tokens(10, 10) == 2 * 2 + 2              # ampliada até o mínimo de 4 tokens


def test_gemini_valor_fixo_e_low():
    gem = MODELS["gemini-3.1-pro-preview"]["vision"]
    assert image_tokens(gem["image"], 4000, 3000).tokens == 1120
    assert image_tokens(gem["image"], 4000, 3000, DETAIL_LOW).tokens == 280
    assert pdf_page_tokens(gem, 612, 792).tokens == 560


def test_pdf_como_imagem_usa_dpi():
    claude = MODELS["claude-haiku-4-5"]["vision"]
    # Carta (612x792 pt) a 150 DPI = 1275x1650 px -> reduzida ao teto da camada padrão
    vc = pdf_page_tokens(claude, 612, 792, dpi=150)
    assert vc.tokens == claude_image_tokens(1275, 1650, 1568, 1568)
    assert "150 DPI" in vc.label


def test_provedores_sem_suporte_retornam_none():
    grok = MODELS["grok-4.7"]
    vc = visual_item_tokens(grok, VisualItem("a.png", "imagem", [(800, 600)]))
    assert vc.tokens is None and vc.note
    ds = MODELS["deepseek-v4-pro"]
    assert visual_item_tokens(ds, VisualItem("d.pdf", "pdf", [(612, 792)])).tokens is None


def test_pdf_valor_manual_substitui_formula():
    item = VisualItem("d.pdf", "pdf", [(612, 792)] * 3)
    vc = visual_item_tokens(MODELS["claude-opus-5-5"], item, pdf_tokens_per_page=1000)
    assert vc.tokens == 3000


def test_estimativa_soma_tokens_visuais_e_gera_linhas():
    ctx = assemble([Block("S", "system", "a" * 400)])
    items = [VisualItem("foto.png", "imagem", [(1000, 1000)]), VisualItem("doc.pdf", "pdf", [(612, 792)])]
    model = MODELS["gemini-3.1-flash-lite"]
    base = estimate(model, ctx, Settings(lang="en", use_local=False))
    est = estimate(model, ctx, Settings(lang="en", use_local=False, visual_items=items))
    assert est.visual_total == 1120 + 560
    assert est.heuristic_total == base.heuristic_total + 1680
    names = [r.name for r in est.rows]
    assert "foto.png" in names and "doc.pdf" in names
    # sem somar PDF como imagem
    est2 = estimate(model, ctx, Settings(lang="en", use_local=False, visual_items=items, pdf_as_image=False))
    assert est2.visual_total == 1120


def test_estimativa_modelo_sem_suporte_avisa():
    ctx = assemble([Block("S", "system", "texto")])
    est = estimate(MODELS["grok-4.3"], ctx,
                   Settings(visual_items=[VisualItem("foto.png", "imagem", [(800, 600)])]))
    assert est.visual_total == 0 and est.visual_notes


def test_extrai_dimensoes_da_imagem():
    PIL = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL.new("RGB", (640, 480)).save(buf, format="PNG")
    f = extract("foto.png", buf.getvalue())
    assert f.is_image and f.error is None
    assert (f.width, f.height) == (640, 480)


def test_todos_os_modelos_tem_config_visual():
    for m in MODELS.values():
        assert "vision" in m, m["id"]
        assert m["vision"]["image"]["method"] in {
            "claude_patches", "openai_patches", "fixed", "gemini_tiles", "qwen_resize", "upper_bound", "none"}
        assert m["vision"]["pdf"]["method"] in {"as_image", "fixed", "none"}


@pytest.mark.parametrize("model_id", list(MODELS))
def test_dimensoes_invalidas_nao_quebram(model_id):
    """Imagem sem dimensões (ex.: vinda de cache antigo) não pode derrubar a UI."""
    for w, h in [(0, 0), (0, 500), (500, 0)]:
        vc = visual_item_tokens(MODELS[model_id], VisualItem("x.png", "imagem", [(w, h)]))
        assert vc.tokens is None


def test_gemini_2x_blocos_exemplo_da_doc():
    from core.vision import gemini_tile_tokens
    assert gemini_tile_tokens(384, 384) == 258           # <= 384 nos dois lados
    assert gemini_tile_tokens(960, 540) == 6 * 258       # exemplo da doc: bloco de 360 px -> 3 x 2
    g25 = MODELS["gemini-2.5-pro"]["vision"]
    assert image_tokens(g25["image"], 960, 540).tokens == 1548
    assert pdf_page_tokens(g25, 612, 792).tokens == 258


def test_gemini_nao_cobra_texto_nativo_do_pdf():
    from core.vision import pdf_text_billed
    for mid, m in MODELS.items():
        assert pdf_text_billed(m) is (m["provider"] != "Google"), mid
