"""Tokens de imagens soltas e de páginas de PDF lidas nativamente.

O custo visual não depende de um tokenizador: cada provedor converte a imagem em
"patches" (ou cobra um valor fixo) segundo uma regra publicada. Aqui ficam essas
regras, parametrizadas pelo bloco ``vision`` de cada modelo no models.json.

Métodos de imagem (``vision.image.method``):
    claude_patches   ceil(w/28) x ceil(h/28), após reduzir para caber em max_edge/max_tokens
    openai_patches   patches de 32px (com teto por nível de detalhe) x multiplicador
    fixed            valor fixo por imagem (Gemini 3: media_resolution)
    gemini_tiles     Gemini 2.x: 258 tokens se <= 384 px; senão blocos (crop unit) de até 768 px x 258
    qwen_resize      smart_resize em múltiplos de 32px + 2 tokens especiais
    upper_bound      só há um teto documentado (DeepSeek); usa o teto
    none             não suportado ou não documentado

``vision.pdf.text_billed`` (padrão true): false quando o provedor não cobra o texto nativo
extraído do PDF (Gemini), só as páginas.

Métodos de página de PDF (``vision.pdf.method``):
    as_image         a página é rasterizada (DPI suposto) e cobrada como imagem
    fixed            valor fixo por página (Gemini)
    none             o provedor não lê PDF nativamente: só o texto extraído conta
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

DETAIL_DEFAULT = "padrao"
DETAIL_LOW = "economico"
DETAIL_LABELS = {DETAIL_DEFAULT: "Padrão do provedor", DETAIL_LOW: "Econômico (low)"}

DEFAULT_PDF_DPI = 150


@dataclass
class VisualItem:
    """Uma imagem (1 tamanho em px) ou um PDF (tamanhos das páginas em pontos)."""
    name: str
    kind: str  # "imagem" | "pdf"
    sizes: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class VisualCount:
    tokens: int | None
    label: str
    note: str | None = None


# --------------------------------------------------------------------------- #
# Fórmulas por provedor
# --------------------------------------------------------------------------- #
def claude_resized_size(width: int, height: int, max_edge: int = 1568, max_tokens: int = 1568) -> tuple[int, int]:
    """Implementação de referência da Anthropic (vision-coordinates): maior tamanho
    com a mesma proporção que respeita o limite de borda e o de tokens visuais."""

    def fits(w: int, h: int) -> bool:
        return (math.ceil(w / 28) * 28 <= max_edge and math.ceil(h / 28) * 28 <= max_edge
                and math.ceil(w / 28) * math.ceil(h / 28) <= max_tokens)

    if fits(width, height):
        return width, height
    if height > width:
        rh, rw = claude_resized_size(height, width, max_edge, max_tokens)
        return rw, rh
    aspect = width / height
    lo, hi = 1, width
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect), 1)


def claude_image_tokens(width: int, height: int, max_edge: int = 1568, max_tokens: int = 1568) -> int:
    w, h = claude_resized_size(width, height, max_edge, max_tokens)
    return math.ceil(w / 28) * math.ceil(h / 28)


def _openai_shrink_to_patches(w: float, h: float, max_patches: int, patch: int = 32) -> tuple[int, int]:
    """Passo B da doc da OpenAI: reduz a imagem para caber no orçamento de patches."""
    shrink = math.sqrt(max_patches * patch * patch / (w * h))
    adj = shrink * min(
        math.floor(w * shrink / patch) / (w * shrink / patch),
        math.floor(h * shrink / patch) / (h * shrink / patch),
    )
    return max(1, math.floor(w * adj)), max(1, math.floor(h * adj))


def openai_image_tokens(width: int, height: int, multiplier: float = 1.2, detail: str = "auto",
                        high_max_patches: int = 2500, low_box: int = 512, max_dim: int = 65535,
                        patch: int = 32) -> tuple[int, str | None]:
    """Patches de 32px x multiplicador. Retorna (tokens, aviso)."""
    w, h = float(width), float(height)
    note = None
    if max(w, h) > max_dim:  # todos os níveis limitam a dimensão máxima
        s = max_dim / max(w, h)
        w, h = w * s, h * s
    if detail == "low":
        if max(w, h) > low_box:
            s = low_box / max(w, h)
            w, h = max(1.0, math.floor(w * s)), max(1.0, math.floor(h * s))
    elif detail == "high":
        if math.ceil(w / patch) * math.ceil(h / patch) > high_max_patches:
            w, h = _openai_shrink_to_patches(w, h, high_max_patches, patch)
    patches = math.ceil(w / patch) * math.ceil(h / patch)
    if patches > 30_000:
        note = "acima de 30.000 patches: a API rejeita esta imagem nesse nível de detalhe"
    return math.ceil(patches * multiplier), note


def gemini_tile_tokens(width: int, height: int, tokens_per_tile: int = 258, small_edge: int = 384,
                       max_unit: int = 768, min_unit: int = 256) -> int:
    """Gemini 2.x (doc de image-understanding): imagens com os dois lados <= 384 px custam 258;
    as maiores são divididas em blocos de ``floor(min(w, h) / 1.5)`` px (até 768), 258 cada.
    O piso de 256 px no bloco é suposto (evita explosão em imagens muito finas)."""
    if width <= small_edge and height <= small_edge:
        return tokens_per_tile
    unit = min(max_unit, max(min_unit, math.floor(min(width, height) / 1.5)))
    return math.ceil(width / unit) * math.ceil(height / unit) * tokens_per_tile


def qwen_image_tokens(width: int, height: int, factor: int = 32, min_tokens: int = 4,
                      max_tokens: int = 1280, extra_tokens: int = 2) -> int:
    """smart_resize (mesma regra do processador Qwen-VL): lados em múltiplos de ``factor``,
    área entre min_tokens e max_tokens patches; cada patch = 1 token, + tokens especiais."""
    min_px, max_px = min_tokens * factor * factor, max_tokens * factor * factor
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_px:
        beta = math.sqrt(height * width / max_px)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar // factor) * (w_bar // factor) + extra_tokens


# --------------------------------------------------------------------------- #
# Despacho pela configuração do modelo
# --------------------------------------------------------------------------- #
def image_tokens(cfg: dict | None, width: int, height: int, detail: str = DETAIL_DEFAULT) -> VisualCount:
    cfg = cfg or {"method": "none"}
    method = cfg.get("method", "none")
    low = detail == DETAIL_LOW
    if method != "none" and (width <= 0 or height <= 0):
        return VisualCount(None, "não estimado", "dimensões da imagem desconhecidas")

    if method == "claude_patches":
        t = claude_image_tokens(width, height, cfg.get("max_edge", 1568), cfg.get("max_tokens", 1568))
        return VisualCount(t, f"fórmula Anthropic (28px, até {cfg.get('max_tokens', 1568)})")
    if method == "openai_patches":
        det = cfg.get("detail_low", "low") if low else cfg.get("detail_default", "auto")
        t, note = openai_image_tokens(width, height, cfg.get("multiplier", 1.2), det,
                                      cfg.get("high_max_patches", 2500), cfg.get("low_box", 512))
        return VisualCount(t, f"fórmula OpenAI (32px ×{cfg.get('multiplier', 1.2)}, detail={det})", note)
    if method == "fixed":
        t = cfg.get("tokens_low", cfg["tokens"]) if low else cfg["tokens"]
        return VisualCount(int(t), f"valor fixo por imagem ({'low' if low else 'padrão'})")
    if method == "gemini_tiles":
        t = gemini_tile_tokens(width, height, cfg.get("tokens_per_tile", 258))
        return VisualCount(t, "blocos Gemini 2.x (258 por bloco de até 768 px)")
    if method == "qwen_resize":
        t = qwen_image_tokens(width, height, cfg.get("factor", 32), cfg.get("min_tokens", 4),
                              cfg.get("max_tokens", 1280), cfg.get("extra_tokens", 2))
        return VisualCount(t, f"smart_resize Qwen (32px, até {cfg.get('max_tokens', 1280)})")
    if method == "upper_bound":
        return VisualCount(int(cfg["tokens"]), "teto documentado por imagem",
                           "só há um teto publicado; a estimativa usa o pior caso")
    return VisualCount(None, "não estimado", cfg.get("reason", "custo de imagem não documentado"))


def pdf_page_tokens(vision: dict | None, page_w_pt: float, page_h_pt: float,
                    detail: str = DETAIL_DEFAULT, dpi: int = DEFAULT_PDF_DPI) -> VisualCount:
    vision = vision or {}
    cfg = vision.get("pdf") or {"method": "none"}
    method = cfg.get("method", "none")
    if method == "fixed":
        t = cfg.get("tokens_low", cfg["tokens"]) if detail == DETAIL_LOW else cfg["tokens"]
        return VisualCount(int(t), "valor fixo por página")
    if method == "as_image":
        w = max(1, round(page_w_pt / 72 * dpi))
        h = max(1, round(page_h_pt / 72 * dpi))
        vc = image_tokens(vision.get("image"), w, h, detail)
        vc.label = f"página rasterizada a {dpi} DPI (suposto) + {vc.label}"
        return vc
    return VisualCount(None, "não lê PDF nativamente", cfg.get("reason", "só o texto extraído conta"))


def pdf_text_billed(model: dict) -> bool:
    """False quando o provedor lê o PDF nativamente e não cobra o texto extraído (Gemini)."""
    return bool(((model.get("vision") or {}).get("pdf") or {}).get("text_billed", True))


def visual_item_tokens(model: dict, item: VisualItem, detail: str = DETAIL_DEFAULT,
                       pdf_dpi: int = DEFAULT_PDF_DPI, pdf_tokens_per_page: int = 0) -> VisualCount:
    """Tokens visuais de um item para um modelo. ``pdf_tokens_per_page`` > 0 substitui a fórmula."""
    vision = model.get("vision") or {}
    if item.kind == "imagem":
        w, h = item.sizes[0]
        return image_tokens(vision.get("image"), int(w), int(h), detail)

    pdf_cfg = (vision.get("pdf") or {}).get("method", "none")
    if pdf_cfg == "none":
        return pdf_page_tokens(vision, 0, 0, detail, pdf_dpi)
    if pdf_tokens_per_page > 0:
        return VisualCount(pdf_tokens_per_page * len(item.sizes),
                           f"{pdf_tokens_per_page} tokens/página (valor manual)")
    total, label, notes = 0, "", set()
    for pw, ph in item.sizes:
        vc = pdf_page_tokens(vision, pw, ph, detail, pdf_dpi)
        if vc.tokens is None:
            return vc
        total += vc.tokens
        label = vc.label
        if vc.note:
            notes.add(vc.note)
    return VisualCount(total, label, "; ".join(sorted(notes)) or None)
