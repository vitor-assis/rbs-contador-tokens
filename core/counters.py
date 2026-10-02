"""Contagem de tokens: heurística, tokenizadores locais abertos e APIs de contagem.

Ordem de preferência para o número "melhor estimativa":
    API (exata, sob demanda) > tokenizador local exato > tokenizador local proxy > heurística.

Chamadas externas às APIs dos provedores só acontecem em ``count_via_api``,
que a UI chama apenas quando o usuário clica em "Contar via API".
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from .context import AssembledContext, block_text_for_count, parse_tools
from .vision import DEFAULT_PDF_DPI, DETAIL_DEFAULT, VisualItem, visual_item_tokens

# Só usamos o tokenizador: silencia avisos do transformers sobre PyTorch/arquitetura.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

LANG_LABELS = {"pt": "Português", "en": "Inglês", "misto": "Misto"}

API_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "xai": ("XAI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}
def api_enabled() -> bool:
    """A contagem via API só aparece quando há alguma chave configurada (a versão publicada não tem)."""
    return any(get_api_key(p) for p in API_KEY_ENV)


# APIs que contam a requisição completa (inclui wrappers e prompt de tools do provedor);
# nelas o overhead fixo configurado NÃO é somado.
API_COUNTS_FULL_REQUEST = {"anthropic", "openai"}

PLACEHOLDER_USER_MESSAGE = "."


# --------------------------------------------------------------------------- #
# Heurística
# --------------------------------------------------------------------------- #
def tokens_per_char(cpt_en: float, cpt_pt: float, lang: str, pt_share: float = 0.5) -> float:
    """Tokens por caractere para o idioma. No modo misto, média ponderada pela fração em PT."""
    if lang == "en":
        return 1.0 / cpt_en
    if lang == "pt":
        return 1.0 / cpt_pt
    w = min(max(pt_share, 0.0), 1.0)
    return w / cpt_pt + (1.0 - w) / cpt_en


def effective_chars_per_token(cpt_en: float, cpt_pt: float, lang: str, pt_share: float = 0.5) -> float:
    return 1.0 / tokens_per_char(cpt_en, cpt_pt, lang, pt_share)


def heuristic_tokens(text: str, cpt_en: float, cpt_pt: float, lang: str, pt_share: float = 0.5) -> int:
    """chars / chars_per_token, arredondado para cima (0 para texto vazio)."""
    if not text:
        return 0
    raw = len(text) * tokens_per_char(cpt_en, cpt_pt, lang, pt_share)
    return max(1, int(-(-raw // 1)))  # ceil


def count_words(text: str) -> int:
    return len(text.split())


# Amostras de troca de mensagens num atendimento, usadas para medir tokens por palavra
# com o tokenizador de cada modelo (converte "palavras por mensagem" em tokens).
CHAT_SAMPLES = {
    "pt": (
        "Oi, boa tarde! Meu pedido ainda não chegou e o prazo venceu ontem. "
        "Olá! Sinto muito pelo atraso. Pode me informar o número do pedido, por favor? "
        "Claro, é o 48213. Comprei uma cafeteira elétrica na promoção da semana passada. "
        "Obrigado. Verifiquei aqui: o pedido saiu do centro de distribuição na segunda-feira e está em "
        "trânsito com a transportadora. A nova previsão de entrega é quinta-feira, até as 18h. "
        "Você vai receber o código de rastreio por e-mail ainda hoje. "
        "Entendi. E se não chegar até quinta, consigo cancelar e receber o reembolso? "
        "Consegue, sim. Se a entrega não acontecer até a nova data, você pode solicitar o cancelamento "
        "pelo aplicativo, e o estorno no cartão de crédito acontece em até duas faturas. "
        "Posso ajudar com mais alguma coisa? Não, era só isso. Muito obrigado pela ajuda!"
    ),
    "en": (
        "Hi, good afternoon! My order still hasn't arrived and the delivery date was yesterday. "
        "Hello! I'm sorry about the delay. Could you share your order number, please? "
        "Sure, it's 48213. I bought an electric coffee maker during last week's sale. "
        "Thanks. I checked: the order left the distribution center on Monday and is in transit with the "
        "carrier. The new delivery estimate is Thursday, by 6 p.m. You'll get the tracking code by email "
        "today. Got it. And if it doesn't arrive by Thursday, can I cancel and get a refund? "
        "Yes, you can. If the delivery doesn't happen by the new date, you can request the cancellation in "
        "the app, and the credit card refund happens within two billing cycles. "
        "Can I help you with anything else? No, that's all. Thank you so much for your help!"
    ),
}


def tokens_per_word(model: dict, lang: str, pt_share: float = 0.5, use_local: bool = True) -> tuple[float, str]:
    """Tokens por palavra numa troca de mensagens típica, medidos com o tokenizador do modelo.

    Usa o tokenizador local (exato ou proxy) quando há; senão, a heurística. No modo misto,
    média ponderada pela parcela em português.
    """
    method, is_exact = local_method_for(model)
    label = "heurística"

    def ratio(sample_lang: str) -> float:
        nonlocal label
        text = CHAT_SAMPLES[sample_lang]
        words = count_words(text)
        if method and use_local:
            try:
                n = count_local(method, text)
                label = f"{'exato' if is_exact else 'aproximado (proxy)'}: {method_label(method)}"
                return n / words
            except LocalTokenizerError:
                pass
        n = heuristic_tokens(text, float(model["chars_per_token_en"]), float(model["chars_per_token_pt"]),
                             sample_lang)
        return n / words

    if lang == "en":
        return ratio("en"), label
    if lang == "pt":
        return ratio("pt"), label
    w = min(max(pt_share, 0.0), 1.0)
    return w * ratio("pt") + (1 - w) * ratio("en"), label


# --------------------------------------------------------------------------- #
# Métodos
# --------------------------------------------------------------------------- #
def parse_method(method: str | None) -> tuple[str, str]:
    """'tiktoken:o200k_base' -> ('tiktoken', 'o200k_base'); None/'none' -> ('none', '')."""
    if not method or method == "none":
        return "none", ""
    kind, _, arg = method.partition(":")
    return kind, arg


def is_local_method(method: str | None) -> bool:
    return parse_method(method)[0] in {"tiktoken", "hf", "genai"}


def is_api_method(method: str | None) -> bool:
    return parse_method(method)[0] == "api"


def method_label(method: str | None) -> str:
    kind, arg = parse_method(method)
    if kind == "tiktoken":
        return f"tiktoken ({arg})"
    if kind == "hf":
        return f"Hugging Face ({arg})"
    if kind == "genai":
        return f"google-genai LocalTokenizer ({arg})"
    if kind == "api":
        return f"API {arg}"
    return "nenhum"


def local_method_for(model: dict) -> tuple[str | None, bool]:
    """Tokenizador local a usar automaticamente e se ele é exato para o modelo."""
    if is_local_method(model.get("exact_method")):
        return model["exact_method"], True
    if is_local_method(model.get("local_method")):
        return model["local_method"], bool(model.get("local_method_is_exact"))
    return None, False


def api_provider_for(model: dict) -> str | None:
    kind, arg = parse_method(model.get("exact_method"))
    return arg if kind == "api" else None


# --------------------------------------------------------------------------- #
# Tokenizadores locais
# --------------------------------------------------------------------------- #
class LocalTokenizerError(RuntimeError):
    pass


_LOAD_ERRORS: dict[str, str] = {}


def clear_local_errors() -> None:
    _LOAD_ERRORS.clear()


@lru_cache(maxsize=None)
def _load_local(method: str) -> Callable[[str], int]:
    kind, arg = parse_method(method)
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(arg)
        return lambda text: len(enc.encode(text, disallowed_special=()))

    if kind == "hf":
        # 1º a biblioteca `tokenizers` (lê só o tokenizer.json: leve, ideal para o plano gratuito);
        # 2º o AutoTokenizer do transformers, se instalado (repositórios sem tokenizer.json).
        hf_token = os.getenv("HF_TOKEN") or None
        try:
            from tokenizers import Tokenizer

            raw = Tokenizer.from_pretrained(arg, token=hf_token)
            return lambda text: len(raw.encode(text, add_special_tokens=False).ids)
        except Exception as raw_err:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(arg, token=hf_token)
                tok.model_max_length = 10**12  # evita aviso de sequência longa
                return lambda text: len(tok.encode(text, add_special_tokens=False))
            except Exception as auto_err:
                raise LocalTokenizerError(
                    f"Falha ao carregar '{arg}' do Hugging Face: {raw_err} (AutoTokenizer: {auto_err})"
                ) from auto_err

    if kind == "genai":
        # Tokenizador local oficial do Google (texto): Gemma 3 (SentencePiece) no 2.x, Gemma 4 no 3.x.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from google.genai.local_tokenizer import LocalTokenizer
            except ImportError as e:
                raise LocalTokenizerError(
                    "Instale o extra do google-genai: pip install \"google-genai[local-tokenizer]\""
                ) from e
            tok = LocalTokenizer(model_name=arg)

        def _count(text: str) -> int:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return int(tok.count_tokens(text).total_tokens or 0)

        return _count

    raise LocalTokenizerError(f"Método local desconhecido: {method}")


def get_local_counter(method: str) -> Callable[[str], int]:
    """Carrega (com cache) o tokenizador. Falhas ficam memorizadas para não repetir downloads."""
    if method in _LOAD_ERRORS:
        raise LocalTokenizerError(_LOAD_ERRORS[method])
    try:
        return _load_local(method)
    except Exception as e:
        _LOAD_ERRORS[method] = str(e)
        raise LocalTokenizerError(str(e)) from e


_COUNT_MEMO: dict[tuple[str, str], int] = {}
_COUNT_MEMO_MAX = 5000


def count_local(method: str, text: str) -> int:
    """Conta com o tokenizador local, memorizando pelo hash do texto (a UI reexecuta a cada edição)."""
    if not text:
        return 0
    key = (method, hashlib.sha1(text.encode("utf-8")).hexdigest())
    if key not in _COUNT_MEMO:
        if len(_COUNT_MEMO) >= _COUNT_MEMO_MAX:
            _COUNT_MEMO.clear()
        _COUNT_MEMO[key] = get_local_counter(method)(text)
    return _COUNT_MEMO[key]


# --------------------------------------------------------------------------- #
# APIs de contagem (somente sob demanda)
# --------------------------------------------------------------------------- #
class ApiCountError(RuntimeError):
    pass


class MissingKeyError(ApiCountError):
    pass


def get_api_key(provider: str) -> str | None:
    for var in API_KEY_ENV.get(provider, ()):
        val = os.getenv(var)
        if val:
            return val
    return None


API_ENDPOINTS = {
    "anthropic": "POST https://api.anthropic.com/v1/messages/count_tokens",
    "openai": "POST https://api.openai.com/v1/responses/input_tokens",
    "gemini": "google-genai: client.models.count_tokens (POST .../models/{modelo}:countTokens)",
    "xai": "POST https://api.x.ai/v1/tokenize-text",
}


def api_payload(provider: str, model_id: str, system_text: str, tools: list[dict] | None,
                full_text: str) -> dict:
    """Corpo exato da requisição de contagem de cada provedor (também exibido na pré-visualização)."""
    if provider == "anthropic":
        payload: dict = {"model": model_id}
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        payload["messages"] = [{"role": "user", "content": PLACEHOLDER_USER_MESSAGE}]
        return payload
    if provider == "openai":
        payload = {"model": model_id}
        if system_text:
            payload["instructions"] = system_text
        if tools:
            payload["tools"] = [
                {"type": "function", "name": t["name"], "description": t["description"],
                 "parameters": t["parameters"]}
                for t in tools
            ]
        payload["input"] = PLACEHOLDER_USER_MESSAGE
        return payload
    if provider == "gemini":
        return {"model": model_id, "contents": full_text}
    if provider == "xai":
        return {"model": model_id, "text": full_text}
    raise ApiCountError(f"Provedor de API desconhecido: {provider}")


def _anthropic_count(payload: dict, key: str) -> int:
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    try:
        return client.messages.count_tokens(**payload).input_tokens
    except anthropic.AuthenticationError as e:
        raise ApiCountError("Anthropic: chave de API inválida.") from e
    except anthropic.NotFoundError as e:
        raise ApiCountError(f"Anthropic: modelo '{payload['model']}' não encontrado.") from e
    except anthropic.RateLimitError as e:
        raise ApiCountError("Anthropic: limite de requisições atingido; tente de novo em instantes.") from e
    except anthropic.APIStatusError as e:
        raise ApiCountError(f"Anthropic: erro {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise ApiCountError("Anthropic: falha de conexão.") from e


def _gemini_count(payload: dict, key: str) -> int:
    from google import genai

    client = genai.Client(api_key=key)
    try:
        resp = client.models.count_tokens(model=payload["model"], contents=payload["contents"])
    except Exception as e:
        raise ApiCountError(f"Gemini: {e}") from e
    return int(resp.total_tokens or 0)


def _http_post(url: str, key: str, payload: dict, provider: str) -> dict:
    import httpx

    try:
        r = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60)
    except httpx.HTTPError as e:
        raise ApiCountError(f"{provider}: falha de conexão ({e}).") from e
    if r.status_code == 401:
        raise ApiCountError(f"{provider}: chave de API inválida.")
    if r.status_code >= 400:
        raise ApiCountError(f"{provider}: erro {r.status_code}: {r.text[:300]}")
    return r.json()


def _api_count_request(provider: str, model_id: str, key: str, system_text: str,
                       tools: list[dict] | None, full_text: str) -> int:
    if provider in ("gemini", "xai") and not full_text:
        return 0
    payload = api_payload(provider, model_id, system_text, tools, full_text)
    if provider == "anthropic":
        return _anthropic_count(payload, key)
    if provider == "openai":
        data = _http_post("https://api.openai.com/v1/responses/input_tokens", key, payload, "OpenAI")
        return int(data["input_tokens"])
    if provider == "gemini":
        return _gemini_count(payload, key)
    data = _http_post("https://api.x.ai/v1/tokenize-text", key, payload, "xAI")
    return len(data.get("token_ids", []))


def count_via_api(model: dict, ctx: AssembledContext, per_block: bool = True,
                  wrap_files: bool = True) -> dict:
    """Conta o contexto montado (e opcionalmente cada bloco) via API do provedor.

    Retorna {"total": int, "blocks": [int|None, ...], "label": str, "includes_wrappers": bool}.
    Para Anthropic/OpenAI, os blocos isolados descontam a linha de base
    (requisição só com a mensagem mínima), para refletir apenas o bloco.
    """
    provider = api_provider_for(model)
    if not provider:
        raise ApiCountError("Este modelo não tem contagem via API configurada.")
    key = get_api_key(provider)
    if not key:
        raise MissingKeyError(
            f"Sem chave para {provider}: defina {' ou '.join(API_KEY_ENV[provider])} no .env."
        )
    mid = model["id"]
    total = _api_count_request(provider, mid, key, ctx.system_text, ctx.tools, ctx.full_text)

    blocks: list[int | None] = []
    if per_block:
        full_request = provider in API_COUNTS_FULL_REQUEST
        baseline = _api_count_request(provider, mid, key, "", None, "") if full_request else 0
        for b in ctx.blocks:
            text = block_text_for_count(b, wrap_files)
            if b.type == "tools":
                parsed = parse_tools(b.text)
                if parsed and full_request:
                    n = _api_count_request(provider, mid, key, "", parsed, "")
                else:
                    n = _api_count_request(provider, mid, key, text, None, text)
            else:
                n = _api_count_request(provider, mid, key, text, None, text)
            blocks.append(max(0, n - baseline))
    return {
        "total": total,
        "blocks": blocks,
        "label": f"exato (API {provider})",
        "includes_wrappers": provider in API_COUNTS_FULL_REQUEST,
    }


def context_fingerprint(model_id: str, ctx: AssembledContext, per_block: bool) -> str:
    """Hash do que foi enviado à API, para saber se um resultado salvo ainda vale."""
    payload = json.dumps(
        {
            "m": model_id,
            "s": ctx.system_text,
            "t": ctx.tools,
            "f": ctx.full_text,
            "b": [(b.name, b.type, b.text) for b in ctx.blocks],
            "p": per_block,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Estimativa por modelo
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    lang: str = "pt"
    pt_share: float = 0.5
    overhead_tokens: int = 0
    wrap_files: bool = True
    use_local: bool = True
    # Visual: imagens soltas e PDFs lidos nativamente (páginas cobradas como imagem)
    visual_items: list[VisualItem] = field(default_factory=list)
    vision_detail: str = DETAIL_DEFAULT
    pdf_as_image: bool = True
    pdf_dpi: int = DEFAULT_PDF_DPI
    pdf_tokens_per_page: int = 0  # > 0 substitui a fórmula do provedor


@dataclass
class BlockRow:
    name: str
    type_label: str
    chars: int
    words: int
    heuristic: int
    exact: int | None
    exact_kind: str | None  # "exato" | "aproximado" | None


@dataclass
class ModelEstimate:
    model: dict
    heuristic_total: int
    heuristic_sum: int
    local_total: int | None = None
    local_sum: int | None = None
    local_label: str | None = None
    local_is_exact: bool = False
    local_error: str | None = None
    api_total: int | None = None
    api_sum: int | None = None
    api_label: str | None = None
    rows: list[BlockRow] = field(default_factory=list)
    calibration: dict | None = None
    visual_total: int = 0
    visual_notes: list[str] = field(default_factory=list)

    @property
    def best_total(self) -> int:
        if self.api_total is not None:
            return self.api_total
        if self.local_total is not None:
            return self.local_total
        return self.heuristic_total

    @property
    def best_sum(self) -> int:
        if self.api_sum is not None:
            return self.api_sum
        if self.local_sum is not None:
            return self.local_sum
        return self.heuristic_sum

    @property
    def best_label(self) -> str:
        if self.api_total is not None:
            return self.api_label or "exato (API)"
        if self.local_total is not None:
            return self.local_label or "local"
        return "estimado (heurística)"

    @property
    def exact_total(self) -> int | None:
        """Total por método exato (API ou local exato); None se só houver proxy/heurística."""
        if self.api_total is not None:
            return self.api_total
        if self.local_total is not None and self.local_is_exact:
            return self.local_total
        return None

    @property
    def window_pct(self) -> float:
        cw = self.model.get("context_window") or 0
        return (self.best_total / cw * 100.0) if cw else 0.0


def estimate(model: dict, ctx: AssembledContext, settings: Settings,
             api_result: dict | None = None) -> ModelEstimate:
    cpt_en = float(model["chars_per_token_en"])
    cpt_pt = float(model["chars_per_token_pt"])

    def heur(t: str) -> int:
        return heuristic_tokens(t, cpt_en, cpt_pt, settings.lang, settings.pt_share)

    texts = [block_text_for_count(b, settings.wrap_files) for b in ctx.blocks]
    h_blocks = [heur(t) for t in texts]

    # Tokens visuais: fórmula do provedor a partir das dimensões (iguais em todos os métodos)
    visual_rows: list[BlockRow] = []
    visual_notes: list[str] = []
    for item in settings.visual_items:
        if item.kind == "pdf" and not settings.pdf_as_image:
            continue
        vc = visual_item_tokens(model, item, settings.vision_detail, settings.pdf_dpi,
                                settings.pdf_tokens_per_page)
        if vc.tokens is None:
            visual_notes.append(f"{item.name}: {vc.label} ({vc.note})")
            continue
        type_label = "Imagem" if item.kind == "imagem" else f"PDF: {len(item.sizes)} pág. como imagem"
        visual_rows.append(BlockRow(item.name, type_label, 0, 0, vc.tokens, vc.tokens, vc.label))
        if vc.note:
            visual_notes.append(f"{item.name}: {vc.note}")
    visual = sum(r.heuristic for r in visual_rows)
    extra = settings.overhead_tokens + visual

    est = ModelEstimate(
        model=model,
        heuristic_total=heur(ctx.full_text) + extra,
        heuristic_sum=sum(h_blocks) + extra,
        visual_total=visual,
        visual_notes=visual_notes,
    )

    # Tokenizador local (automático)
    l_blocks: list[int] | None = None
    method, is_exact = local_method_for(model)
    if method and settings.use_local:
        try:
            l_blocks = [count_local(method, t) for t in texts]
            est.local_total = count_local(method, ctx.full_text) + extra
            est.local_sum = sum(l_blocks) + extra
            est.local_is_exact = is_exact
            est.local_label = f"{'exato' if is_exact else 'aproximado (proxy)'}: {method_label(method)}"
        except LocalTokenizerError as e:
            est.local_error = str(e)
            l_blocks = None

    # API (resultado obtido sob demanda pela UI)
    a_blocks: list[int | None] | None = None
    if api_result:
        # a contagem via API cobre só o texto; o visual vem da fórmula
        api_extra = visual + (0 if api_result.get("includes_wrappers") else settings.overhead_tokens)
        est.api_total = api_result["total"] + api_extra
        est.api_label = api_result.get("label", "exato (API)")
        if api_result.get("blocks") and len(api_result["blocks"]) == len(ctx.blocks):
            a_blocks = api_result["blocks"]
            est.api_sum = sum(x or 0 for x in a_blocks) + api_extra

    # Linhas da tabela
    for i, (b, t) in enumerate(zip(ctx.blocks, texts)):
        exact, kind = None, None
        if a_blocks is not None and a_blocks[i] is not None:
            exact, kind = a_blocks[i], "exato"
        elif l_blocks is not None:
            exact, kind = l_blocks[i], ("exato" if is_exact else "aproximado")
        est.rows.append(BlockRow(b.name, b.type_label, len(t), count_words(t), h_blocks[i], exact, kind))
    has_exact_col = any(r.exact is not None for r in est.rows)
    est.rows.extend(visual_rows)
    if settings.overhead_tokens:
        wrapped = bool(api_result and api_result.get("includes_wrappers"))
        est.rows.append(BlockRow("Overhead fixo", "Overhead", 0, 0, settings.overhead_tokens,
                                 (0 if wrapped else settings.overhead_tokens) if has_exact_col else None,
                                 None))

    # Calibração: chars/token reais observados no contexto montado
    chars, obs_tokens, source, exact_src = len(ctx.full_text), None, None, False
    if a_blocks is not None:
        # blocos isolados, já sem a linha de base da requisição
        chars = sum(len(t) for t in texts)
        obs_tokens, source, exact_src = sum(x or 0 for x in a_blocks), est.api_label, True
    elif api_result:
        obs_tokens, source, exact_src = api_result["total"], est.api_label, True
        if api_result.get("includes_wrappers"):
            source = f"{source} (inclui wrappers da requisição)"
    elif est.local_total is not None:
        obs_tokens, source, exact_src = est.local_total - extra, est.local_label, est.local_is_exact
    if chars and obs_tokens:
        est.calibration = {
            "chars": chars,
            "tokens": obs_tokens,
            "observed_chars_per_token": chars / obs_tokens,
            "configured_chars_per_token": effective_chars_per_token(cpt_en, cpt_pt, settings.lang, settings.pt_share),
            "source": source,
            "is_exact": exact_src,
        }
    return est

