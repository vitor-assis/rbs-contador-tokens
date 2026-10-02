"""Estimador de Tokens de Contexto: UI em Streamlit.

Rode com:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.context import (
    ALL_TYPE_LABELS,
    INTERNAL_BLOCK_TYPES,
    FILE_BLOCK_TYPE,
    PROMPT_BLOCK_TYPES,
    SEPARATOR,
    AssembledContext,
    Block,
    assemble,
    block_text_for_count,
    minify_json,
    validate_json,
)
from core.counters import (
    API_ENDPOINTS,
    API_KEY_ENV,
    LANG_LABELS,
    ApiCountError,
    LocalTokenizerError,
    ModelEstimate,
    Settings,
    api_enabled,
    api_payload,
    api_provider_for,
    clear_local_errors,
    context_fingerprint,
    count_local,
    count_via_api,
    estimate,
    get_api_key,
    heuristic_tokens,
    local_method_for,
    method_label,
    tokens_per_word,
)
from core.extractors import EXTRACTOR_VERSION, IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, extract
from core.vision import (
    DEFAULT_PDF_DPI,
    DETAIL_DEFAULT,
    DETAIL_LABELS,
    VisualItem,
    image_tokens,
    pdf_text_billed,
    visual_item_tokens,
)
from core.conversation import ConversationEstimate, ConversationParams, simulate_conversation
from core.pricing import format_usd, load_models

load_dotenv()

st.set_page_config(page_title="Estimador de Tokens de Contexto", page_icon="🧮", layout="wide")

COMPARE = "__comparar__"
WARN_PCT, CRIT_PCT = 50.0, 80.0

MODELS = load_models()
API_ON = api_enabled()  # sem nenhuma chave (versão publicada): só tokenizadores locais e heurística
MODELS_BY_ID = {m["id"]: m for m in MODELS}


def fmt_int(n: int | None) -> str:
    return "—" if n is None else f"{n:,}".replace(",", ".")


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%".replace(".", ",")


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
ss = st.session_state


def _new_id() -> int:
    ss.next_id = ss.get("next_id", 0) + 1
    return ss.next_id


def _add(list_name: str, block_type: str, name: str | None = None, text: str = "") -> None:
    bid = _new_id()
    n = len(ss[list_name]) + 1
    ss[f"name_{bid}"] = name or (f"Bloco {n}" if list_name == "blocks" else f"Prompt interno {n}")
    ss[f"type_{bid}"] = block_type
    ss[f"text_{bid}"] = text
    ss[list_name].append(bid)


def _remove(list_name: str, bid: int) -> None:
    ss[list_name] = [b for b in ss[list_name] if b != bid]
    for k in ("name", "type", "text"):
        ss.pop(f"{k}_{bid}", None)


def _duplicate(list_name: str, bid: int) -> None:
    nid = _new_id()
    ss[f"name_{nid}"] = f"{ss.get(f'name_{bid}', '')} (cópia)"
    ss[f"type_{nid}"] = ss.get(f"type_{bid}")
    ss[f"text_{nid}"] = ss.get(f"text_{bid}", "")
    lst = ss[list_name]
    lst.insert(lst.index(bid) + 1, nid)


def _move(list_name: str, bid: int, delta: int) -> None:
    lst = ss[list_name]
    i = lst.index(bid)
    j = i + delta
    if 0 <= j < len(lst):
        lst[i], lst[j] = lst[j], lst[i]


def _minify(bid: int) -> None:
    try:
        ss[f"text_{bid}"] = minify_json(ss.get(f"text_{bid}", ""))
    except ValueError:
        pass


if "blocks" not in ss:
    ss.blocks, ss.internos = [], []
    ss.api_results = {}
    _add("blocks", "system", name="System prompt")


# --------------------------------------------------------------------------- #
# Barra lateral: modelo e configurações
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Modelo")
    model_choice = st.selectbox(
        "Modelo",
        [m["id"] for m in MODELS] + [COMPARE],
        format_func=lambda i: "Comparar modelos…" if i == COMPARE
        else f"{MODELS_BY_ID[i]['provider']} · {MODELS_BY_ID[i]['label']}",
        label_visibility="collapsed",
        key="modelo",
    )
    compare_mode = model_choice == COMPARE
    selected_models: list[dict] = []
    if compare_mode:
        if "cmp_models" not in ss:  # padrão: o primeiro modelo de cada provedor
            first_by_provider: dict[str, str] = {}
            for m in MODELS:
                first_by_provider.setdefault(m["provider"], m["id"])
            ss.cmp_models = list(first_by_provider.values())
        st.multiselect(
            "Modelos para comparar", [m["id"] for m in MODELS], key="cmp_models",
            format_func=lambda i: f"{MODELS_BY_ID[i]['provider']} · {MODELS_BY_ID[i]['label']}",
            placeholder="Escolha os modelos",
        )
        providers = list(dict.fromkeys(m["provider"] for m in MODELS))

        def _select_all():
            ss.cmp_models = [m["id"] for m in MODELS]

        def _select_none():
            ss.cmp_models = []

        def _add_provider():
            prov = ss.get("cmp_provider")
            ss.cmp_models = list(dict.fromkeys(ss.cmp_models + [m["id"] for m in MODELS if m["provider"] == prov]))

        b1, b2 = st.columns(2)
        b1.button("Todos", on_click=_select_all, width="stretch")
        b2.button("Nenhum", on_click=_select_none, width="stretch")
        p1, p2 = st.columns([3, 2])
        p1.selectbox("Provedor", providers, key="cmp_provider", label_visibility="collapsed")
        p2.button("＋ Provedor", on_click=_add_provider, width="stretch",
                  help="Adiciona todos os modelos do provedor escolhido.")
        selected_models = [MODELS_BY_ID[i] for i in ss.cmp_models if i in MODELS_BY_ID]
    if compare_mode:
        ref_model = selected_models[0] if selected_models else MODELS[0]
    else:
        ref_model = MODELS_BY_ID[model_choice]

    if not compare_mode:
        m = ref_model
        local, local_exact = local_method_for(m)
        with st.expander("Detalhes do modelo", expanded=False):
            st.markdown(
                f"- **ID:** `{m['id']}`\n"
                f"- **Janela de contexto:** {fmt_int(m['context_window'])} tokens\n"
                f"- **Método exato:** {method_label(m['exact_method'])}\n"
                f"- **Tokenizador local:** "
                f"{method_label(local) + (' (exato)' if local_exact else ' (proxy)') if local else 'nenhum'}\n"
                f"- **Preço entrada:** US$ {m['price_input_per_1m']}/1M · "
                f"cache leitura US$ {m['price_cache_read_per_1m']}/1M · "
                f"cache escrita US$ {m['price_cache_write_per_1m']}/1M\n"
                f"- **Preços conferidos em:** {m.get('price_checked_at', '?')}"
            )
            if m.get("notes"):
                st.caption(m["notes"])
            vis = m.get("vision") or {}
            if vis:
                img = image_tokens(vis.get("image"), 1000, 1000)
                st.caption(
                    f"Visual: {img.label}"
                    + (f" (1000×1000 px ≈ {fmt_int(img.tokens)} tokens)" if img.tokens else f" ({img.note})")
                    + ". " + (vis.get("notes") or "")
                )

    st.header("Configurações")
    lang = st.radio(
        "Idioma predominante", list(LANG_LABELS), format_func=LANG_LABELS.get, horizontal=True,
        help="Usado apenas na heurística (caracteres por token).",
    )
    pt_share = 0.5
    if lang == "misto":
        pt_share = st.slider("Parcela em português", 0, 100, 50, 5, format="%d%%") / 100
    overhead = st.number_input(
        "Overhead fixo (tokens)", min_value=0, value=0, step=10,
        help="Wrappers de mensagem e prompt de sistema que o provedor injeta ao usar tools. "
             "Não é somado às contagens via API Anthropic/OpenAI, que já incluem isso.",
    )

    st.header("Conversa")
    conv_month = st.number_input("Conversas por mês", min_value=0, value=1000, step=100)
    turns = st.number_input(
        "Mensagens do usuário por conversa", min_value=1, value=5, step=1,
        help="Turnos da conversa. 1 turno = mensagem do usuário → raciocínio do agente → resposta do agente. "
             "Uma conversa com 10 mensagens no total (5 do usuário + 5 do agente) tem 5 turnos.",
    )
    user_words = st.slider(
        "Palavras por mensagem do usuário", 1, 100, (5, 15),
        help="Faixa de tamanho (só texto). A simulação usa a média.",
    )
    response_words = st.number_input(
        "Palavras por resposta do agente", min_value=1, value=80, step=10,
        help="Tamanho médio da resposta visível ao usuário.",
    )
    thinking_tokens = st.number_input(
        "Raciocínio por resposta (tokens)", min_value=0, value=500, step=100,
        help="Tokens de 'thinking' gerados antes de cada resposta. São cobrados como saída. O Gemini 3.x e o "
             "2.5 Pro/Flash raciocinam por padrão; o volume varia muito com a tarefa e o nível de raciocínio. "
             "É a maior incerteza da simulação: meça o campo de tokens de raciocínio (thoughts) no uso real.",
    )
    thoughts_in_history = st.toggle(
        "Raciocínio anterior volta como entrada (pior caso)", value=False,
        help="Desligado: só mensagens e respostas acumulam no histórico; do raciocínio anterior volta apenas "
             "uma assinatura criptografada, cujo custo a doc não detalha. Ligue para o pior caso.",
    )
    cache_on = st.toggle("Cache implícito (prompt caching)", value=True,
                         help="O Gemini 2.5+ usa cache implícito por padrão: a parte repetida da requisição "
                              "anterior é cobrada a ~10% do preço de entrada.")
    hit_rate = 0.95
    if cache_on:
        hit_rate = st.slider(
            "Taxa de acerto do cache", 0, 100, 95, 5, format="%d%%",
            help="Fração do prefixo repetido que é efetivamente lida do cache (o Google não garante acerto).",
        ) / 100

    with st.expander("Avançado"):
        use_local = st.toggle(
            "Usar tokenizadores locais", value=True,
            help="Rodam localmente; o tokenizador é baixado na 1ª vez e fica em cache.",
        )
        wrap_files = st.toggle("Envolver arquivos em <documento>", value=True)
        api_per_block = st.toggle(
            "Via API: contar também cada bloco", value=True,
            help="Faz uma chamada extra por bloco. Desligue para uma única chamada.",
        ) if API_ON else False
        if st.button("Tentar recarregar tokenizadores"):
            clear_local_errors()

    if API_ON:
        st.header("Chaves de API")
        for prov, envs in API_KEY_ENV.items():
            ok = bool(get_api_key(prov))
            st.caption(f"{'✅' if ok else '⬜'} {prov} ({' / '.join(envs)})")
        st.caption("As chaves ficam no arquivo .env. Sem chave, o app usa tokenizador local ou heurística.")
    else:
        st.caption("Contagem por tokenizadores locais abertos (Hugging Face / google-genai) e heurística. "
                   "Nenhum conteúdo é enviado a provedores de IA.")


# --------------------------------------------------------------------------- #
# Contagem rápida (para economia da minificação)
# --------------------------------------------------------------------------- #
def quick_count(text: str) -> tuple[int, str]:
    method, _ = local_method_for(ref_model)
    if method and use_local:
        try:
            return count_local(method, text), method_label(method)
        except LocalTokenizerError:
            pass
    return (
        heuristic_tokens(text, ref_model["chars_per_token_en"], ref_model["chars_per_token_pt"], lang, pt_share),
        "heurística",
    )


# --------------------------------------------------------------------------- #
# Cabeçalho e entradas
# --------------------------------------------------------------------------- #
st.title("Estimador de Tokens de Contexto")
st.caption(
    "Quantos tokens o contexto inicial do seu assistente consome antes da primeira mensagem do usuário: "
    "system prompt, instruções, informações extras e base de conhecimento."
)


def render_block_list(list_name: str, type_options: dict[str, str]) -> None:
    ids = list(ss[list_name])
    for pos, bid in enumerate(ids):
        with st.container(border=True):
            c_name, c_type, c_up, c_down, c_dup, c_del = st.columns([4, 3, 0.6, 0.6, 1.1, 1.1])
            c_name.text_input("Nome", key=f"name_{bid}")
            c_type.selectbox("Tipo", list(type_options), format_func=type_options.get, key=f"type_{bid}")
            c_up.button("↑", key=f"up_{bid}", help="Mover para cima", disabled=pos == 0,
                        on_click=_move, args=(list_name, bid, -1), width="stretch")
            c_down.button("↓", key=f"down_{bid}", help="Mover para baixo", disabled=pos == len(ids) - 1,
                          on_click=_move, args=(list_name, bid, 1), width="stretch")
            c_dup.button("Duplicar", key=f"dup_{bid}", on_click=_duplicate, args=(list_name, bid),
                         width="stretch")
            c_del.button("Remover", key=f"del_{bid}", on_click=_remove, args=(list_name, bid),
                         width="stretch")
            is_json = ss.get(f"type_{bid}") == "interno_json"
            st.text_area(
                "Conteúdo", key=f"text_{bid}", height=160, label_visibility="collapsed",
                placeholder='{"chave": "valor"}' if is_json else "Cole o texto aqui…",
            )
            text = ss.get(f"text_{bid}", "")
            st.caption(f"{fmt_int(len(text))} caracteres · {fmt_int(len(text.split()))} palavras")
            if is_json and text.strip():
                ok, msg = validate_json(text)
                if not ok:
                    st.error(msg)
                else:
                    mini = minify_json(text)
                    if mini != text:
                        before, how = quick_count(text)
                        after, _ = quick_count(mini)
                        saved = before - after
                        pct = (saved / before * 100) if before else 0
                        cm, cb = st.columns([4, 1])
                        cm.caption(
                            f"Minificar economiza **{fmt_int(saved)} tokens** ({fmt_pct(pct)}) "
                            f"— {fmt_int(before)} → {fmt_int(after)} ({how}, {ref_model['label']})."
                        )
                        cb.button("Minificar", key=f"min_{bid}", on_click=_minify, args=(bid,),
                                  width="stretch")
                    else:
                        st.caption("JSON válido e já minificado.")


tab_blocks, tab_extras, tab_files, tab_preview = st.tabs(
    ["Blocos de prompt", "Prompt interno", "Arquivos", "Contexto inicial (JSON)"]
)

with tab_blocks:
    st.caption("Prompts que o **cliente parametriza** no agente: system prompt, instruções, persona, few-shot, tools.")
    render_block_list("blocks", PROMPT_BLOCK_TYPES)
    st.button("＋ Adicionar bloco", on_click=_add, args=("blocks", "instrucao"))

with tab_extras:
    st.caption(
        "Prompts **da aplicação que consome a API do agente**: instruções e dados que a própria aplicação "
        "injeta em toda conversa, fora da parametrização do cliente. Podem ser texto ou JSON."
    )
    render_block_list("internos", INTERNAL_BLOCK_TYPES)
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("＋ Texto", on_click=_add, args=("internos", "interno_texto"), width="stretch")
    c2.button("＋ JSON", on_click=_add, args=("internos", "interno_json"), width="stretch")


@st.cache_data(show_spinner=False, max_entries=200)
def _extract_cached(name: str, digest: str, data: bytes, version: int = EXTRACTOR_VERSION):
    return extract(name, data)


with tab_files:
    exts = sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS)
    uploads = st.file_uploader(
        "Base de conhecimento (.txt, .md, .csv, .json, .xml, .html, .pdf, .docx) e imagens "
        "(.png, .jpg, .gif, .webp)",
        type=exts, accept_multiple_files=True,
    )
    files = []
    for up in uploads or []:
        data = up.getvalue()
        files.append(_extract_cached(up.name, hashlib.sha256(data).hexdigest(), data, EXTRACTOR_VERSION))

    for f in files:
        if f.error:
            st.error(f"{f.name}: {f.error}")
    readable = [f for f in files if f.ok]
    images = [f for f in files if f.is_image and not f.error and f.width > 0 and f.height > 0]
    for f in files:
        if f.is_image and not f.error and f not in images:
            st.warning(f"{f.name}: não foi possível obter as dimensões; imagem fora da contagem.")
    pdfs = [f for f in readable if f.extension == ".pdf"]
    pdf_pages = sum(f.pages for f in pdfs)

    if readable:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Arquivo": f.name,
                        "Formato": f.extension.lstrip("."),
                        "Páginas": f.pages or None,
                        "Caracteres extraídos": f.chars,
                        "Observações": " ".join(f.warnings),
                    }
                    for f in readable
                ]
            ),
            hide_index=True, width="stretch",
        )

    vision_detail, pdf_as_image, pdf_dpi, pdf_tokens_per_page = DETAIL_DEFAULT, True, DEFAULT_PDF_DPI, 0
    if images or pdfs:
        st.subheader("Conteúdo visual")
        st.caption(
            "Imagens e páginas de PDF lidas nativamente não passam por tokenizador: cada provedor as converte "
            "em tokens por uma regra própria (blocos de 28 ou 32 px, valor fixo por imagem etc.). O app aplica a "
            "regra documentada de cada modelo a partir das dimensões. Veja as fórmulas no models.json (campo vision)."
        )
        vision_detail = st.radio(
            "Nível de detalhe visual", list(DETAIL_LABELS), format_func=DETAIL_LABELS.get, horizontal=True,
            help="Econômico = detail 'low' (OpenAI) / media_resolution 'low' (Gemini). "
                 "Claude, Qwen e DeepSeek não têm essa opção.",
        )
    if pdfs:
        c1, c2, c3 = st.columns(3)
        pdf_as_image = c1.toggle(
            "Somar páginas de PDF como imagem", value=True,
            help="Claude, Gemini e OpenAI leem o PDF nativamente e cobram cada página como imagem, "
                 "além do texto. Nos provedores sem leitura nativa, só o texto extraído conta.",
        )
        pdf_dpi = c2.number_input(
            "DPI suposto da página", min_value=50, max_value=600, value=DEFAULT_PDF_DPI, step=25,
            disabled=not pdf_as_image,
            help="Claude e OpenAI não publicam a resolução em que rasterizam o PDF. "
                 "Usado só por eles; o Gemini cobra um valor fixo por página.",
        )
        pdf_tokens_per_page = c3.number_input(
            f"Tokens por página (0 = fórmula; {fmt_int(pdf_pages)} págs.)", min_value=0, value=0, step=100,
            disabled=not pdf_as_image, help="Valor manual que substitui a fórmula do provedor.",
        )

    visual_items = [VisualItem(f.name, "imagem", [(f.width, f.height)]) for f in images]
    visual_items += [VisualItem(f.name, "pdf", f.page_sizes) for f in pdfs if f.page_sizes]

    if visual_items:
        preview_models = selected_models if compare_mode else [ref_model]
        prev_rows = []
        for it in visual_items:
            row = {
                "Arquivo": it.name,
                "Dimensões": (f"{it.sizes[0][0]}×{it.sizes[0][1]} px" if it.kind == "imagem"
                              else f"{len(it.sizes)} pág."),
            }
            for m in preview_models:
                if it.kind == "pdf" and not pdf_as_image:
                    row[m["label"]] = "—"
                    continue
                vc = visual_item_tokens(m, it, vision_detail, int(pdf_dpi), int(pdf_tokens_per_page))
                row[m["label"]] = fmt_int(vc.tokens) if vc.tokens is not None else "n/d"
            prev_rows.append(row)
        st.markdown("**Tokens visuais estimados por arquivo**")
        st.dataframe(pd.DataFrame(prev_rows), hide_index=True, width="stretch")
        if not compare_mode:
            info = image_tokens((ref_model.get("vision") or {}).get("image"), 1000, 1000, vision_detail)
            st.caption(
                f"{ref_model['label']}: {info.label}. "
                + ((ref_model.get("vision") or {}).get("notes") or "")
                + " 'n/d' = não suportado ou não documentado pelo provedor."
            )
        else:
            st.caption("'n/d' = não suportado ou não documentado pelo provedor; '—' = PDF sem soma visual.")


# --------------------------------------------------------------------------- #
# Montagem do contexto
# --------------------------------------------------------------------------- #
blocks: list[Block] = []
for bid in ss.blocks + ss.internos:
    blocks.append(Block(ss.get(f"name_{bid}", ""), ss.get(f"type_{bid}", "outro"), ss.get(f"text_{bid}", "")))
for f in readable:
    blocks.append(Block(f.name, FILE_BLOCK_TYPE, f.text, native_pdf=f.extension == ".pdf"))

ctx = assemble(blocks, wrap_files=wrap_files)
_ctx_cache: dict[bool, AssembledContext] = {True: ctx}


def ctx_for(model: dict) -> AssembledContext:
    """Contexto do modelo: sem o texto dos PDFs quando o provedor lê o PDF nativamente e não o cobra."""
    keep_pdf_text = not pdf_as_image or pdf_text_billed(model) or not any(b.native_pdf for b in ctx.blocks)
    if keep_pdf_text not in _ctx_cache:
        _ctx_cache[keep_pdf_text] = assemble([b for b in blocks if not b.native_pdf], wrap_files=wrap_files)
    return _ctx_cache[keep_pdf_text]
settings = Settings(
    lang=lang, pt_share=pt_share, overhead_tokens=int(overhead), wrap_files=wrap_files, use_local=use_local,
    visual_items=visual_items, vision_detail=vision_detail, pdf_as_image=pdf_as_image,
    pdf_dpi=int(pdf_dpi), pdf_tokens_per_page=int(pdf_tokens_per_page),
)


def stored_api_result(model: dict) -> tuple[dict | None, bool]:
    """(resultado, desatualizado?) da última contagem via API deste modelo."""
    saved = ss.api_results.get(model["id"])
    if not saved:
        return None, False
    fresh = saved["fp"] == context_fingerprint(model["id"], ctx_for(model), saved["per_block"])
    return (saved["result"], False) if fresh else (None, True)


def run_api(models: list[dict]) -> None:
    for m in models:
        try:
            with st.spinner(f"Contando via API: {m['label']}…"):
                res = count_via_api(m, ctx_for(m), per_block=api_per_block, wrap_files=wrap_files)
            ss.api_results[m["id"]] = {
                "fp": context_fingerprint(m["id"], ctx_for(m), api_per_block),
                "per_block": api_per_block,
                "result": res,
            }
        except ApiCountError as e:
            st.error(f"{m['label']}: {e}")


def run_estimate(model: dict) -> ModelEstimate:
    api_res, _ = stored_api_result(model)
    return estimate(model, ctx_for(model), settings, api_res)


def settings_dict() -> dict:
    return {
        "idioma": lang,
        "parcela_pt": pt_share if lang == "misto" else None,
        "overhead_tokens": int(overhead),
        "cache_implicito": cache_on,
        "taxa_acerto_cache": hit_rate if cache_on else None,
        "conversas_por_mes": int(conv_month),
        "mensagens_usuario_por_conversa": int(turns),
        "palavras_mensagem_usuario": list(user_words),
        "palavras_resposta_agente": int(response_words),
        "raciocinio_tokens_por_resposta": int(thinking_tokens),
        "raciocinio_volta_ao_historico": thoughts_in_history,
        "paginas_pdf": pdf_pages,
        "imagens": len(images),
        "detalhe_visual": vision_detail,
        "pdf_paginas_como_imagem": pdf_as_image,
        "pdf_dpi_suposto": int(pdf_dpi),
        "pdf_tokens_por_pagina_manual": int(pdf_tokens_per_page) or None,
        "envolver_arquivos": wrap_files,
    }


def simulate(model: dict, context_tokens: int) -> tuple[ConversationEstimate, float, str]:
    """Simula a conversa: palavras -> tokens com o tokenizador do modelo; depois turno a turno."""
    tpw, tpw_label = tokens_per_word(model, lang, pt_share, use_local)
    params = ConversationParams(
        turns=int(turns),
        user_tokens=max(1, round(sum(user_words) / 2 * tpw)),
        response_tokens=max(1, round(int(response_words) * tpw)),
        thinking_tokens=int(thinking_tokens),
        thoughts_in_history=thoughts_in_history,
        conversations_per_month=int(conv_month),
        cache_enabled=cache_on,
        cache_hit_rate=hit_rate,
    )
    return simulate_conversation(model, context_tokens, params), tpw, tpw_label


def conversation_dict(c: ConversationEstimate, tpw: float, tpw_label: str) -> dict:
    return {
        "tokens_por_palavra": round(tpw, 3),
        "tokens_por_palavra_metodo": tpw_label,
        "tokens_mensagem_usuario": c.params.user_tokens,
        "tokens_resposta_agente": c.params.response_tokens,
        "tokens_raciocinio_por_resposta": c.params.thinking_tokens,
        "entrada_por_conversa": c.input_total,
        "entrada_do_cache_por_conversa": round(c.cached_total),
        "saida_por_conversa": c.output_total,
        "raciocinio_por_conversa": c.thinking_total,
        "custo_entrada_por_conversa_usd": c.cost_input,
        "custo_saida_por_conversa_usd": c.cost_output,
        "custo_por_conversa_usd": c.cost,
        "custo_mensal_usd": c.monthly_cost,
        "pico_janela_pct": c.peak_window_pct,
        "turnos": [
            {"turno": t.turn, "entrada": t.input_tokens, "do_cache": round(t.cached_tokens),
             "raciocinio": t.thinking_tokens, "resposta": t.response_tokens, "custo_usd": t.cost}
            for t in c.turns
        ],
    }


# --------------------------------------------------------------------------- #
# Pré-visualização do contexto inicial (antes de tokenizar)
# --------------------------------------------------------------------------- #
def context_preview(model: dict | None) -> dict:
    mctx = ctx_for(model) if model else ctx
    text_billed = pdf_text_billed(model) if model else True
    preview: dict = {
        "descricao": "Contexto inicial carregado a cada nova conversa, antes da 1ª mensagem do usuário. "
                     "É este conteúdo que é tokenizado.",
        "modelo": model["id"] if model else None,
        "blocos": [
            {
                "ordem": i + 1,
                "nome": b.name,
                "tipo": b.type,
                "tipo_rotulo": ALL_TYPE_LABELS.get(b.type, b.type),
                "caracteres": len(block_text_for_count(b, wrap_files)),
                "conteudo": block_text_for_count(b, wrap_files),
            }
            for i, b in enumerate(mctx.blocks)
        ],
        "tools": mctx.tools,
        "conteudo_visual": [
            {"nome": it.name, "tipo": "imagem", "largura_px": it.sizes[0][0], "altura_px": it.sizes[0][1]}
            if it.kind == "imagem" else
            {"nome": it.name, "tipo": "pdf", "paginas": len(it.sizes),
             "tamanho_pagina_pt": list(it.sizes[0]) if it.sizes else None,
             "paginas_como_imagem": pdf_as_image,
             "texto_nativo_cobrado": text_billed or not pdf_as_image}
            for it in visual_items
        ],
        "montagem": {
            "separador_entre_blocos": SEPARATOR,
            "arquivos_em_tag_documento": wrap_files,
            "overhead_fixo_tokens": int(overhead),
        },
        "texto_montado": mctx.full_text,
    }
    omitted = [b.name for b in ctx.blocks if b.native_pdf and b not in mctx.blocks]
    if omitted:
        preview["omitidos_neste_modelo"] = {
            "blocos": omitted,
            "motivo": "o provedor lê o PDF nativamente e não cobra o texto extraído (só as páginas)",
        }
    provider = api_provider_for(model) if model and API_ON else None
    if provider:
        preview["requisicao_contagem_api"] = {
            "endpoint": API_ENDPOINTS[provider],
            "corpo": api_payload(provider, model["id"], mctx.system_text, mctx.tools, mctx.full_text),
            "observacao": "Enviada só ao clicar em 'Contar via API'. Imagens e páginas de PDF não vão na "
                          "requisição de texto: são estimadas pela fórmula do provedor.",
        }
    return preview


with tab_preview:
    if not ctx.blocks and not visual_items:
        st.caption("O contexto está vazio.")
    else:
        pv_models = selected_models if compare_mode else [ref_model]
        pv_model = ref_model if pv_models else None
        if len(pv_models) > 1:
            pv_id = st.selectbox(
                "Ver como o contexto fica para o modelo", [m["id"] for m in pv_models],
                format_func=lambda i: f"{MODELS_BY_ID[i]['provider']} · {MODELS_BY_ID[i]['label']}",
                key="preview_model",
            )
            pv_model = MODELS_BY_ID[pv_id]
        preview = context_preview(pv_model)
        preview_json = json.dumps(preview, ensure_ascii=False, indent=2)
        st.caption(
            f"{fmt_int(len(preview['blocos']))} blocos · {fmt_int(len(preview['texto_montado']))} caracteres "
            f"no texto montado · {fmt_int(len(preview['conteudo_visual']))} item(ns) visual(is). "
            "Nada foi tokenizado nem enviado ainda."
        )
        if preview.get("omitidos_neste_modelo"):
            st.info(
                "Neste modelo o texto extraído do PDF não entra no contexto de texto "
                f"({', '.join(preview['omitidos_neste_modelo']['blocos'])}): "
                + preview["omitidos_neste_modelo"]["motivo"] + "."
            )
        st.download_button(
            "Baixar JSON", preview_json, mime="application/json",
            file_name=f"contexto_inicial_{pv_model['id'] if pv_model else 'neutro'}.json",
        )
        st.json(preview, expanded=2)


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
st.divider()
st.header("Resultado")

for w in ctx.warnings:
    st.warning(w)

if not ctx.blocks and not settings.visual_items:
    st.info("Adicione texto em algum bloco, informação extra ou arquivo para ver a estimativa.")
    st.stop()

now = datetime.now(timezone.utc).isoformat(timespec="seconds")

if not compare_mode:
    model = ref_model
    provider = api_provider_for(model)
    has_key = bool(provider and get_api_key(provider))

    if API_ON:
        col_btn, col_msg = st.columns([1, 3])
        if col_btn.button("Contar via API", disabled=not has_key, type="primary", width="stretch"):
            run_api([model])
        if not provider:
            col_msg.caption("Este modelo não tem endpoint de contagem configurado: usando tokenizador local/heurística.")
        elif not has_key:
            col_msg.caption(
                f"Sem chave para {provider} no .env ({' / '.join(API_KEY_ENV[provider])}): resultado **estimado**."
            )
        else:
            col_msg.caption("Envia o contexto ao provedor só quando você clicar. Nada é enviado automaticamente.")

    _, stale = stored_api_result(model)
    if stale:
        st.info("O conteúdo mudou desde a última contagem via API. Clique de novo para atualizar.")

    with st.spinner("Carregando tokenizador local (o Hugging Face baixa na primeira vez)…"):
        est = run_estimate(model)
    if est.local_error:
        st.warning(f"Tokenizador local indisponível; usando heurística. Detalhe: {est.local_error}")
    for note in est.visual_notes:
        st.warning(f"Visual: {note}")
    if ctx_for(model) is not ctx:
        st.info(f"{model['label']} lê o PDF nativamente e não cobra o texto extraído: "
                "só as páginas (como imagem) entram na contagem.")

    # ---- Card principal
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens: heurística", fmt_int(est.heuristic_total), help="Sempre estimado (chars ÷ chars/token).")
        exact_title = "Tokens: " + ("exato" if est.exact_total is not None else
                                    "aproximado" if est.local_total is not None else "estimado")
        c2.metric(exact_title, fmt_int(est.best_total), help=est.best_label)
        c2.caption(est.best_label)
        pct = est.window_pct
        c3.metric("Janela de contexto usada", fmt_pct(pct), help=f"de {fmt_int(model['context_window'])} tokens")
        st.progress(min(pct / 100, 1.0))
        if pct >= CRIT_PCT:
            st.error(f"O contexto inicial ocupa {fmt_pct(pct)} da janela: sobra pouco espaço para a conversa.")
        elif pct >= WARN_PCT:
            st.warning(f"O contexto inicial ocupa {fmt_pct(pct)} da janela: mais da metade antes da 1ª mensagem.")
        st.caption(
            f"Contexto montado (requisição única): **{fmt_int(est.best_total)}** · "
            f"soma dos blocos isolados: **{fmt_int(est.best_sum)}** "
            f"(diferença {fmt_int(est.best_total - est.best_sum)}). "
            "A soma das partes difere do total por causa de separadores, wrappers e fusões de tokens nas fronteiras."
            + (f" Inclui **{fmt_int(est.visual_total)}** tokens visuais (imagens/páginas de PDF, pela fórmula "
               "do provedor)." if est.visual_total else "")
        )

    # ---- Tabela por bloco
    st.subheader("Por bloco")
    rows = est.rows
    weights = [r.exact if r.exact is not None else r.heuristic for r in rows]
    total_w = sum(weights) or 1
    df = pd.DataFrame(
        [
            {
                "Nome": r.name,
                "Tipo": r.type_label,
                "Caracteres": r.chars,
                "Palavras": r.words,
                "Tokens (heurística)": r.heuristic,
                "Tokens (exato)": r.exact,
                "Método": r.exact_kind or "",
                "% do total": w / total_w * 100,
            }
            for r, w in zip(rows, weights)
        ]
    )
    top = set(df["% do total"].nlargest(min(3, len(df))).index) if len(df) > 1 else set()

    def _hl(row):
        return ["background-color: rgba(255, 165, 0, 0.22); font-weight: 600" if row.name in top else ""] * len(row)

    st.dataframe(
        df.style.apply(_hl, axis=1).format(
            {"% do total": lambda v: fmt_pct(v), "Tokens (exato)": lambda v: fmt_int(None if pd.isna(v) else int(v))},
            thousands=".",
        ),
        hide_index=True, width="stretch",
    )
    kinds = {r.exact_kind for r in rows if r.exact_kind in ("exato", "aproximado")}
    st.caption(
        "Clique no cabeçalho para ordenar. Destaque: os 3 blocos mais pesados. "
        + ("Coluna 'exato' com tokenizador **proxy** (aproximado). " if kinds == {"aproximado"} else "")
        + "% calculado sobre a soma dos blocos."
    )

    # ---- Simulação da conversa
    st.subheader("Simulação da conversa: entrada e saída")
    conv, tpw, tpw_label = simulate(model, est.best_total)
    p = conv.params
    st.caption(
        f"Cada mensagem do usuário dispara uma requisição que reprocessa o **contexto inicial + todo o histórico** "
        f"(a API não guarda estado entre chamadas; o que barateia é o cache implícito). "
        f"Mensagem do usuário ≈ **{fmt_int(p.user_tokens)}** tokens, resposta ≈ **{fmt_int(p.response_tokens)}**, "
        f"raciocínio = **{fmt_int(p.thinking_tokens)}** por resposta "
        f"({str(round(tpw, 2)).replace('.', ',')} tokens/palavra medidos com {tpw_label})."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entrada por conversa", fmt_int(conv.input_total),
              help=f"Soma da entrada dos {p.turns} turnos. Do cache: {fmt_int(round(conv.cached_total))}. "
                   f"O contexto inicial reprocessado é {fmt_pct(conv.context_share_of_input * 100)} da entrada.")
    m2.metric("Saída por conversa", fmt_int(conv.output_total),
              help=f"Raciocínio {fmt_int(conv.thinking_total)} + respostas {fmt_int(conv.response_total)}.")
    m3.metric("Custo por conversa", format_usd(conv.cost),
              help=f"Entrada {format_usd(conv.cost_input)} + saída {format_usd(conv.cost_output)}.")
    m4.metric(f"Mensal ({fmt_int(int(conv_month))} conversas)", format_usd(conv.monthly_cost),
              help=f"Entrada: {fmt_int(conv.monthly_input_tokens)} tokens · "
                   f"saída: {fmt_int(conv.monthly_output_tokens)} tokens por mês.")
    if conv.output_price_missing:
        st.warning("Este modelo não tem preço de saída no catálogo: o custo da saída aparece como zero.")
    if conv.peak_window_pct >= CRIT_PCT:
        st.error(f"No último turno a conversa ocupa {fmt_pct(conv.peak_window_pct)} da janela de contexto.")
    elif conv.peak_window_pct >= WARN_PCT:
        st.warning(f"No último turno a conversa ocupa {fmt_pct(conv.peak_window_pct)} da janela de contexto.")
    if cache_on and conv.turns and conv.turns[-1].input_tokens < conv.cache_min_tokens:
        st.info(f"As requisições ficam abaixo do mínimo de {fmt_int(conv.cache_min_tokens)} tokens do cache "
                "implícito deste modelo: nenhuma parte é lida do cache.")
    elif cache_on and est.best_total < conv.cache_min_tokens:
        st.info(f"O contexto inicial ({fmt_int(est.best_total)} tokens) fica abaixo do mínimo de "
                f"{fmt_int(conv.cache_min_tokens)} tokens do cache implícito: o cache só entra quando o histórico "
                "faz a requisição passar desse tamanho.")
    tdf = pd.DataFrame(
        [
            {
                "Turno": t.turn,
                "Entrada": t.input_tokens,
                "Do cache": round(t.cached_tokens),
                "Entrada nova": round(t.new_input_tokens),
                "Raciocínio": t.thinking_tokens,
                "Resposta": t.response_tokens,
                "Custo entrada": t.cost_input,
                "Custo saída": t.cost_output,
                "Custo do turno": t.cost,
                "% janela": t.window_pct,
            }
            for t in conv.turns
        ]
    )
    st.dataframe(
        tdf.style.format(
            {"Custo entrada": format_usd, "Custo saída": format_usd, "Custo do turno": format_usd,
             "% janela": fmt_pct},
            thousands=".",
        ),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Entrada do turno n = contexto inicial + mensagens e respostas anteriores"
        + (" + raciocínios anteriores" if thoughts_in_history else "")
        + " + nova mensagem. 'Do cache' = prefixo repetido da requisição anterior × taxa de acerto, se a "
        f"requisição tiver ≥ {fmt_int(conv.cache_min_tokens)} tokens. Saída = raciocínio + resposta "
        f"(o preço de saída inclui o raciocínio). Preços de {model.get('price_checked_at', '?')} (models.json)."
        + (" **Faixa de contexto longo aplicada em algum turno.**" if any(t.long_context for t in conv.turns)
           else "")
        + " Fora da conta: chamadas de ferramentas (tools) e poucos tokens de formatação por mensagem."
    )

    # ---- Calibração
    st.subheader("Calibração da heurística")
    cal = est.calibration
    if cal and cal["is_exact"]:
        diff = (cal["configured_chars_per_token"] / cal["observed_chars_per_token"] - 1) * 100
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("chars/token reais", f"{cal['observed_chars_per_token']:.2f}".replace(".", ","))
        cc2.metric("chars/token configurados", f"{cal['configured_chars_per_token']:.2f}".replace(".", ","))
        cc3.metric("Erro da heurística", fmt_pct(diff))
        field_name = {"pt": "chars_per_token_pt", "en": "chars_per_token_en"}.get(lang)
        st.caption(
            f"Fonte: {cal['source']} sobre {fmt_int(cal['chars'])} caracteres. "
            + (f"Para calibrar, ajuste `{field_name}` de `{model['id']}` para "
               f"**{cal['observed_chars_per_token']:.2f}** no models.json."
               if field_name else "No modo misto, calibre com textos só em PT ou só em EN.")
        )
    elif cal:
        st.caption(
            f"Observado com tokenizador proxy ({cal['source']}): "
            f"{cal['observed_chars_per_token']:.2f} chars/token (configurado: "
            f"{cal['configured_chars_per_token']:.2f})."
            + (" Use 'Contar via API' para uma calibração exata." if API_ON else "")
        )
    else:
        st.caption("Sem contagem exata disponível para este modelo"
                   + (": use 'Contar via API' (com chave) para medir os chars/token reais." if API_ON else "."))

    # ---- Exportação
    export = {
        "gerado_em": now,
        "configuracoes": settings_dict(),
        "modelo": {k: model.get(k) for k in ("id", "label", "provider", "context_window", "exact_method",
                                             "local_method", "price_checked_at")},
        "totais": {
            "heuristico_contexto": est.heuristic_total,
            "heuristico_soma_blocos": est.heuristic_sum,
            "local_contexto": est.local_total,
            "local_soma_blocos": est.local_sum,
            "local_metodo": est.local_label,
            "api_contexto": est.api_total,
            "api_soma_blocos": est.api_sum,
            "melhor": est.best_total,
            "melhor_metodo": est.best_label,
            "pct_janela": est.window_pct,
        },
        "blocos": df.to_dict(orient="records"),
        "conversa": conversation_dict(conv, tpw, tpw_label),
        "calibracao": est.calibration,
        "avisos": ctx.warnings,
    }
    e1, e2, _ = st.columns([1, 1, 3])
    e1.download_button("Exportar JSON", json.dumps(export, ensure_ascii=False, indent=2, default=str),
                       file_name=f"estimativa_{model['id']}.json", mime="application/json",
                       width="stretch")
    e2.download_button("Exportar CSV", df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"estimativa_{model['id']}.csv", mime="text/csv", width="stretch")

else:
    # ------------------------------------------------------------------- #
    # Modo comparar
    # ------------------------------------------------------------------- #
    if not selected_models:
        st.info("Escolha na barra lateral quais modelos comparar.")
        st.stop()
    if API_ON:
        with_api = [m for m in selected_models if api_provider_for(m) and get_api_key(api_provider_for(m))]
        col_btn, col_msg = st.columns([1, 3])
        if col_btn.button("Contar via API", disabled=not with_api, type="primary", width="stretch"):
            run_api(with_api)
        col_msg.caption(
            f"Conta via API os {len(with_api)} modelos com chave configurada. Os demais usam tokenizador local "
            "ou heurística (rótulo 'estimado')."
        )
    with st.spinner("Carregando tokenizadores locais (o Hugging Face baixa na primeira vez)…"):
        ests = [run_estimate(m) for m in selected_models]

    errors = [f"{e.model['label']}: {e.local_error}" for e in ests if e.local_error]
    if errors:
        with st.expander(f"{len(errors)} tokenizador(es) local(is) indisponível(is): usando heurística"):
            for msg in errors:
                st.caption(msg)
    vnotes = [f"{e.model['label']}: {n}" for e in ests for n in e.visual_notes]
    if vnotes:
        with st.expander(f"{len(vnotes)} aviso(s) sobre conteúdo visual"):
            for msg in vnotes:
                st.caption(msg)

    records, convs = [], []
    for e in ests:
        c, tpw, tpw_label = simulate(e.model, e.best_total)
        convs.append((c, tpw, tpw_label))
        records.append(
            {
                "Provedor": e.model["provider"],
                "Modelo": e.model["label"],
                "Método": e.best_label,
                "Contexto inicial (tokens)": e.best_total,
                "Tokens visuais": e.visual_total,
                "% da janela (pico)": c.peak_window_pct,
                "Entrada/conversa": c.input_total,
                "Do cache/conversa": round(c.cached_total),
                "Saída/conversa": c.output_total,
                "Custo/conversa (US$)": c.cost,
                "Mensal (US$)": c.monthly_cost,
            }
        )
    cdf = pd.DataFrame(records)

    def _hl_pct(v):
        if v >= CRIT_PCT:
            return "background-color: rgba(220, 53, 69, 0.25)"
        if v >= WARN_PCT:
            return "background-color: rgba(255, 165, 0, 0.25)"
        return ""

    money = lambda v: "—" if v is None or pd.isna(v) else format_usd(v)  # noqa: E731
    st.dataframe(
        cdf.style.map(_hl_pct, subset=["% da janela (pico)"]).format(
            {
                "% da janela (pico)": fmt_pct,
                "Custo/conversa (US$)": money,
                "Mensal (US$)": money,
            },
            thousands=".",
        ),
        hide_index=True, width="stretch", height=38 * (len(cdf) + 1),
    )
    st.caption(
        "Clique no cabeçalho para ordenar. 'Melhor' = API > tokenizador local exato > proxy > heurística. "
        f"Conversa simulada: {fmt_int(int(turns))} mensagens do usuário de {user_words[0]}–{user_words[1]} "
        f"palavras, respostas de {fmt_int(int(response_words))} palavras e {fmt_int(int(thinking_tokens))} tokens "
        f"de raciocínio por resposta; {fmt_int(int(conv_month))} conversas por mês"
        + (f", cache implícito com {fmt_pct(hit_rate * 100)} de acerto." if cache_on else ", sem cache.")
        + " Preços em models.json (campo price_checked_at)."
    )

    export = {
        "gerado_em": now,
        "configuracoes": settings_dict(),
        "modelos": [
            {**rec, "id": e.model["id"], "soma_blocos_melhor": e.best_sum, "calibracao": e.calibration,
             "conversa": conversation_dict(*cv)}
            for rec, e, cv in zip(records, ests, convs)
        ],
        "avisos": ctx.warnings,
    }
    e1, e2, _ = st.columns([1, 1, 3])
    e1.download_button("Exportar JSON", json.dumps(export, ensure_ascii=False, indent=2, default=str),
                       file_name="comparacao_modelos.json", mime="application/json", width="stretch")
    e2.download_button("Exportar CSV", cdf.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="comparacao_modelos.csv", mime="text/csv", width="stretch")
