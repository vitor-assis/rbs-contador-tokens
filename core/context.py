"""Montagem do contexto inicial (system prompt + extras + arquivos + tools).

O contexto montado é o que seria enviado a cada nova conversa, antes da primeira
mensagem do usuário. Ele existe em duas formas:

* ``full_text``: tudo concatenado como texto (usado pela heurística e pelos
  tokenizadores locais);
* ``system_text`` + ``tools``: a forma estruturada usada pelas APIs de contagem
  que aceitam ferramentas separadamente (Anthropic, OpenAI).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# Tipos de bloco. As chaves são estáveis (vão para o JSON exportado);
# os rótulos aparecem na UI.
PROMPT_BLOCK_TYPES: dict[str, str] = {
    "system": "System prompt",
    "instrucao": "Instrução",
    "persona": "Persona",
    "few_shot": "Few-shot",
    "tools": "Definição de tools/functions",
    "outro": "Outro",
}
# Prompt interno: prompts da aplicação que consome a API do agente (não parametrizados pelo cliente).
INTERNAL_BLOCK_TYPES: dict[str, str] = {
    "interno_texto": "Prompt interno (texto)",
    "interno_json": "Prompt interno (JSON)",
}
FILE_BLOCK_TYPE = "arquivo"
PDF_PAGES_TYPE = "pdf_paginas"

ALL_TYPE_LABELS: dict[str, str] = {
    **PROMPT_BLOCK_TYPES,
    **INTERNAL_BLOCK_TYPES,
    FILE_BLOCK_TYPE: "Arquivo",
    PDF_PAGES_TYPE: "Páginas de PDF (imagem)",
}

SEPARATOR = "\n\n"


@dataclass
class Block:
    name: str
    type: str
    text: str
    native_pdf: bool = False  # texto extraído de um PDF (alguns provedores não cobram esse texto)

    @property
    def type_label(self) -> str:
        return ALL_TYPE_LABELS.get(self.type, self.type)


@dataclass
class AssembledContext:
    blocks: list[Block]
    system_text: str
    tools: list[dict] | None
    tools_text: str
    full_text: str
    warnings: list[str] = field(default_factory=list)


def validate_json(text: str) -> tuple[bool, str | None]:
    """Valida JSON e devolve (ok, mensagem amigável em PT-BR)."""
    if not text.strip():
        return False, "O bloco JSON está vazio."
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"JSON inválido na linha {e.lineno}, coluna {e.colno}: {e.msg}."
    return True, None


def minify_json(text: str) -> str:
    """Remove espaços e quebras de linha supérfluos. Lança ValueError se inválido."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido: {e.msg}") from e
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def wrap_file(name: str, text: str) -> str:
    return f'<documento nome="{name}">\n{text}\n</documento>'


def parse_tools(text: str) -> list[dict] | None:
    """Converte um bloco de tools em lista normalizada {name, description, parameters}.

    Aceita o formato Anthropic (``input_schema``), OpenAI Chat
    (``{"type": "function", "function": {...}}``), OpenAI Responses
    (``{"type": "function", "name": ...}``) ou Gemini (``function_declarations``).
    Devolve None se o texto não for JSON reconhecível como tools.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        if "tools" in data and isinstance(data["tools"], list):
            data = data["tools"]
        elif "function_declarations" in data:
            data = data["function_declarations"]
        else:
            data = [data]
    if not isinstance(data, list):
        return None

    tools: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        if "function_declarations" in item:
            for fd in item["function_declarations"]:
                tools.append(_norm_tool(fd))
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            item = item["function"]
        if "name" not in item:
            return None
        tools.append(_norm_tool(item))
    return tools or None


def _norm_tool(item: dict) -> dict:
    schema = item.get("input_schema") or item.get("parameters") or {"type": "object", "properties": {}}
    return {
        "name": str(item.get("name", "tool")),
        "description": str(item.get("description", "")),
        "parameters": schema,
    }


def assemble(blocks: list[Block], wrap_files: bool = True) -> AssembledContext:
    """Monta o contexto na ordem dos blocos.

    - Blocos de tools vão para ``tools`` (se forem JSON reconhecível) e também
      para ``full_text``, pois contam como entrada.
    - Arquivos são envolvidos em ``<documento>`` quando ``wrap_files``.
    - Blocos vazios são ignorados.
    """
    system_parts: list[str] = []
    full_parts: list[str] = []
    tools: list[dict] = []
    tools_texts: list[str] = []
    warnings: list[str] = []

    kept = [b for b in blocks if b.text and b.text.strip() and b.type != PDF_PAGES_TYPE]
    for b in kept:
        if b.type == "tools":
            parsed = parse_tools(b.text)
            if parsed is None:
                warnings.append(
                    f"O bloco de tools '{b.name}' não é um JSON de tools reconhecível; "
                    "ele será tratado como texto no system prompt."
                )
                system_parts.append(b.text)
            else:
                tools.extend(parsed)
                tools_texts.append(b.text)
            full_parts.append(b.text)
            continue
        text = wrap_file(b.name, b.text) if (b.type == FILE_BLOCK_TYPE and wrap_files) else b.text
        system_parts.append(text)
        full_parts.append(text)

    return AssembledContext(
        blocks=kept,
        system_text=SEPARATOR.join(system_parts),
        tools=tools or None,
        tools_text=SEPARATOR.join(tools_texts),
        full_text=SEPARATOR.join(full_parts),
        warnings=warnings,
    )


def block_text_for_count(block: Block, wrap_files: bool = True) -> str:
    """Texto do bloco exatamente como entra no contexto montado."""
    if block.type == FILE_BLOCK_TYPE and wrap_files:
        return wrap_file(block.name, block.text)
    return block.text
