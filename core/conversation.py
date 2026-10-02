"""Simulação de uma conversa: entrada e saída turno a turno.

As APIs de LLM não guardam estado entre chamadas: a cada mensagem do usuário o modelo
processa de novo (e cobra como entrada) o contexto inicial + todo o histórico até ali.
O que barateia é o cache implícito: a parte repetida da requisição anterior é lida do
cache a um preço reduzido, desde que a requisição tenha um tamanho mínimo.

Turno n (1 turno = 1 mensagem do usuário + 1 resposta do agente):
    entrada(n) = contexto + soma_{j<n}(msg_usuario + resposta [+ raciocínio]) + msg_usuario
    em cache(n) = entrada(n-1) (ou o contexto, no 1º turno) x taxa de acerto,
                  se a parte em cache tiver >= cache_min_tokens
    saída(n)   = raciocínio + resposta     (o preço de saída inclui o raciocínio)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import prices_for


@dataclass
class ConversationParams:
    turns: int = 5                      # mensagens do usuário por conversa
    user_tokens: int = 15               # tokens por mensagem do usuário
    response_tokens: int = 150          # tokens por resposta do agente
    thinking_tokens: int = 500          # tokens de raciocínio por resposta
    thoughts_in_history: bool = False   # pior caso: o raciocínio anterior volta como entrada
    conversations_per_month: int = 1000
    cache_enabled: bool = True
    cache_hit_rate: float = 0.95


@dataclass
class TurnRow:
    turn: int
    input_tokens: int
    cached_tokens: float
    new_input_tokens: float
    thinking_tokens: int
    response_tokens: int
    cost_input: float
    cost_output: float
    window_pct: float
    long_context: bool

    @property
    def output_tokens(self) -> int:
        return self.thinking_tokens + self.response_tokens

    @property
    def cost(self) -> float:
        return self.cost_input + self.cost_output


@dataclass
class ConversationEstimate:
    context_tokens: int
    params: ConversationParams
    turns: list[TurnRow] = field(default_factory=list)
    cache_min_tokens: int = 0
    output_price_missing: bool = False

    @property
    def input_total(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def cached_total(self) -> float:
        return sum(t.cached_tokens for t in self.turns)

    @property
    def thinking_total(self) -> int:
        return sum(t.thinking_tokens for t in self.turns)

    @property
    def response_total(self) -> int:
        return sum(t.response_tokens for t in self.turns)

    @property
    def output_total(self) -> int:
        return self.thinking_total + self.response_total

    @property
    def cost_input(self) -> float:
        return sum(t.cost_input for t in self.turns)

    @property
    def cost_output(self) -> float:
        return sum(t.cost_output for t in self.turns)

    @property
    def cost(self) -> float:
        return self.cost_input + self.cost_output

    @property
    def monthly_cost(self) -> float:
        return self.cost * max(0, self.params.conversations_per_month)

    @property
    def monthly_input_tokens(self) -> int:
        return self.input_total * max(0, self.params.conversations_per_month)

    @property
    def monthly_output_tokens(self) -> int:
        return self.output_total * max(0, self.params.conversations_per_month)

    @property
    def monthly_cached_tokens(self) -> int:
        return round(self.cached_total * max(0, self.params.conversations_per_month))

    @property
    def monthly_total_tokens(self) -> int:
        return self.monthly_input_tokens + self.monthly_output_tokens

    @property
    def peak_window_pct(self) -> float:
        return max((t.window_pct for t in self.turns), default=0.0)

    @property
    def context_share_of_input(self) -> float:
        """Fração da entrada da conversa que é o contexto inicial reprocessado."""
        total = self.input_total
        return (self.context_tokens * len(self.turns) / total) if total else 0.0


def simulate_conversation(model: dict, context_tokens: int, params: ConversationParams) -> ConversationEstimate:
    turns = max(1, int(params.turns))
    hit = min(max(params.cache_hit_rate, 0.0), 1.0) if params.cache_enabled else 0.0
    min_cache = int(model.get("cache_min_tokens") or 0)
    window = model.get("context_window") or 0
    out_missing = model.get("price_output_per_1m") is None
    est = ConversationEstimate(context_tokens=context_tokens, params=params, cache_min_tokens=min_cache,
                               output_price_missing=out_missing)

    history = 0           # tokens das trocas anteriores que voltam como entrada
    prev_input = None     # entrada do turno anterior (prefixo do turno atual)
    for n in range(1, turns + 1):
        input_tokens = context_tokens + history + params.user_tokens
        cacheable = context_tokens if prev_input is None else prev_input
        cached = cacheable * hit if cacheable >= min_cache else 0.0
        new_input = input_tokens - cached

        p = prices_for(model, input_tokens)
        # sem cache: entrada nova paga o preço normal; com cache: o que é gravado paga a escrita
        # (no Gemini o cache implícito não tem taxa de escrita: escrita = entrada)
        new_price = p.cache_write if params.cache_enabled else p.input
        cost_in = (cached * p.cache_read + new_input * new_price) / 1_000_000
        out_tokens = params.thinking_tokens + params.response_tokens
        cost_out = out_tokens * (p.output or 0.0) / 1_000_000
        est.turns.append(TurnRow(
            turn=n,
            input_tokens=input_tokens,
            cached_tokens=cached,
            new_input_tokens=new_input,
            thinking_tokens=params.thinking_tokens,
            response_tokens=params.response_tokens,
            cost_input=cost_in,
            cost_output=cost_out,
            window_pct=((input_tokens + out_tokens) / window * 100.0) if window else 0.0,
            long_context=p.long_context,
        ))
        prev_input = input_tokens
        history += params.user_tokens + params.response_tokens
        if params.thoughts_in_history:
            history += params.thinking_tokens
    return est
