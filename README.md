# Estimador de Tokens de Contexto

App web simples (Streamlit, PT-BR) que estima quantos tokens o **contexto inicial** de um
agente de IA consome **antes da primeira mensagem do usuário**, ou seja, tudo o que é
carregado a cada nova conversa:

- **Blocos de prompt:** os prompts que o **cliente parametriza** (system prompt, instruções,
  persona, few-shot, definições de tools).
- **Prompt interno:** os prompts **da aplicação que consome a API do agente**, injetados em toda
  conversa fora da parametrização do cliente (texto ou JSON).
- **Arquivos** da base de conhecimento (.txt, .md, .csv, .json, .xml, .html, .pdf, .docx) e imagens.

A versão publicada traz **só os modelos Gemini (Google)** e roda **sem nenhuma chave de API**:
a contagem usa tokenizadores locais abertos e a heurística. Nenhum conteúdo é enviado a
provedores de IA.

## Publicar no Streamlit Community Cloud (plano gratuito)

1. Acesse [share.streamlit.io](https://share.streamlit.io) e entre com a conta do GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repositório `vitor-assis/rbs-contador-tokens`, branch `main`, arquivo principal `app.py`.
4. Em **Advanced settings**, escolha **Python 3.12**. Não é preciso configurar *Secrets*.
5. **Deploy**. A 1ª abertura de cada modelo baixa o tokenizador (Gemma 4 do Hugging Face e
   Gemma 3 do GitHub do Google). Depois ele fica em cache enquanto o app estiver no ar.

O `requirements.txt` contém só o que a versão publicada usa (veja "Memória" abaixo).

## Rodar localmente

Requer Python 3.11+.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
streamlit run app.py
```

Testes:

```bash
pytest
```

Para usar o **catálogo com todos os provedores** (Anthropic, OpenAI, xAI, DeepSeek, Qwen e
Google), defina `MODELS_FILE=catalogo/todos_os_provedores.json` antes de rodar. Com chaves no
`.env` (veja `.env.example`), aparece o botão "Contar via API". Sem nenhuma chave, como na versão
publicada, a parte de API some da interface.

## Como a contagem funciona

| Nível | Quando roda | Rótulo na UI |
|---|---|---|
| **Heurística** `chars ÷ chars_per_token` | sempre | "estimado (heurística)" |
| **Tokenizador local aberto** | automaticamente | "exato" ou "aproximado (proxy)" |
| **API de contagem do provedor** (só uso local, com chave) | só ao clicar em "Contar via API" | "exato (API …)" |

A "melhor estimativa" segue a ordem **API > local exato > local proxy > heurística**.

- **Heurística:** usa `chars_per_token_en` / `chars_per_token_pt` do modelo. No modo *misto*
  é a média ponderada dos tokens/caractere pela parcela em português.
- **Total vs. soma:** cada bloco é contado isolado **e** o contexto é contado montado como
  uma requisição única. Os dois aparecem, porque a soma das partes ≠ total.
- **Overhead fixo:** tokens que o provedor injeta (wrappers, prompt de sistema de tools).
- **Ordem de montagem:** blocos de prompt → prompt interno → arquivos (veja a aba
  "Contexto inicial (JSON)").

### Tokenizadores dos modelos Gemini

| Modelos | Tokenizador local | Situação |
|---|---|---|
| Gemini 3.5 Flash, 3.1 Pro (preview), 3.1 Flash-Lite | `hf:google/gemma-4-E4B-it` | **exato para texto**: é o tokenizador que o `LocalTokenizer` oficial do google-genai usa para esses modelos (contagens idênticas conferidas em 5 tipos de texto) |
| Gemini 3.8 Flash, 3.5 Flash-Lite | `hf:google/gemma-4-E4B-it` | **proxy**: ainda fora do mapeamento oficial do google-genai |
| Gemini 2.5 Pro, 2.5 Flash | `genai:<modelo>` (`LocalTokenizer` oficial, Gemma 3 / SentencePiece) | **exato para texto** |

O Gemma 4 é carregado **uma única vez** pela biblioteca `tokenizers`, leve, e compartilhado entre
todos os Gemini 3.x. Pelo `LocalTokenizer`, cada modelo carregaria a própria cópia via
`transformers`, cerca de 230 MB cada.

## Consumo e custo

Cada **interação** de uma conversa é uma requisição que **reenvia o contexto inicial**. Por isso
a barra lateral tem **Conversas por mês** e **Interações por conversa**:

- tokens por conversa = contexto inicial × interações;
- tokens por mês = × conversas;
- sem cache: cada interação paga o preço de entrada;
- com cache: a 1ª interação escreve o cache e as seguintes leem. "Cache já válido" é quando outra
  conversa recente já deixou o prefixo no cache;
- mensal: `conversas × interações` requisições, divididas entre leituras e escritas pela **taxa
  de acerto do cache**. A 1ª requisição do mês é sempre escrita.

A janela de contexto contém o contexto inicial uma vez só (a % da janela não muda com as
interações). Fica fora da conta: as mensagens, as respostas e o histórico que cresce na conversa.
Também fica fora o armazenamento por hora do cache explícito do Gemini.

## Conteúdo visual (imagens e páginas de PDF)

Imagem não passa por tokenizador: o Google converte a imagem em tokens por uma regra publicada,
baseada nas dimensões. O app aplica essa regra (bloco `vision` do `models.json`, `core/vision.py`):

| Família | Imagem | Página de PDF |
|---|---|---|
| Gemini 3.x | fixo: 1120 (padrão) / 280 (econômico) por imagem | fixo: 560 (padrão) / 280 (econômico) por página |
| Gemini 2.5 | 258 se os dois lados ≤ 384 px; senão blocos de `floor(min(w,h)/1,5)` px (até 768), 258 cada | 258 por página |

O Gemini lê o PDF nativamente e **não cobra o texto extraído**, só as páginas. Por isso, com
"Somar páginas de PDF como imagem" ligado, o texto do PDF sai da contagem. Desligado, conta o
texto extraído, como se a aplicação o enviasse como texto.

## Contexto inicial (JSON)

A aba **"Contexto inicial (JSON)"** mostra, antes de qualquer tokenização, o que será contado:
os blocos na ordem de montagem, as tools normalizadas, os itens visuais, as regras de montagem e o
`texto_montado` (string exata que vai para o tokenizador). Há um botão para baixar o JSON.

## Modo comparar

Escolha "Comparar modelos…" e selecione na barra lateral quais modelos entram. Os botões "Todos",
"Nenhum" e "＋ Provedor" ajudam na seleção.

## Modelos (`models.json`)

Nada é hardcoded na UI: modelos, janelas, preços e métodos vêm do `models.json`. Os dados foram
conferidos na documentação oficial em **30/09/2026** (campo `price_checked_at`).

| Modelo | Entrada (US$/1M) | Cache leitura | Janela |
|---|---|---|---|
| Gemini 3.8 Flash | 0,75 (promocional até 31/12/2026; depois 1,50) | 0,075 | 1.048.576 |
| Gemini 3.5 Flash | 1,50 | 0,15 | 1.048.576 |
| Gemini 3.5 Flash-Lite | 0,30 | 0,03 | 1.048.576 |
| Gemini 3.1 Pro (preview) | 2,00 (> 200k: 4,00) | 0,20 (> 200k: 0,40) | 1.048.576 |
| Gemini 3.1 Flash-Lite | 0,25 | 0,025 | 1.048.576 |
| Gemini 2.5 Pro | 1,25 (> 200k: 2,50) | 0,125 (> 200k: 0,25) | 1.048.576 |
| Gemini 2.5 Flash | 0,30 | 0,03 | 1.048.576 |

A família 2.5 tem acesso limitado a quem já a usava (o Google recomenda 3.5 Flash-Lite ou
3.8 Flash para projetos novos). Para **adicionar um modelo**, inclua um objeto no `models.json`.
Para **calibrar a heurística**, use a seção "Calibração" do app.

## Memória

Medido num ambiente limpo só com o `requirements.txt` (sem `torch` nem `transformers`), com os 7
Gemini carregados e um PDF + imagem no contexto: **cerca de 300 MB**. O plano gratuito do
Streamlit Community Cloud oferece de 690 MB a 2,7 GB.

Não troque as dependências diretas `sentencepiece` + `protobuf` pelo extra
`google-genai[local-tokenizer]`: ele instala `torch`, `torchvision` e `transformers`, centenas de MB
que o app não usa.

## Estrutura

```
app.py                           UI (Streamlit)
core/context.py                  montagem do contexto (blocos, prompt interno, arquivos, tools, JSON)
core/counters.py                 heurística, tokenizadores locais, APIs de contagem, estimativa por modelo
core/extractors.py               extração de texto e dimensões de imagens/páginas
core/vision.py                   fórmulas de tokens visuais
core/pricing.py                  catálogo de modelos, consumo e custo
models.json                      catálogo publicado (só Google)
catalogo/todos_os_provedores.json  catálogo completo (uso local)
.streamlit/config.toml           configuração do Streamlit
requirements.txt                 dependências da versão publicada
requirements-dev.txt             testes + libs dos demais provedores
tests/                           testes unitários
```

## Limitações conhecidas

- Gemini 3.8 Flash e 3.5 Flash-Lite usam o tokenizador do Gemma 4 como **proxy**.
- Tokens visuais são estimativas por fórmula. No Gemini 2.5, a não cobrança do texto do PDF e o
  piso de 256 px por bloco de imagem são supostos (a doc só os afirma para o Gemini 3).
