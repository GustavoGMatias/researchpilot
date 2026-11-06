# ResearchPilot

Um agente de pesquisa em Python que construí para estudar como agentes de IA funcionam por dentro. Ele conversa com a API do Gemini, decide sozinho quando precisa de uma ferramenta, pesquisa na Wikipedia e nos meus documentos locais, e responde citando as fontes (ou dizendo que não achou informação suficiente).

Escrevi o loop de tool calling à mão, sem LangChain ou framework de agentes, para entender cada passo.

## Arquitetura

```
 cliente ──► FastAPI (app/api) ──► Agent (app/agent) ──► GeminiLLM (app/services/llm.py)
 /chat /documents /health              │  ▲                  retry, timeout, 429
                                       │  │ resultados
                                       ▼  │
                                 ToolRegistry
                                  ├─ search_wikipedia ──► API pública da Wikipedia
                                  └─ search_documents ──► DocumentIndex (RAG)
                                                            embeddings gemini-embedding-001
                                                            cosseno em Python puro, JSON em data/
                                 SessionMemory (app/memory) ──► data/sessions.json
```

Como o loop funciona (`app/agent/agent.py`):
1. Monto o histórico da sessão + a pergunta nova e envio ao modelo com as declarações das ferramentas.
2. Se o modelo pede uma ferramenta, eu a executo, devolvo o resultado como `function_response` e repito.
3. Se responde em texto, essa é a resposta final. Há um limite de iterações (padrão: 5) para evitar loops infinitos.
4. Só a pergunta e a resposta final entram na memória; os passos intermediários das ferramentas não.

Estrutura:

```
app/
  agent/     loop do agente e registro das ferramentas
  api/       rotas FastAPI
  services/  cliente Gemini, Wikipedia, RAG
  memory/    memória de sessão em JSON
  schemas/   modelos Pydantic
tests/       pytest, com o LLM mockado
docs/        meus documentos para a busca local
cli.py       chat no terminal
```

## Como rodar

```bash
git clone https://github.com/GustavoGMatias/researchpilot.git
cd researchpilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # coloque sua chave do Gemini em API_KEY
```

Chat no terminal:

```bash
python cli.py            # sessão "cli"; python cli.py minha_sessao para outra
```

API:

```bash
uvicorn app.api.main:app --reload
# documentação interativa em http://localhost:8000/docs
```

```bash
curl localhost:8000/health

curl -F "file=@minhas_notas.md" localhost:8000/documents

curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"Quem foi Santos Dumont?"}'
```

Resposta do `/chat`:

```json
{"session_id":"demo","answer":"...","sources":["https://pt.wikipedia.org/wiki/Santos_Dumont"],
 "tools_used":["search_wikipedia"],"iterations":2}
```

Testes (não usam a API real):

```bash
pytest -q
```

Configuração opcional por variável de ambiente: `DOCS_DIR`, `DATA_DIR`, `LOG_LEVEL`, `WIKI_USER_AGENT`.

## Decisões e tratamento de erros

- **Retry e timeout:** cada chamada ao Gemini tem timeout de 30 s e até 3 tentativas com espera exponencial em 429, 5xx e timeout. A API responde 429, 504 ou 502 conforme o caso.
- **Falha de ferramenta:** a exceção é capturada e devolvida ao modelo como `{"error": ...}`, que então avisa o usuário ou tenta outra ferramenta, em vez de derrubar a requisição.
- **RAG:** trechos de 800 caracteres com sobreposição de 100, embeddings com `RETRIEVAL_DOCUMENT` e `RETRIEVAL_QUERY`, similaridade de cosseno em Python puro. O índice é incremental: só reindexa arquivos cujo hash mudou.
- **Upload:** apenas `.txt` e `.md` em UTF-8 até 1 MB; o nome do arquivo é saneado para não escapar de `docs/`.
- **Testes:** o LLM é substituído por um roteiro de respostas, então testo o loop, o limite de iterações, as falhas de ferramenta, o retry e a API sem gastar cota.

## Limitações

- **Limiar do RAG calibrado com pouco material.** O corte de similaridade (0.65) foi ajustado olhando poucos documentos; com um acervo maior pode precisar de ajuste.
- **Citação depende do modelo.** Já vi o modelo listar fontes pouco relevantes ao admitir que não tinha a informação. O campo `sources` da API reflete o que as ferramentas retornaram, não uma verificação de que cada fonte sustenta a resposta.
- **Busca linear.** Comparo a pergunta com todos os trechos; funciona para dezenas de documentos, não para milhares (aí seria preciso um banco vetorial).
- **Concorrência simples.** A memória é um JSON com lock dentro do processo; duas requisições simultâneas na mesma sessão podem se intercalar, e não há suporte a vários processos.
- **Memória truncada.** Mantenho as últimas 40 mensagens por sessão, sem resumo.
- **Sem autenticação** na API; não deve ser exposta na internet como está.
- **Só Wikipedia em português** e documentos de texto (sem PDF).
