# ResearchPilot

Agente de IA de pesquisa em Python: Gemini API (google-genai) com tool calling, FastAPI, Pydantic, RAG simples e memória. Projeto de portfólio, solo, do Gustavo (Ciência da Computação, 3º período).

## Como trabalhar
- Vá direto ao ponto: construa, rode e teste. Explique só o que é importante, em poucas linhas e sem jargão.
- Ao fim de cada etapa, dê 2 linhas do que eu preciso saber explicar numa entrevista sobre aquela parte.
- Comentários mínimos no código.
- Comandos de terminal sempre prontos pra colar (Ubuntu).
- Commits pequenos, com mensagem descritiva. Rode git status antes de commitar.
- README em primeira pessoa do singular.

## Regras de segurança
- Nunca abra, leia, mostre ou imprima o .env nem a API key.
- A chave está em API_KEY no .env (carregada com python-dotenv).
- Modelo: models/gemini-3.5-flash-lite (o 3.8-flash deu timeout). Use timeout e retry nas chamadas.
