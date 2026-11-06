import logging

from google.genai import types

from app.services.wikipedia import search_wikipedia

log = logging.getLogger(__name__)

DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_wikipedia",
        description="Pesquisa na Wikipedia (fatos gerais, conceitos, pessoas, eventos).",
        parameters_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Termos de busca"}},
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="search_documents",
        description="Busca nos documentos locais do usuário (pasta docs/). Use para perguntas sobre os arquivos dele.",
        parameters_json_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Pergunta ou termos de busca"}},
            "required": ["query"],
        },
    ),
]


class ToolRegistry:
    def __init__(self, doc_index):
        self._fns = {
            "search_wikipedia": lambda query: search_wikipedia(query),
            "search_documents": lambda query: doc_index.search(query),
        }
        self.tools = [types.Tool(function_declarations=DECLARATIONS)]

    def run(self, name: str, args: dict) -> dict:
        fn = self._fns.get(name)
        if not fn:
            return {"error": f"ferramenta desconhecida: {name}"}
        try:
            return fn(**args)
        except Exception as e:
            log.exception("ferramenta %s falhou", name)
            return {"error": f"A ferramenta {name} falhou: {e}"}
