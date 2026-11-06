from app.agent import tools as tools_mod
from app.agent.agent import LIMIT_MESSAGE
from tests.conftest import FakeIndex, call_response, text_response


def test_resposta_direta_sem_ferramenta(make_agent):
    agent, _ = make_agent([text_response("Olá!")])
    r = agent.chat("s", "oi")
    assert r.answer == "Olá!" and r.tools_used == [] and r.iterations == 1


def test_tool_calling_coleta_fontes(make_agent):
    idx = FakeIndex({"results": [{"text": "R$ 480"}], "sources": ["docs/a.md"]})
    agent, llm = make_agent(
        [call_response("search_documents", query="orçamento"), text_response("R$ 480")],
        index=idx,
    )
    r = agent.chat("s", "orçamento?")
    assert r.answer == "R$ 480"
    assert r.tools_used == ["search_documents"]
    assert r.sources == ["docs/a.md"]
    # a 2ª chamada ao LLM carrega a resposta da ferramenta
    last = llm.calls[1][-1]
    assert last.parts[0].function_response.name == "search_documents"


def test_falha_da_ferramenta_vira_erro_para_o_modelo(make_agent):
    agent, llm = make_agent(
        [call_response("search_documents", query="x"), text_response("A busca falhou.")],
        index=FakeIndex(error=RuntimeError("boom")),
    )
    r = agent.chat("s", "x")
    resp = llm.calls[1][-1].parts[0].function_response.response
    assert "error" in resp and "boom" in resp["error"]
    assert r.answer == "A busca falhou."


def test_ferramenta_desconhecida(make_agent):
    agent, llm = make_agent([call_response("inexistente"), text_response("ok")])
    agent.chat("s", "x")
    assert "error" in llm.calls[1][-1].parts[0].function_response.response


def test_limite_de_iteracoes(make_agent):
    loop = [call_response("search_documents", query="x") for _ in range(10)]
    agent, llm = make_agent(loop, max_iterations=3)
    r = agent.chat("s", "x")
    assert r.answer == LIMIT_MESSAGE and r.iterations == 3
    assert len(llm.calls) == 3


def test_historico_vai_para_o_llm(make_agent):
    agent, llm = make_agent([text_response("a"), text_response("b")])
    agent.chat("s", "primeira")
    agent.chat("s", "segunda")
    roles = [c.role for c in llm.calls[1]]
    assert roles == ["user", "model", "user"]


def test_sessoes_isoladas(make_agent):
    agent, llm = make_agent([text_response("a"), text_response("b")])
    agent.chat("s1", "um")
    agent.chat("s2", "dois")
    assert len(llm.calls[1]) == 1
