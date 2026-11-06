from types import SimpleNamespace

import pytest
from google.genai import types

from app.agent.agent import Agent
from app.agent.tools import ToolRegistry
from app.memory.store import SessionMemory


def text_response(text):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(
        text=text, function_calls=None, candidates=[SimpleNamespace(content=content)]
    )


def call_response(name, **args):
    call = types.FunctionCall(name=name, args=args)
    content = types.Content(role="model", parts=[types.Part(function_call=call)])
    return SimpleNamespace(
        text=None, function_calls=[call], candidates=[SimpleNamespace(content=content)]
    )


class FakeLLM:
    """Devolve respostas roteirizadas; embeddings determinísticos por palavra-chave."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    def generate(self, contents, system_instruction, tools=None):
        self.calls.append(list(contents))
        return self.responses.pop(0) if self.responses else text_response("fim")

    def embed(self, texts, task_type):
        keys = ["aurora", "wikipedia", "orçamento"]
        return [[float(k in t.lower()) for k in keys] for t in texts]


class FakeIndex:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def search(self, query):
        if self.error:
            raise self.error
        return self.result or {"results": [], "sources": []}


@pytest.fixture
def make_agent():
    def _make(responses, index=None, max_iterations=5, memory=None):
        llm = FakeLLM(responses)
        tools = ToolRegistry(index or FakeIndex())
        agent = Agent(llm, memory or SessionMemory(), tools, max_iterations)
        return agent, llm

    return _make
