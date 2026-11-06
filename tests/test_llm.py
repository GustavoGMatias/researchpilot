import httpx
import pytest
from google.genai import errors

from app.services import llm as llm_mod
from app.services.llm import GeminiLLM, LLMError, LLMRateLimitError, LLMTimeoutError


@pytest.fixture
def gem(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    return GeminiLLM(api_key="fake")


def api_error(code):
    return errors.APIError(code, {"error": {"message": "x", "status": "X"}})


def flaky(*excs, result="ok"):
    seq = list(excs)

    def fn():
        if seq:
            raise seq.pop(0)
        return result

    return fn


def test_retry_e_sucesso(gem):
    assert gem._call(flaky(api_error(503), httpx.ReadTimeout("t"))) == "ok"


def test_429_esgota_retries(gem):
    with pytest.raises(LLMRateLimitError):
        gem._call(flaky(*[api_error(429)] * 5))


def test_timeout_esgota_retries(gem):
    with pytest.raises(LLMTimeoutError):
        gem._call(flaky(*[httpx.ReadTimeout("t")] * 5))


def test_erro_nao_retentavel_falha_direto(gem):
    fn = flaky(api_error(400), api_error(400))
    with pytest.raises(LLMError):
        gem._call(fn)
