import logging
import time

import httpx
from google import genai
from google.genai import errors, types

from app import config

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model: str = config.MODEL):
        self.model = model
        self.client = genai.Client(
            api_key=api_key or config.get_api_key(),
            http_options=types.HttpOptions(timeout=config.TIMEOUT_MS),
        )

    def _call(self, fn):
        last: Exception | None = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                return fn()
            except errors.APIError as e:
                retryable = e.code in (429, 500, 502, 503, 504)
                last = (
                    LLMRateLimitError("Limite de requisições atingido (429)")
                    if e.code == 429
                    else LLMError(f"Erro da API Gemini ({e.code})")
                )
                if not retryable:
                    raise last from e
                log.warning("tentativa %d falhou: código %s", attempt, e.code)
            except (httpx.TimeoutException, TimeoutError) as e:
                last = LLMTimeoutError("Timeout ao chamar o Gemini")
                log.warning("tentativa %d: timeout", attempt)
            if attempt < config.MAX_RETRIES:
                time.sleep(2**attempt)
        raise last  # type: ignore[misc]

    def generate(self, contents, system_instruction: str, tools=None):
        cfg = types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return self._call(
            lambda: self.client.models.generate_content(
                model=self.model, contents=contents, config=cfg
            )
        )

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        cfg = types.EmbedContentConfig(task_type=task_type)
        res = self._call(
            lambda: self.client.models.embed_content(
                model=config.EMBED_MODEL, contents=texts, config=cfg
            )
        )
        return [e.values for e in res.embeddings]
