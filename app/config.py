import logging
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = "models/gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
TIMEOUT_MS = 30000
MAX_RETRIES = 3
MAX_ITERATIONS = 5
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
DATA_DIR = os.getenv("DATA_DIR", "data")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def get_api_key() -> str:
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError("API_KEY não definida no .env")
    return key

logging.getLogger("httpx").setLevel(logging.WARNING)
