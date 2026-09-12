"""Task-local settings. Never load the token from a different task implicitly."""
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


def load_settings():
    # Import on use to avoid circular module imports and side effects at import time.
    from shared7 import rag
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
    url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname not in
            ('localhost', '127.0.0.1', '::1', 'host.docker.internal') or
            parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError('OLLAMA_URL must point to local Ollama or host.docker.internal')
    model = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
    device = os.getenv('EMBEDDING_DEVICE', 'cpu')
    offline = os.getenv('EMBEDDING_OFFLINE', 'true').lower()
    if model != 'qwen3:8b' or device not in ('cpu', 'cuda') or offline not in ('true', 'false'):
        raise ValueError('Invalid model, embedding device or offline setting')
    return rag.Settings(ollama_url=url, llm_model=model, embedding_device=device,
                        offline_embeddings=offline == 'true')
