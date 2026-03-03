from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embed_model: str = "nomic-embed-text"
    chroma_persist_dir: str = "./data/vectorstore"
    documents_dir: str = "./data/documents"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 5
    temperature: float = 0.1
    watch_documents_dir: bool = False
    log_level: str = "INFO"
    api_key: str = ""            # empty = auth disabled
    ollama_concurrency: int = 1  # max concurrent Ollama requests


settings = Settings()
