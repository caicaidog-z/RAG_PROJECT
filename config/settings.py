from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_base_url: str = "https://xiaoai.plus/v1"
    model_name: str = "gpt-4o-mini"

    milvus_uri: str = "http://1.95.116.112:19530"
    collection_name: str = "t_collection01"

    chunking_embedding_provider: Literal["bge", "openai"] = "bge"
    bge_model_name: str = "BAAI/bge-small-zh-v1.5"
    bge_device: str = "cpu"
    pdf_asset_dir: str = "logs/pdf_assets"
    pdf_enable_image_ocr: bool = True
    pdf_image_ocr_score_threshold: float = 0.5
    pdf_image_ocr_min_chars: int = 8
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    web_upload_dir: str = "uploads"

    retriever_k: int = 4
    retriever_score_threshold: float = 0.1
    retriever_ranker_k: int = 100
    retriever_category_filter: str = "content"
    web_search_max_results: int = 2

    max_transform_count: int = 2

    def require_chat_api_key(self) -> str:
        if self.model_name.startswith("deepseek"):
            if not self.deepseek_api_key:
                raise ConfigurationError(
                    "缺少 DEEPSEEK_API_KEY。当前 MODEL_NAME 使用了 deepseek 模型。"
                )
            return self.deepseek_api_key

        if not self.openai_api_key:
            raise ConfigurationError(
                "缺少 OPENAI_API_KEY。请在 .env 中配置后再运行 chat。"
            )
        return self.openai_api_key

    def require_openai_embedding_key(self) -> str:
        if not self.openai_api_key:
            raise ConfigurationError(
                "缺少 OPENAI_API_KEY。当前切块配置使用 openai embedding。"
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
