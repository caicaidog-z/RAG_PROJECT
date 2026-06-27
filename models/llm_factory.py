from functools import lru_cache

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from config.settings import AppSettings, get_settings


def build_llm(settings: AppSettings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    api_key = settings.require_chat_api_key()
    return ChatOpenAI(
        temperature=0,
        model=settings.model_name,
        api_key=api_key,
        base_url=settings.openai_base_url,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return build_llm()


def build_web_search_tool(settings: AppSettings | None = None) -> TavilySearchResults:
    settings = settings or get_settings()
    return TavilySearchResults(max_results=settings.web_search_max_results)


@lru_cache(maxsize=1)
def get_web_search_tool() -> TavilySearchResults:
    return build_web_search_tool()
