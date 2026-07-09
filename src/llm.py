"""LLM provider factory with resilience built in.

Primary: Google Gemini (as suggested by the assignment).
Fallback: any OpenRouter model via the OpenAI-compatible API - engaged
automatically by LangChain's `with_fallbacks` when the primary provider
errors out (rate limit, downtime). Each provider additionally retries
transient errors internally (max_retries), so a single blip never surfaces
to the user.
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.settings import settings


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    primary = ChatGoogleGenerativeAI(
        model=settings.model_name,
        temperature=0.1,
        max_retries=2,
        google_api_key=settings.google_api_key or None,
    )
    if settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI

        fallback = ChatOpenAI(
            model=settings.openrouter_model,
            temperature=0.1,
            max_retries=2,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return primary.with_fallbacks([fallback])
    return primary
