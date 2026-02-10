"""
Thin adapter layer over LLM provider SDKs (OpenAI, Anthropic, …).

Each provider exposes the same interface so callers never import
provider-specific code.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from config.settings import config

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Common interface that every concrete provider implements."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | str:
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, default_model: str = "gpt-4o"):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | str:
        model = model or self.default_model

        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {}
        if output_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
            messages[0]["content"] += (
                f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(output_schema, indent=2)}"
            )

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        text = response.choices[0].message.content or ""

        if output_schema is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning("LLM did not return valid JSON; returning raw text")
                return {"raw": text}

        return text

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        model = model or self.default_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════════════════════════════


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, default_model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | str:
        model = model or self.default_model

        extra_instruction = ""
        if output_schema is not None:
            extra_instruction = (
                f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(output_schema, indent=2)}"
            )

        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt + extra_instruction}],
        )

        text = response.content[0].text

        if output_schema is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning("LLM did not return valid JSON; returning raw text")
                return {"raw": text}

        return text

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        model = model or self.default_model

        async with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

_provider_cache: Dict[str, BaseLLMProvider] = {}


def get_llm_provider(
    provider_name: str,
    *,
    api_key: str | None = None,
    default_model: str | None = None,
) -> BaseLLMProvider:
    """
    Return (and cache) an LLM provider instance.

    Parameters
    ----------
    provider_name : "openai" | "anthropic"
    api_key       : explicit key; if omitted, read from config.
    default_model : override the default model for this provider instance.
    """

    cache_key = f"{provider_name}:{default_model or 'default'}"
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    if provider_name == "openai":
        key = api_key or config.openai_api_key
        instance = OpenAIProvider(api_key=key, default_model=default_model or "gpt-4o")
    elif provider_name == "anthropic":
        key = api_key or (config.anthropic_api_key or "")
        instance = AnthropicProvider(
            api_key=key,
            default_model=default_model or "claude-3-5-sonnet-20241022",
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

    _provider_cache[cache_key] = instance
    return instance
