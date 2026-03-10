"""
Thin adapter layer over LLM provider SDKs (OpenAI, Anthropic, …).

Each provider exposes the same interface so callers never import
provider-specific code.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from config.settings import config

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Structured return value from LLM generate() calls.

    Attributes:
        text:  Raw text response from the LLM.
        data:  Parsed JSON dict when output_schema was requested, else None.
        usage: Token usage from the provider API.
        model: The model that actually served the request.
    """
    text: str = ""
    data: Dict[str, Any] | None = None
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    model: str = ""


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
    ) -> LLMResult:
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
    ) -> LLMResult:
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
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        parsed_data = None
        if output_schema is not None:
            try:
                parsed_data = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("LLM did not return valid JSON; returning raw text")
                parsed_data = {"raw": text}

        return LLMResult(text=text, data=parsed_data, usage=usage, model=model)

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
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


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
    ) -> LLMResult:
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
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            }

        parsed_data = None
        if output_schema is not None:
            try:
                parsed_data = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("LLM did not return valid JSON; returning raw text")
                parsed_data = {"raw": text}

        return LLMResult(text=text, data=parsed_data, usage=usage, model=model)

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
# Google (Gemini)
# ═══════════════════════════════════════════════════════════════════════════════


class GoogleProvider(BaseLLMProvider):
    def __init__(self, api_key: str, default_model: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        import asyncio

        model_name = model or self.default_model

        extra = ""
        if output_schema is not None:
            extra = (
                f"\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{json.dumps(output_schema, indent=2)}"
            )

        gen_model = self._genai.GenerativeModel(
            model_name,
            generation_config=self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        response = await asyncio.to_thread(
            gen_model.generate_content, prompt + extra
        )

        text = response.text or ""
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        parsed_data = None
        if output_schema is not None:
            try:
                parsed_data = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("LLM did not return valid JSON; returning raw text")
                parsed_data = {"raw": text}

        return LLMResult(text=text, data=parsed_data, usage=usage, model=model_name)

    async def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        import asyncio

        model_name = model or self.default_model

        gen_model = self._genai.GenerativeModel(
            model_name,
            generation_config=self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        response = await asyncio.to_thread(
            gen_model.generate_content, prompt, stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text


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
    elif provider_name == "google":
        key = api_key or (config.google_api_key or "")
        instance = GoogleProvider(
            api_key=key,
            default_model=default_model or "gemini-2.0-flash",
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

    _provider_cache[cache_key] = instance
    return instance
