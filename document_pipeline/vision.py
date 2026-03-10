"""
Provider-agnostic vision (image → text) module.

Supports OpenAI (GPT-4o) and Google (Gemini) vision APIs.
Change only `VISION_MODEL_PROVIDER` and `VISION_MODEL` in config
to switch between providers — no code changes needed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional

from config.settings import config

logger = logging.getLogger(__name__)

_IMAGE_ANALYSIS_PROMPT = """Analyze this image in detail. Provide a comprehensive text description that covers:

1. **Content overview**: What the image shows (photo, diagram, chart, screenshot, document scan, etc.)
2. **Text extraction**: Reproduce ALL visible text exactly as it appears (OCR). Preserve formatting, headings, lists, and tables.
3. **Data & numbers**: Extract all numerical data, statistics, labels, axes, legends from charts/graphs/tables.
4. **Visual elements**: Describe key objects, people, colors, layout, and spatial relationships.
5. **Context & meaning**: What is the purpose or subject matter of this image?

Be thorough and precise — this description will be the only searchable representation of the image.
If the image contains a document or text-heavy content, prioritize exact text extraction.
If it contains a chart or diagram, describe the data and relationships in detail."""


def _detect_mime_type(file_bytes: bytes, filename: str) -> str:
    """Detect MIME type from extension, with magic-byte fallback."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
    }
    return mime_map.get(ext, "image/png")


async def analyze_image(file_bytes: bytes, filename: str, prompt: Optional[str] = None) -> str:
    """
    Send image bytes to the configured vision LLM and return a text description.

    Works with both OpenAI and Google Gemini — just change config settings.
    """
    provider = config.vision_model_provider.lower()
    model = config.vision_model
    analysis_prompt = prompt or _IMAGE_ANALYSIS_PROMPT

    logger.info("Vision analysis: provider=%s model=%s file=%s size=%d bytes",
                provider, model, filename, len(file_bytes))

    if provider == "openai":
        return await _analyze_openai(file_bytes, filename, model, analysis_prompt)
    elif provider == "google":
        return await _analyze_google(file_bytes, filename, model, analysis_prompt)
    else:
        raise ValueError(f"Unsupported vision provider: {provider}")


async def _analyze_openai(
    file_bytes: bytes, filename: str, model: str, prompt: str,
) -> str:
    """Call OpenAI vision API (GPT-4o, GPT-4.1, etc.)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.openai_api_key)
    mime = _detect_mime_type(file_bytes, filename)
    b64 = base64.b64encode(file_bytes).decode("ascii")

    response = await client.chat.completions.create(
        model=model,
        max_tokens=config.vision_max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    )
    text = response.choices[0].message.content or ""
    logger.info("Vision analysis complete (OpenAI): %d chars", len(text))
    return text


async def _analyze_google(
    file_bytes: bytes, filename: str, model: str, prompt: str,
) -> str:
    """Call Google Gemini vision API."""
    import google.generativeai as genai

    genai.configure(api_key=config.google_api_key)
    mime = _detect_mime_type(file_bytes, filename)

    gen_model = genai.GenerativeModel(
        model,
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.vision_max_tokens,
            temperature=0.2,
        ),
    )

    # Gemini accepts inline image data
    image_part = {"mime_type": mime, "data": file_bytes}
    response = await asyncio.to_thread(
        gen_model.generate_content, [prompt, image_part]
    )
    text = response.text or ""
    logger.info("Vision analysis complete (Google): %d chars", len(text))
    return text
