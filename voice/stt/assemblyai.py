"""
AssemblyAI Speech-to-Text provider.

Flow:  upload audio → create transcript → poll until complete.
Uses raw httpx calls for full async control (no sync SDK dependency).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from config.settings import config
from voice.stt.base import BaseSTTProvider, STTResult

logger = logging.getLogger(__name__)

_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
_POLL_INTERVAL = 2.0  # seconds between status polls
_MAX_POLL_ATTEMPTS = 120  # 2 minutes max wait


class AssemblyAISTTProvider(BaseSTTProvider):
    """AssemblyAI async STT using REST API."""

    @property
    def provider_name(self) -> str:
        return "assemblyai"

    def _headers(self) -> dict:
        return {"authorization": config.assemblyai_api_key}

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/webm",
        language: Optional[str] = None,
    ) -> STTResult:
        if not config.assemblyai_api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY is not configured. "
                "Set it in .env to use voice transcription."
            )

        async with httpx.AsyncClient(timeout=30) as client:
            upload_resp = await client.post(
                _UPLOAD_URL,
                headers={**self._headers(), "content-type": "application/octet-stream"},
                content=audio_bytes,
            )
            upload_resp.raise_for_status()
            upload_url = upload_resp.json()["upload_url"]
            logger.debug("AssemblyAI upload done: %s", upload_url)
            body: dict = {"audio_url": upload_url}
            if language:
                body["language_code"] = language

            create_resp = await client.post(
                _TRANSCRIPT_URL,
                headers=self._headers(),
                json=body,
            )
            create_resp.raise_for_status()
            transcript_id = create_resp.json()["id"]
            logger.debug("AssemblyAI transcript created: %s", transcript_id)
            poll_url = f"{_TRANSCRIPT_URL}/{transcript_id}"
            for _ in range(_MAX_POLL_ATTEMPTS):
                poll_resp = await client.get(poll_url, headers=self._headers())
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data["status"]

                if status == "completed":
                    logger.info(
                        "AssemblyAI transcription done (%d ms, confidence=%.2f)",
                        data.get("audio_duration", 0) * 1000,
                        data.get("confidence", 0),
                    )
                    return STTResult(
                        text=data.get("text", ""),
                        confidence=data.get("confidence"),
                        language=data.get("language_code"),
                        duration_ms=int(data.get("audio_duration", 0) * 1000),
                        provider="assemblyai",
                        raw=data,
                    )

                if status == "error":
                    error_msg = data.get("error", "Unknown transcription error")
                    logger.error("AssemblyAI transcription failed: %s", error_msg)
                    raise RuntimeError(f"Transcription failed: {error_msg}")

                await asyncio.sleep(_POLL_INTERVAL)

            raise TimeoutError(
                f"AssemblyAI transcription timed out after {_MAX_POLL_ATTEMPTS} polls"
            )
