"""
Base class for Text-to-Speech providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TTSResult:
    """Result returned by any TTS provider."""

    audio_bytes: bytes
    content_type: str = "audio/mpeg"
    duration_ms: Optional[int] = None
    provider: str = ""
    voice: str = ""
    raw: dict = field(default_factory=dict)


class BaseTTSProvider(ABC):
    """Abstract Text-to-Speech provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Convert text to audio bytes."""
        ...
