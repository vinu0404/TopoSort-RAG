"""
Base class for Speech-to-Text providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class STTResult:
    """Result returned by any STT provider."""

    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration_ms: Optional[int] = None
    provider: str = ""
    raw: dict = field(default_factory=dict)


class BaseSTTProvider(ABC):
    """Abstract Speech-to-Text provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/webm",
        language: Optional[str] = None,
    ) -> STTResult:
        """Transcribe audio bytes to text."""
        ...
