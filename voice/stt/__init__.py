"""STT provider factory."""

from __future__ import annotations

from voice.stt.base import BaseSTTProvider, STTResult


def get_stt_provider() -> BaseSTTProvider:
    """Return the configured STT provider instance."""
    from config.settings import config

    name = (config.stt_provider or "assemblyai").lower()

    if name == "assemblyai":
        from voice.stt.assemblyai import AssemblyAISTTProvider
        return AssemblyAISTTProvider()

    raise ValueError(f"Unknown STT provider: {name}")


__all__ = ["get_stt_provider", "BaseSTTProvider", "STTResult"]
