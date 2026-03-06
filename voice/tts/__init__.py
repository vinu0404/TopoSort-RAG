"""TTS provider factory."""

from __future__ import annotations

from voice.tts.base import BaseTTSProvider, TTSResult


def get_tts_provider() -> BaseTTSProvider:
    """Return the configured TTS provider instance."""
    from config.settings import config

    name = (config.tts_provider or "aws_polly").lower()

    if name in ("aws_polly", "polly"):
        from voice.tts.aws_polly import AWSPollyTTSProvider
        return AWSPollyTTSProvider()

    raise ValueError(f"Unknown TTS provider: {name}")


__all__ = ["get_tts_provider", "BaseTTSProvider", "TTSResult"]
