"""
AWS Polly Text-to-Speech provider.

Uses aioboto3 for fully async Polly calls with the Neural engine.
"""

from __future__ import annotations

import logging
from typing import Optional

import aioboto3

from config.settings import config
from voice.tts.base import BaseTTSProvider, TTSResult

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "Kajal" 
_SUPPORTED_VOICES = {
    "en": "Matthew",
    "en-us": "Matthew",
    "en-gb": "Amy",
    "hi": "Kajal",
    "es": "Sergio",
    "fr": "Lea",
    "de": "Vicki",
    "ja": "Kazuha",
    "ko": "Seoyeon",
    "zh": "Zhiyu",
}


class AWSPollyTTSProvider(BaseTTSProvider):
    """AWS Polly async TTS using Neural engine."""

    @property
    def provider_name(self) -> str:
        return "aws_polly"

    def _get_voice(self, voice: str | None, language: str | None) -> str:
        """Resolve voice ID from explicit param, language, or config default."""
        if voice:
            return voice
        if config.aws_polly_voice_id:
            return config.aws_polly_voice_id
        if language:
            return _SUPPORTED_VOICES.get(language.lower(), _DEFAULT_VOICE)
        return _DEFAULT_VOICE

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        if not config.aws_access_key_id or not config.aws_secret_access_key:
            raise RuntimeError(
                "AWS credentials are not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env "
                "to use voice synthesis."
            )

        voice_id = self._get_voice(voice, language)

        session = aioboto3.Session(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_region,
        )

        async with session.client("polly") as polly:
            response = await polly.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=voice_id,
                Engine="neural",
            )

            audio_stream = response["AudioStream"]
            audio_bytes = await audio_stream.read()

            logger.info(
                "Polly TTS done: voice=%s, %d bytes",
                voice_id, len(audio_bytes),
            )

            return TTSResult(
                audio_bytes=audio_bytes,
                content_type="audio/mpeg",
                provider="aws_polly",
                voice=voice_id,
                raw={
                    "content_type": response.get("ContentType", "audio/mpeg"),
                    "request_characters": len(text),
                },
            )
