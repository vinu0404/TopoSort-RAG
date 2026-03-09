"""
S3-compatible cloud storage — Backblaze B2, AWS S3, MinIO, etc.

All file I/O is centralised here.  The rest of the codebase calls
``upload_file`` and ``generate_presigned_url`` without knowing
which cloud provider is behind it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config.settings import config

logger = logging.getLogger(__name__)

_s3_client = None


def _get_client():
    """Lazy-initialised, reusable S3 client."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    if not config.s3_endpoint or not config.s3_access_key_id:
        raise RuntimeError(
            "S3 storage is not configured.  Set S3_ENDPOINT, "
            "S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_BUCKET in .env"
        )

    _s3_client = boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint,
        region_name=config.s3_region,
        aws_access_key_id=config.s3_access_key_id,
        aws_secret_access_key=config.s3_secret_access_key,
        config=BotoConfig(signature_version="s3v4"),
    )
    return _s3_client


# ── Helpers ──────────────────────────────────────────────────────────────────

_SAFE_FILENAME_RE = re.compile(r"[^\w.\-() ]", re.ASCII)


def _sanitize_filename(name: str) -> str:
    """Strip characters that could cause issues in S3 keys."""
    return _SAFE_FILENAME_RE.sub("_", name).strip("_")


def build_storage_key(user_id: str, doc_id: str, filename: str) -> str:
    """
    Deterministic storage path:  ``uploads/{user_id}/{doc_id}/{filename}``

    User isolation is enforced at the path level — even if the DB
    ownership check were somehow bypassed, a pre-signed URL would
    only point at files inside the correct user's folder.
    """
    safe_name = _sanitize_filename(filename)
    return f"uploads/{user_id}/{doc_id}/{safe_name}"


# ── Upload ───────────────────────────────────────────────────────────────────

def upload_file(
    file_bytes: bytes,
    storage_key: str,
    content_type: str = "application/octet-stream",
    bucket: str | None = None,
) -> dict[str, Any]:
    """
    Upload raw bytes to the configured S3 bucket.

    Returns ``{"bucket": str, "key": str, "size": int}``.
    """
    bucket = bucket or config.s3_bucket
    client = _get_client()

    client.put_object(
        Bucket=bucket,
        Key=storage_key,
        Body=file_bytes,
        ContentType=content_type,
    )

    logger.info("Uploaded %s (%d bytes) to s3://%s/%s",
                storage_key.rsplit("/", 1)[-1],
                len(file_bytes), bucket, storage_key)

    return {"bucket": bucket, "key": storage_key, "size": len(file_bytes)}


# ── Pre-signed URL ───────────────────────────────────────────────────────────

def generate_presigned_url(
    storage_key: str,
    bucket: str | None = None,
    expires_in: int | None = None,
) -> str:
    """
    Generate a time-limited pre-signed URL for reading an object.

    The URL is valid for ``expires_in`` seconds (defaults to
    ``config.s3_presign_expiry``, typically 300 s / 5 min).
    """
    bucket = bucket or config.s3_bucket
    expires_in = expires_in or config.s3_presign_expiry
    client = _get_client()

    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": storage_key},
            ExpiresIn=expires_in,
        )
    except ClientError:
        logger.exception("Failed to generate pre-signed URL for %s", storage_key)
        raise

    return url


# ── Delete ───────────────────────────────────────────────────────────────────

def delete_file(
    storage_key: str,
    bucket: str | None = None,
) -> None:
    """Delete an object from S3. Silently succeeds if the key doesn't exist."""
    bucket = bucket or config.s3_bucket
    client = _get_client()

    try:
        client.delete_object(Bucket=bucket, Key=storage_key)
        logger.info("Deleted s3://%s/%s", bucket, storage_key)
    except ClientError:
        logger.exception("Failed to delete S3 object %s", storage_key)
        raise
