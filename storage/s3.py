"""
S3-compatible cloud storage — Backblaze B2, AWS S3, MinIO, etc.

All file I/O is centralised here.  The rest of the codebase calls
``upload_file`` and ``generate_presigned_url`` without knowing
which cloud provider is behind it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config.settings import config

if TYPE_CHECKING:
    from fastapi import UploadFile

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


def build_artifact_storage_key(
    user_id: str, conversation_id: str, artifact_id: str, filename: str,
) -> str:
    """
    Artifact storage path:  ``artifacts/{user_id}/{conversation_id}/{artifact_id}/{filename}``
    """
    safe_name = _sanitize_filename(filename)
    return f"artifacts/{user_id}/{conversation_id}/{artifact_id}/{safe_name}"


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


# ── Multipart streaming upload ───────────────────────────────────────────────

# S3 minimum part size is 5 MB (except for the last part).
# We read the file in 1 MB chunks and accumulate until we hit 5 MB,
# then flush a part — keeping RAM usage to ~5 MB per file regardless of size.
_PART_SIZE = 5 * 1024 * 1024   # 5 MB
_READ_CHUNK = 1 * 1024 * 1024  # 1 MB read granularity


async def multipart_upload_file(
    f: "UploadFile",
    storage_key: str,
    content_type: str = "application/octet-stream",
    bucket: str | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[str, int]:
    """Stream an UploadFile to S3 using multipart upload.

    Never holds the full file in memory — only one 5 MB part at a time.
    S3 part uploads run in a thread-pool executor so the event loop stays free
    to handle other requests / interleave concurrent file streams.

    Returns:
        (sha256_hex, total_bytes_written)

    Raises:
        ValueError: if the file is empty (0 bytes).
        ClientError: on any S3 API failure (multipart upload is aborted first).
    """
    bucket = bucket or config.s3_bucket
    client = _get_client()
    loop = asyncio.get_running_loop()

    hasher = hashlib.sha256()
    total_size = 0
    buffer = bytearray()
    parts: list[dict] = []
    part_number = 1

    # Initiate multipart upload — get the upload_id that ties all parts together
    mpu = await loop.run_in_executor(
        executor,
        lambda: client.create_multipart_upload(
            Bucket=bucket, Key=storage_key, ContentType=content_type
        ),
    )
    upload_id = mpu["UploadId"]

    async def _flush_part(data: bytes, pn: int) -> dict:
        resp = await loop.run_in_executor(
            executor,
            lambda: client.upload_part(
                Bucket=bucket,
                Key=storage_key,
                UploadId=upload_id,
                PartNumber=pn,
                Body=data,
            ),
        )
        return {"PartNumber": pn, "ETag": resp["ETag"]}

    async def _abort():
        try:
            await loop.run_in_executor(
                executor,
                lambda: client.abort_multipart_upload(
                    Bucket=bucket, Key=storage_key, UploadId=upload_id
                ),
            )
        except Exception:
            logger.warning("Failed to abort multipart upload %s", upload_id)

    try:
        # Stream: read 1 MB chunks, accumulate into 5 MB parts, upload each part
        while True:
            chunk = await f.read(_READ_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
            total_size += len(chunk)
            buffer.extend(chunk)

            # Flush full 5 MB parts as they accumulate
            while len(buffer) >= _PART_SIZE:
                part_data = bytes(buffer[:_PART_SIZE])
                del buffer[:_PART_SIZE]
                parts.append(await _flush_part(part_data, part_number))
                part_number += 1

        if total_size == 0:
            await _abort()
            raise ValueError(f"File is empty: {storage_key}")

        # Upload remaining bytes as the final part (may be < 5 MB — S3 allows this)
        if buffer:
            parts.append(await _flush_part(bytes(buffer), part_number))

        # Seal the multipart upload
        await loop.run_in_executor(
            executor,
            lambda: client.complete_multipart_upload(
                Bucket=bucket,
                Key=storage_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            ),
        )

    except Exception:
        await _abort()
        raise

    logger.info(
        "Multipart upload complete: %s (%d bytes, %d part(s)) → s3://%s/%s",
        storage_key.rsplit("/", 1)[-1],
        total_size,
        len(parts),
        bucket,
        storage_key,
    )
    return hasher.hexdigest(), total_size


# ── Download ──────────────────────────────────────────────────────────────────

def download_file(
    storage_key: str,
    bucket: str | None = None,
) -> bytes:
    """
    Download file bytes from S3.

    Used by Celery workers to retrieve files uploaded by the API.
    """
    bucket = bucket or config.s3_bucket
    client = _get_client()

    try:
        response = client.get_object(Bucket=bucket, Key=storage_key)
        file_bytes = response["Body"].read()
    except ClientError:
        logger.exception("Failed to download S3 object %s", storage_key)
        raise

    logger.info("Downloaded %s (%d bytes) from s3://%s/%s",
                storage_key.rsplit("/", 1)[-1],
                len(file_bytes), bucket, storage_key)

    return file_bytes


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
