"""Production storage adapter: Cloudflare R2 (S3-compatible).

Shares the same `Storage` contract as `LocalFileStorage` so `STORAGE=r2` is a
config change, not a rewrite. Its tests skip themselves without live R2
credentials — see `tests/unit/storage/test_storage_contract.py`.
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config

from .base import StorageError


class R2Storage:
    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        public_base: str,
    ) -> None:
        self.bucket = bucket
        self.public_base = public_base.rstrip("/")
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

    def put_presigned(
        self,
        key: str,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,  # no per-URL byte cap on R2/S3; enforce via bucket policy
        expires_in: int = 900,
    ) -> str:
        params: dict[str, str] = {"Bucket": self.bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        result: str = self._client.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=expires_in
        )
        return result

    def get_url(self, key: str) -> str:
        return f"{self.public_base}/{key}"

    def get_object_by_url(self, url: str) -> bytes:
        prefix = f"{self.public_base}/"
        if not url.startswith(prefix):
            raise StorageError(f"url not served by this storage: {url!r}")
        key = url[len(prefix) :]
        body: bytes = self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return body

    def put_object(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key, "Body": data}
        if content_type:
            params["ContentType"] = content_type
        self._client.put_object(**params)
        return self.get_url(key)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if not keys:
                continue
            # delete_objects caps at 1000 keys per call; list pages already
            # come back in <=1000-key chunks so this never needs sub-batching.
            resp = self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
            if errors := resp.get("Errors"):
                raise StorageError(f"R2 delete_prefix({prefix!r}) failed: {errors}")
