"""Verifies the caller's identity.

`AUTH_MODE=mock` (tests, local dev without Clerk keys): the bearer token
itself is the user id, no cryptographic check, just "some token was
provided" — lets tests exercise more than one user without minting real
JWTs. `AUTH_MODE=clerk` verifies a real Clerk session JWT against its JWKS
(the Frontend API URL plus `/.well-known/jwks.json`), cached by
`PyJWKClient` (refetches only on an unknown `kid`, its own built-in
behavior). Either way, a request with no bearer token at all is 401.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient

from app.config import Settings, get_settings


class AuthError(HTTPException):
    def __init__(self, detail: str = "Unauthorized") -> None:
        super().__init__(status_code=401, detail=detail)


@dataclass(frozen=True)
class Identity:
    user_id: str
    email: str | None = None


@lru_cache
def _jwks_client(jwks_url: str) -> PyJWKClient:
    return PyJWKClient(jwks_url)


def _extract_bearer_token(request: Request) -> str:
    header = request.headers.get("authorization")
    if not header or not header.startswith("Bearer ") or len(header) <= len("Bearer "):
        raise AuthError("missing bearer token")
    return header.removeprefix("Bearer ")


def _verify_clerk_jwt(token: str, jwks_url: str) -> Identity:
    client = _jwks_client(jwks_url)
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, signing_key.key, algorithms=["RS256"], options={"verify_aud": False}
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from None
    sub = claims.get("sub")
    if not sub:
        raise AuthError("token missing sub claim")
    return Identity(user_id=str(sub), email=claims.get("email"))


def resolve_identity(request: Request, settings: Settings) -> Identity:
    token = _extract_bearer_token(request)
    if settings.auth_mode == "mock":
        return Identity(user_id=token)
    if not settings.clerk_jwks_url:
        raise AuthError("CLERK_JWKS_URL not configured")
    return _verify_clerk_jwt(token, settings.clerk_jwks_url)


async def get_identity(request: Request) -> Identity:
    return resolve_identity(request, get_settings())
