"""Vision content-block mapping (the part that doesn't need a live API
call). Each provider's own `_content_block`/`_to_*_content` maps a
`ContentPart` to that provider's wire shape — this proves the mapping is
correct; a real vision call proving Claude/GPT can actually read the
resulting image still needs live credentials (see anthropic_provider.py /
openai_provider.py).
"""

from __future__ import annotations

import base64

from app.llm import anthropic_provider as ap
from app.llm import openai_provider as op

_PNG_BYTES = b"\x89PNG\r\n\x1a\n"  # not a full PNG, just recognizable bytes
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode()


def test_anthropic_maps_url_image() -> None:
    block = ap._content_block({"type": "image", "url": "https://example.com/a.jpg"})
    assert block == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.jpg"}}


def test_anthropic_maps_bare_base64_image_as_jpeg() -> None:
    block = ap._content_block({"type": "image", "b64": _PNG_B64})
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(block["source"]["data"]) == _PNG_BYTES


def test_anthropic_maps_data_uri_base64_image_with_its_mime_type() -> None:
    data_uri = f"data:image/png;base64,{_PNG_B64}"
    block = ap._content_block({"type": "image", "b64": data_uri})
    assert block["source"]["media_type"] == "image/png"
    assert base64.b64decode(block["source"]["data"]) == _PNG_BYTES


def test_anthropic_maps_text_part() -> None:
    block = ap._content_block({"type": "text", "text": "what's in this image?"})
    assert block == {"type": "text", "text": "what's in this image?"}


def test_openai_maps_url_image() -> None:
    block = op._content_block({"type": "image", "url": "https://example.com/a.jpg"})
    assert block == {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}


def test_openai_maps_bare_base64_image_as_a_data_uri() -> None:
    block = op._content_block({"type": "image", "b64": _PNG_B64})
    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG_BYTES


def test_openai_maps_text_part() -> None:
    block = op._content_block({"type": "text", "text": "what's in this image?"})
    assert block == {"type": "text", "text": "what's in this image?"}
