"""The `[IDENTITY]` placeholder token, shared between the prompt writer and
the panel generator.

`nodes/prompts.py` writes this literal token into `ImagePrompt.positive`
exactly once, in place of describing the person's actual appearance.
`images/base.py` substitutes it for `IdentityRef.trigger_token` right before
calling a generator. Kept as its own small module so neither node has to
import the other just for one string constant.
"""

from __future__ import annotations

IDENTITY_TOKEN = "[IDENTITY]"


def apply_identity(positive: str, trigger_token: str) -> str:
    """Substitutes the placeholder for the real trigger token. Assumes
    `positive` contains `IDENTITY_TOKEN` exactly once, as `nodes/prompts.py`'s
    `constraint_errors` enforces before a prompt reaches a generator."""
    return positive.replace(IDENTITY_TOKEN, trigger_token)
