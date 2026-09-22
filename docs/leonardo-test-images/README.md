# Leonardo live test images (task 4.3)

Three real images generated live against a real Leonardo account on
2026-09-08, confirming task 4.3's accept line ("live call produces 3
images for one prompt"). Model: Phoenix 1.0
(`de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3`), seed `12345`, no Character
Reference (identity was a placeholder trigger token — Phase 5 doesn't
exist yet).

Prompt used:

> `[IDENTITY], a person with a warm smile, simple clean linework, a cozy
> kitchen, a chipped ceramic mug on the counter, morning light through a
> window, wide shot`

The exact request/response pair is frozen as
`tests/cassettes/leonardo_generation_001.json` and replayed in
`tests/unit/llm/test_leonardo_provider.py`.

One of the three (`240e6f90-...`) came back flagged `nsfw: true` by
Leonardo despite being an entirely SFW image — a false positive worth
knowing about; no moderation/re-roll handling exists yet (see
`services/worker/app/images/leonardo.py`'s module docstring).
