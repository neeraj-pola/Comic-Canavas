"""ml/identity/train_lora.py's pure config-building logic.
`_training_config` mirrors ai-toolkit's real
`config/examples/train_lora_flux_24gb.yaml` — these tests check the
values actually landed on, not the Modal/GPU execution itself (that was
live-verified separately: a real LoRA trained end-to-end on a real
photo dataset — see the module's own docstring).
"""

from __future__ import annotations

from ml.identity.train_lora import (
    BASE_MODEL,
    LORA_ALPHA,
    LORA_RANK,
    TRAIN_STEPS,
    _training_config,
)


def test_training_config_uses_the_requested_steps_and_rank() -> None:
    config = _training_config(
        dataset_dir="/root/dataset",
        output_name="test_lora",
        trigger_token="zkqid",
        steps=1000,
        rank=16,
        alpha=32,
    )
    process = config["config"]["process"][0]  # type: ignore[index]
    assert process["train"]["steps"] == 1000
    assert process["network"]["linear"] == 16


def test_training_config_alpha_is_independent_of_rank() -> None:
    """Alpha defaults to 2x rank (LORA_ALPHA), not equal to it —
    ai-toolkit scales the LoRA's applied update by alpha/rank (verified
    against `toolkit/lora_special.py`), so this doubles the LoRA's
    effective strength without changing its capacity."""
    config = _training_config(
        dataset_dir="/root/dataset",
        output_name="test_lora",
        trigger_token="zkqid",
        steps=1000,
        rank=16,
        alpha=32,
    )
    process = config["config"]["process"][0]  # type: ignore[index]
    assert process["network"]["linear"] == 16
    assert process["network"]["linear_alpha"] == 32


def test_training_config_uses_cosine_schedule_with_total_iters() -> None:
    """"constant" (ai-toolkit's default) is replaced with "cosine" —
    verified against `toolkit/scheduler.py` that "cosine" requires
    `total_iters` in kwargs (no default, would error without it), hence
    `lr_scheduler_params` here rather than just the name."""
    config = _training_config(
        dataset_dir="/root/dataset",
        output_name="test_lora",
        trigger_token="zkqid",
        steps=1234,
        rank=16,
        alpha=32,
    )
    train = config["config"]["process"][0]["train"]  # type: ignore[index]
    assert train["lr_scheduler"] == "cosine"
    assert train["lr_scheduler_params"] == {"total_iters": 1234}


def test_training_config_uses_flux_dev_and_quantizes() -> None:
    config = _training_config(
        dataset_dir="/root/dataset",
        output_name="test_lora",
        trigger_token="zkqid",
        steps=1000,
        rank=16,
        alpha=32,
    )
    process = config["config"]["process"][0]  # type: ignore[index]
    assert process["model"]["name_or_path"] == BASE_MODEL
    assert process["model"]["is_flux"] is True
    assert process["model"]["quantize"] is True


def test_training_config_points_at_the_given_dataset_dir() -> None:
    config = _training_config(
        dataset_dir="/root/my-dataset",
        output_name="test_lora",
        trigger_token="zkqid",
        steps=1000,
        rank=16,
        alpha=32,
    )
    process = config["config"]["process"][0]  # type: ignore[index]
    assert process["datasets"][0]["folder_path"] == "/root/my-dataset"


def test_training_config_sample_prompts_include_the_trigger_token() -> None:
    # A single sample prompt hits a real ai-toolkit tokenizer edge case —
    # several prompts, all carrying the trigger token, is what works.
    config = _training_config(
        dataset_dir="/root/dataset",
        output_name="test_lora",
        trigger_token="zkqid",
        steps=1000,
        rank=16,
        alpha=32,
    )
    process = config["config"]["process"][0]  # type: ignore[index]
    prompts = process["sample"]["prompts"]
    assert len(prompts) >= 2
    assert all("zkqid" in p for p in prompts)


def test_default_constants_match_claude_md_task_5_6() -> None:
    # TRAIN_STEPS is 2000, not 1000: an identity test render scored below
    # target at 1000 steps, so it was raised for stronger identity
    # signal. LORA_ALPHA is 2x LORA_RANK, not equal to it.
    assert TRAIN_STEPS == 2000
    assert LORA_RANK == 16
    assert LORA_ALPHA == 32
    assert BASE_MODEL == "black-forest-labs/FLUX.1-dev"
