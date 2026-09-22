"""Flux LoRA training on Modal — fine-tunes Flux Dev on one person's photos
using `ai-toolkit` (ostris/ai-toolkit), producing a small LoRA weights file
(`IdentityRef.flux_lora_url` / `IdentityModel.flux_lora_url`) rather than a
full copy of the 12B-parameter base model.

Config field names/nesting mirror ai-toolkit's own example
(`config/examples/train_lora_flux_24gb.yaml`). Dependencies come from
cloning ai-toolkit and running its own `requirements.txt` rather than a
hand-picked package list, since that repo pins `diffusers` from a specific
git commit and versions of `transformers`/`peft` that are easy to get
wrong by guessing; `torchaudio` needs a separate explicit install since
`toolkit/config_modules.py` imports it unconditionally but ai-toolkit's own
requirements file doesn't list it. The image also needs `libgl1`/
`libglib2.0-0` since ai-toolkit's requirements pull in `opencv-python`
(not the headless variant), which needs OpenGL/GLib shared libraries a
minimal Debian container doesn't ship.

GPU is `"L40S"` (48GB), not `"A10"` (24GB, Modal's own naming for the
underlying A10G chip) — a 24GB card runs right at the edge of OOM
converting the Flux transformer to its quantized dtype, and 24GB Flux LoRA
training is widely reported as unreliable for this reason. Costs somewhat
more per hour but training time is similar or faster.

`TrainingStatus` reflects what's actually observable from Modal's polling
API (`FunctionCall.get(timeout=0)`, which only distinguishes "still
running" from "done" from "errored"): `"running" | "ready" | "failed"`,
not a `queued -> training -> ready` progression Modal has no way to report.

The sample config needs multiple varied prompts (not a single-prompt
list) — ai-toolkit's tokenizer call hits an internal edge case
(`ValueError: text input must be of type str...`) with just one prompt.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Literal

import modal

APP_NAME = "comiccanvas-identity"
FUNCTION_NAME = "train_lora_remote"

app = modal.App(APP_NAME)

TRAIN_STEPS = 2000  # a weak/inconsistent identity signal at 1000 steps left the LoRA as a
# weak nudge easily overridden by Flux's own strong demographic/scene priors; doubling steps
# doubles the identity signal's gradient exposure. Rank stays at 16 (rank buys capacity for
# more identity detail, not a stronger pull).
LORA_RANK = 16
LORA_ALPHA = 32  # 2x rank, not equal to it — ai-toolkit scales the LoRA's applied update by
# alpha/rank, so this doubles the effective strength of every training update without
# changing rank's capacity (common alpha/rank ratios in the wild run 1x-2x).
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
GPU_TYPE = "L40S"  # see module docstring for why not a 24GB card
TIMEOUT_SECONDS = 5400  # generous headroom over the observed ~44 min for 2000 steps
# plus sampling and cold-start/package-install overhead.

TrainingStatus = Literal["running", "ready", "failed"]

_image = (
    modal.Image.debian_slim(python_version="3.12")
    # libgl1/libglib2.0-0: ai-toolkit's requirements.txt pulls in
    # opencv-python (not the headless variant), which needs OpenGL/GLib
    # shared libraries a minimal Debian container doesn't ship.
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit.git /root/ai-toolkit",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
        # ai-toolkit's own requirements — not a hand-copied list: torch/
        # torchvision aren't pinned here either, pip resolves them
        # transitively from packages that do need a specific torch
        # (torchao, torchcodec, ...).
        "cd /root/ai-toolkit && pip install -r requirements.txt",
        # ai-toolkit's own requirements.txt is itself incomplete — torchaudio
        # is imported unconditionally by toolkit/config_modules.py but never
        # listed as a dependency. Unpinned deliberately: let pip match
        # whatever torch version requirements.txt already resolved, above.
        "pip install torchaudio",
    )
)

# Persists the downloaded Flux base model across runs — it's ~24GB;
# re-downloading it every training job would be slow and wasteful.
_model_cache = modal.Volume.from_name("comiccanvas-flux-cache", create_if_missing=True)


def _training_config(
    *, dataset_dir: str, output_name: str, trigger_token: str, steps: int, rank: int, alpha: int
) -> dict[str, object]:
    """Mirrors ai-toolkit's `train_lora_flux_24gb.yaml` example — same
    keys/nesting, values swapped in for this job's own hyperparameters
    and dataset/output name."""
    return {
        "job": "extension",
        "config": {
            "name": output_name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": "/root/output",
                    "device": "cuda:0",
                    "network": {"type": "lora", "linear": rank, "linear_alpha": alpha},
                    "save": {
                        "dtype": "float16",
                        "save_every": steps,  # only the final checkpoint — one job, one result
                        "max_step_saves_to_keep": 1,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_dir,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [512, 768, 1024],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": 1e-4,
                        # "cosine" dispatches to torch's CosineAnnealingLR, which requires
                        # `T_max` in kwargs (passing "cosine" alone errors) hence `total_iters`
                        # below, which the scheduler factory converts to `T_max`. Decays
                        # smoothly to ~0 by the last step instead of taking full-size steps
                        # right up to the end. No warmup added (ai-toolkit's warmup and cosine
                        # are separate, non-combinable scheduler names, and warmup mainly
                        # guards against instability at much higher LRs / larger models).
                        "lr_scheduler": "cosine",
                        "lr_scheduler_params": {"total_iters": steps},
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": BASE_MODEL,
                        "is_flux": True,
                        "quantize": True,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": steps,  # once, at the end — not the reference's
                        # every-250 cadence (this job is short enough not to need progress
                        # previews), but a baseline sample still renders unconditionally
                        # before training starts either way
                        "width": 1024,
                        "height": 1024,
                        # Several varied prompts, matching the reference example's shape —
                        # a single sample prompt hits an edge case in ai-toolkit's tokenizer
                        # call.
                        "prompts": [
                            f"{trigger_token} a photo of a person, standing outdoors",
                            f"{trigger_token} a photo of a person, sitting at a desk",
                            f"{trigger_token} a photo of a person, smiling indoors",
                        ],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {"name": f"[{output_name}]", "version": "1.0"},
    }


@app.function(
    image=_image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
    volumes={"/root/.cache/huggingface": _model_cache},
)
def train_lora_remote(
    dataset_tar: bytes,
    *,
    trigger_token: str,
    output_name: str,
    steps: int = TRAIN_STEPS,
    rank: int = LORA_RANK,
    alpha: int = LORA_ALPHA,
) -> bytes:
    """Runs inside the Modal container: unpacks the dataset, writes
    ai-toolkit's config, runs training, returns the resulting LoRA
    `.safetensors` file's bytes."""
    import subprocess

    import yaml

    dataset_dir = Path("/root/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(dataset_tar), mode="r:gz") as tar:
        tar.extractall(dataset_dir)

    config = _training_config(
        dataset_dir=str(dataset_dir),
        output_name=output_name,
        trigger_token=trigger_token,
        steps=steps,
        rank=rank,
        alpha=alpha,
    )
    config_path = Path("/root/config.yaml")
    config_path.write_text(yaml.safe_dump(config))

    subprocess.run(
        ["python", "/root/ai-toolkit/run.py", str(config_path)],
        check=True,
        cwd="/root/ai-toolkit",
    )

    output_dir = Path("/root/output") / output_name
    safetensors_files = sorted(output_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"no .safetensors output found under {output_dir}")
    return safetensors_files[-1].read_bytes()  # last/highest step if more than one saved


# --- Character LoRA ------------------------------------------------------
#
# Reuses this module's already-built `app`/`_image`/`_model_cache` (same
# base image, same persisted Flux download) rather than standing up a
# second Modal app — the only real differences are the dataset shape
# (character-sheet images + face masks + a regularization set) and the
# config (`configs/character_v1.yaml`), not the training environment itself.

CHARACTER_CONFIG_PATH = Path(__file__).parent / "configs" / "character_v1.yaml"


@app.function(
    image=_image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
    volumes={"/root/.cache/huggingface": _model_cache},
)
def train_character_lora_remote(dataset_tar: bytes, reg_tar: bytes, *, config_yaml: str) -> bytes:
    """Runs inside the Modal container: unpacks the character-sheet
    dataset (images + `masks/`) to `/root/dataset` and the regularization
    set to `/root/reg_dataset` (both fixed paths `configs/character_v1.yaml`
    already points at), writes the given config, runs training, and
    returns a tarball of the entire output directory — every checkpoint
    and its sample grid, not just the last one, since checkpoint selection
    needs all of them to pick the best."""
    import subprocess

    dataset_dir = Path("/root/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(dataset_tar), mode="r:gz") as tar:
        tar.extractall(dataset_dir)

    reg_dir = Path("/root/reg_dataset")
    reg_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(reg_tar), mode="r:gz") as tar:
        tar.extractall(reg_dir)

    config_path = Path("/root/config.yaml")
    config_path.write_text(config_yaml)

    subprocess.run(
        ["python", "/root/ai-toolkit/run.py", str(config_path)],
        check=True,
        cwd="/root/ai-toolkit",
    )

    output_root = Path("/root/output")
    if not any(output_root.iterdir()):
        raise FileNotFoundError(f"no output produced under {output_root}")
    return _build_dataset_tarball(output_root)


def _build_dataset_tarball(dataset_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(dataset_dir, arcname=".")
    return buf.getvalue()


def submit_character_training_job(
    dataset_dir: Path, reg_dir: Path, *, output_name: str, config_path: Path = CHARACTER_CONFIG_PATH
) -> str:
    """Same submit-now-poll-later shape as `submit_training_job` — loads
    the committed `character_v1.yaml`, overrides only `name` (the one
    field that legitimately varies per run; see the config file's own
    header comment for why everything else is a fixed in-container path),
    and spawns the run. Requires `modal deploy ml/identity/train_lora.py`
    first, same precondition as the baseline job."""
    import yaml

    config = yaml.safe_load(config_path.read_text())
    config["config"]["name"] = output_name

    dataset_tar = _build_dataset_tarball(dataset_dir)
    reg_tar = _build_dataset_tarball(reg_dir)
    fn = modal.Function.from_name(APP_NAME, "train_character_lora_remote")
    call = fn.spawn(dataset_tar, reg_tar, config_yaml=yaml.safe_dump(config))
    return call.object_id


def submit_training_job(dataset_dir: Path, *, trigger_token: str, output_name: str) -> str:
    """Kicks off training, returns a job id to poll later —
    `check_training_status`/`get_training_result` reconstruct the handle
    from this id, so it can be polled from a separate process/request.

    Looks the function up by name (`modal.Function.from_name`) rather
    than spawning through a local `app.run()` context — this app must
    already be deployed (`modal deploy ml/identity/train_lora.py`) first.
    That's the pattern Modal's own docs show for "submit now, poll later,
    maybe from a different process," which is exactly this job's shape.
    """
    dataset_tar = _build_dataset_tarball(dataset_dir)
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    call = fn.spawn(dataset_tar, trigger_token=trigger_token, output_name=output_name)
    return call.object_id


def check_training_status(job_id: str) -> TrainingStatus:
    call = modal.FunctionCall.from_id(job_id)
    try:
        call.get(timeout=0)
    except TimeoutError:
        return "running"
    except Exception:
        return "failed"
    return "ready"


def get_training_result(job_id: str) -> bytes:
    """Only valid once `check_training_status` reports `"ready"`."""
    call = modal.FunctionCall.from_id(job_id)
    result: bytes = call.get(timeout=0)
    return result
