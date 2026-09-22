# 0010. LoRA training GPU: L40S, not A10G

Status: Accepted

## Context

Task 5.6 specifies Modal + A10G for Flux LoRA training. Live-verified
2026-09-09: Modal's actual GPU parameter is `"A10"` (not `"A10G"` —
that's the underlying AWS/Nvidia chip name), reporting 22.06 GiB usable
VRAM. Training with ai-toolkit's own `train_lora_flux_24gb.yaml`
reference config — the config this project's own `_training_config`
mirrors — hit `torch.OutOfMemoryError` converting the Flux transformer
to its quantized dtype, mid-model-load, before any training step ran.

This isn't a config bug: community reports independently describe 24GB
Flux LoRA training as unreliable at this exact edge, and ai-toolkit
offers `low_vram`/CPU-offloading options specifically because of it —
options explicitly flagged elsewhere as trading reliability for cost
("you do not want an OOM 2 hours into training").

## Decision

Train on Modal's `"L40S"` (48GB) instead. Real numbers from this
project's own 6 live attempts (2026-09-09): A10 failed OOM in under a
minute every time; L40S succeeded in ~25 minutes on the first attempt
once the earlier dependency bugs were fixed. Per-hour cost is higher
(~$1.95 vs ~$1.10), but a successful run's *total* cost lands close to
CLAUDE.md's original ~$0.50 estimate regardless of which GPU, since
A10's attempts never got far enough to spend meaningful GPU-time before
failing.

L40S requires a payment method on the Modal account (confirmed via a
real `Please add a payment method to use L40S GPU functions` error —
contrary to some third-party pricing writeups claiming full GPU access
without one); the user added one. Usage still draws from the $30/month
free credit first.

## Consequences

`train_lora.py`'s `GPU_TYPE` stays `"L40S"` going forward — re-run this
comparison if ai-toolkit's memory footprint changes (a version bump, a
different quantization path) or if training resolution/config changes
meaningfully, since the OOM margin on A10 was close enough that a small
config change could plausibly tip it either way. Not switched back
without measuring again.
