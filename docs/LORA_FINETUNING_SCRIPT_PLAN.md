# Building a LoRA Fine-Tuning Script for DiffRhythm

A detailed, line-by-line rationale for every design decision in a DiffRhythm LoRA fine-tuning script. This document assumes you've read `docs/LORA_FINETUNING_GUIDE.md` for general PEFT/LoRA concepts and `docs/TRAINING_SCRIPTS.md` for the existing training architecture.

---

## Table of Contents

1. [What We're Building and Why](#1-what-were-building-and-why)
2. [How the Existing Training Pipeline Works](#2-how-the-existing-training-pipeline-works)
3. [The Pretrained Checkpoint Format](#3-the-pretrained-checkpoint-format)
4. [New Files We Need to Create](#4-new-files-we-need-to-create)
5. [The LoRA Training Entry Point: train/train_lora.py](#5-the-lora-training-entry-point-traintrain_lorapy)
   - [Why a Separate Entry Point](#why-a-separate-entry-point)
   - [Loading Pretrained Weights: The Critical Order](#loading-pretrained-weights-the-critical-order)
   - [Why We Target DiT, Not CFM](#why-we-target-dit-not-cfm)
   - [Choosing Target Modules for This Architecture](#choosing-target-modules-for-this-architecture)
   - [The PEFT Wrapping Step](#the-peft-wrapping-step)
   - [Wrapping in CFM After LoRA](#wrapping-in-cfm-after-lora)
   - [Full Entry Point Code](#full-entry-point-code)
6. [The LoRA Trainer: model/trainer_lora.py](#6-the-lora-trainer-modeltrainer_lorapy)
   - [Why We Can't Reuse the Existing Trainer](#why-we-cant-reuse-the-existing-trainer)
   - [Constructor Changes](#constructor-changes)
   - [The Optimizer: Why Filtering Parameters Matters](#the-optimizer-why-filtering-parameters-matters)
   - [Why We Don't Need EMA](#why-we-dont-need-ema)
   - [Learning Rate and Scheduler Choices](#learning-rate-and-scheduler-choices)
   - [The Training Loop: What Stays the Same and Why](#the-training-loop-what-stays-the-same-and-why)
   - [Checkpoint Saving: Three Formats](#checkpoint-saving-three-formats)
   - [Checkpoint Loading: Resuming LoRA Training](#checkpoint-loading-resuming-lora-training)
   - [Full Trainer Code](#full-trainer-code)
7. [The LoRA Config File: config/lora_default.ini](#7-the-lora-config-file-configlora_defaultini)
8. [The Launch Script: scripts/train_lora.sh](#8-the-launch-script-scriptstrain_lorash)
9. [LoRA Inference: How to Use the Trained Adapter](#9-lora-inference-how-to-use-the-trained-adapter)
10. [Design Decisions Deep Dive](#10-design-decisions-deep-dive)
    - [Why the Dataset Stays Unchanged](#why-the-dataset-stays-unchanged)
    - [Why CFM.forward() Stays Unchanged](#why-cfmforward-stays-unchanged)
    - [Why Gradient Clipping Works Without Changes](#why-gradient-clipping-works-without-changes)
    - [Why Gradient Checkpointing is Trickier with PEFT](#why-gradient-checkpointing-is-trickier-with-peft)
    - [VRAM Budget Comparison](#vram-budget-comparison)
11. [Summary of All Changes](#11-summary-of-all-changes)

---

## 1. What We're Building and Why

We need to create a training pipeline that:

1. Loads the pretrained DiffRhythm model (1B parameters)
2. Freezes all existing weights
3. Injects small trainable LoRA adapters into the transformer layers
4. Trains only those adapters (~8M parameters) on a new dataset
5. Saves just the adapter weights (not the full 1B model)
6. Supports loading those adapters at inference time

We need **new files** rather than modifying existing ones because:

- The existing training scripts assume full-parameter training. Modifying them to conditionally support LoRA would add complexity and risk breaking the original training path.
- The checkpoint format is fundamentally different (LoRA adapters vs. full model weights).
- Several components (EMA, optimizer, scheduler) need different configurations.
- A separate script makes it clear to anyone reading the code: "this is for fine-tuning, that is for pre-training."

---

## 2. How the Existing Training Pipeline Works

Before building the LoRA version, let's trace exactly what happens in the current training pipeline, because our LoRA script will mirror this structure with specific modifications at each stage.

### Current Flow (from `train/train.py` → `model/trainer.py`)

```
1. train/train.py: Load config via prefigure
2. train/train.py: Build DiT from JSON config (random weights)
3. train/train.py: Wrap DiT in CFM
4. train/train.py: Create Trainer with CFM model
5. model/trainer.py.__init__:
   a. Create Accelerator with DDP + WandB
   b. Create EMA model (full copy of weights, main process only)
   c. Create AdamW optimizer over ALL model.parameters()
   d. Build DataLoader from DiffusionDataset
   e. Build SequentialLR scheduler (warmup → decay)
   f. accelerator.prepare(model, optimizer, scheduler, dataloader)
6. model/trainer.py.train:
   a. Load checkpoint if exists (full model + optimizer + EMA + scheduler + step)
   b. For each epoch/batch:
      - CFM.forward() computes flow matching loss
      - backward, clip gradients, optimizer step, scheduler step
      - EMA update
      - Log to WandB
      - Save checkpoints periodically
```

Our LoRA version will follow the same high-level structure but with modifications at steps 2–3 (load pretrained + apply LoRA), 5a-5e (no EMA, filtered optimizer, different scheduler), and 6a (different checkpoint format).

---

## 3. The Pretrained Checkpoint Format

Understanding the checkpoint format is critical because we need to load it correctly before applying LoRA. Looking at `infer/infer_utils.py`:

```python
def load_checkpoint(model, ckpt_path, device, use_ema=True):
    model = model.half()
    checkpoint = torch.load(ckpt_path, weights_only=True)

    if use_ema:
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.to(device)
```

There are two key observations here:

1. **The checkpoint contains keys under `model_state_dict`** — these are the CFM-level state dict keys (e.g., `transformer.transformer_blocks.0.self_attn.q_proj.weight`). The `transformer.` prefix exists because DiT is stored as `self.transformer` inside CFM.

2. **The EMA weights are stored separately** under `ema_model_state_dict`. For fine-tuning, we want the EMA weights (they're the smoothed, higher-quality version of the trained weights). But note they have an `ema_model.` prefix that gets stripped.

3. **`strict=False` is used** because the checkpoint may have extra or missing keys depending on the version. This is also what allows us to use the same pattern for LoRA loading later.

Additionally, from `infer/infer_utils.py` we can see the **HuggingFace Hub download pattern**:

```python
dit_ckpt_path = hf_hub_download(
    repo_id="ASLP-lab/DiffRhythm-1_2", filename="cfm_model.pt", cache_dir="./pretrained"
)
```

Our LoRA script will use this same function to download the base model.

### What This Means for Our LoRA Script

When we load pretrained weights into the model, we need to:
1. Build the full `CFM(DiT(...))` model first (to match the state dict key structure)
2. Load the checkpoint with `strict=False`
3. **Then** extract the `DiT` from inside the `CFM`, apply LoRA to it, and put it back

OR (the approach we'll take):
1. Build the `DiT` alone
2. Load just the DiT weights (stripping the `transformer.` prefix from the CFM-level state dict)
3. Apply LoRA to the DiT
4. Then wrap the LoRA-DiT in `CFM`

We'll use approach 2 because it's cleaner — we can apply LoRA directly to the DiT without having to unwrap it from CFM first.

---

## 4. New Files We Need to Create

| File | Purpose | Why New (vs. Modify) |
|------|---------|---------------------|
| `train/train_lora.py` | LoRA training entry point | Different model construction flow (load pretrained + apply LoRA) |
| `model/trainer_lora.py` | LoRA-specific Trainer class | Different optimizer, no EMA, different checkpoint format |
| `config/lora_default.ini` | LoRA training hyperparameters | Different LR, epochs, warmup, plus new LoRA-specific params |
| `scripts/train_lora.sh` | Launch script for LoRA training | Different entry point and default args |

Files that **stay unchanged** (and why):

| File | Why Unchanged |
|------|--------------|
| `model/cfm.py` | The CFM wrapper doesn't care what's inside the transformer. LoRA-wrapped DiT has the same forward signature. The flow matching loss computation is identical. |
| `model/dit.py` | LoRA wraps the model externally via PEFT. The DiT code doesn't need to know about LoRA. |
| `model/modules.py` | Building blocks are unchanged — LoRA targets the Linear layers inside them, not the module definitions. |
| `dataset/dataset.py` | The data format and loading are identical for fine-tuning. The same `.scp` file format works. |
| `model/utils.py` | Helper functions are generic and work the same way. |
| `config/diffrhythm-1b.json` | The model architecture is identical — LoRA adds to it, doesn't change it. |

---

## 5. The LoRA Training Entry Point: `train/train_lora.py`

### Why a Separate Entry Point

The existing `train/train.py` does this:

```python
model = CFM(
    transformer=DiT(**model_config["model"], max_frames=args.max_frames),
    ...
)
trainer = Trainer(model, args, ...)
trainer.train(...)
```

Our LoRA version needs to insert three extra steps between model creation and trainer creation:
1. Load pretrained weights into the DiT
2. Apply LoRA configuration to the DiT
3. Use a different Trainer class

These changes are fundamental enough that a conditional `if args.use_lora:` inside the existing script would make the code harder to follow.

### Loading Pretrained Weights: The Critical Order

The order of operations is **the single most important thing to get right**:

```
1. Build DiT with random weights
2. Load pretrained weights into DiT       ← MUST happen before LoRA
3. Apply LoRA to DiT (freezes base weights, adds adapters)
4. Wrap LoRA-DiT in CFM
5. Pass to Trainer
```

**Why this order matters:**

- **Step 2 before Step 3**: `get_peft_model()` freezes all base parameters and adds LoRA adapters. If you apply LoRA first, you'd be adding adapters to random weights. The pretrained knowledge would never be loaded. Your "fine-tuning" would actually be training from scratch with a crippled architecture (frozen random weights + tiny adapters).

- **Step 3 before Step 4**: If you wrap in CFM first, then apply LoRA to the whole CFM, PEFT would search through the entire CFM module tree. It would still find the same Linear layers (PEFT searches recursively), but the module paths in the LoRA state dict would include `transformer.` prefix (e.g., `base_model.model.transformer.transformer_blocks.0...`). This makes the adapter checkpoint CFM-specific. Applying LoRA to just the DiT produces cleaner adapter paths and decouples the adapter from the CFM wrapper.

### Why We Target DiT, Not CFM

Let's look at what's inside CFM (`model/cfm.py`):

```python
class CFM(nn.Module):
    def __init__(self, transformer, ...):
        self.transformer = transformer    # ← the DiT, all the weights live here
        # Everything else is just scalars, config, or non-parametric
        self.sigma = sigma
        self.frac_lengths_mask = frac_lengths_mask
        self.audio_drop_prob = audio_drop_prob
        # ... etc
```

CFM has **zero trainable parameters of its own**. It's a pure wrapper that implements the flow matching math (noise sampling, interpolation, loss computation) and delegates all neural network computation to `self.transformer`. There's nothing to adapt in CFM.

If we applied `get_peft_model(cfm, config)`, PEFT would recursively find the same Linear layers inside `cfm.transformer.transformer_blocks[*]`, but:
- The adapter key paths would be longer: `base_model.model.transformer.transformer_blocks.0.self_attn.q_proj`
- We'd need to unwrap both the PEFT wrapper AND the CFM wrapper when saving/loading
- The adapter wouldn't be reusable with a different wrapper (e.g., if someone wanted to use the DiT in a different framework)

Targeting DiT directly produces adapter keys like:
`base_model.model.transformer_blocks.0.self_attn.q_proj`

### Choosing Target Modules for This Architecture

The DiT model has these Linear layers (from `model/dit.py`):

**Inside each of the 16 `LlamaDecoderLayer` blocks:**
```
transformer_blocks.{i}.self_attn.q_proj   (2048 → 2048)  — Query projection
transformer_blocks.{i}.self_attn.k_proj   (2048 → 2048)  — Key projection
transformer_blocks.{i}.self_attn.v_proj   (2048 → 2048)  — Value projection
transformer_blocks.{i}.self_attn.o_proj   (2048 → 2048)  — Output projection
transformer_blocks.{i}.mlp.gate_proj      (2048 → 8192)  — Gated MLP input
transformer_blocks.{i}.mlp.up_proj        (2048 → 8192)  — MLP up-project
transformer_blocks.{i}.mlp.down_proj      (8192 → 2048)  — MLP down-project
```

**Outside the transformer blocks (NOT targeted by LoRA):**
```
text_embed.text_embed                   — Embedding lookup (not a Linear)
text_embed.text_blocks.{i}.pwconv1      — ConvNeXtV2 pointwise conv (Linear)
text_embed.text_blocks.{i}.pwconv2      — ConvNeXtV2 pointwise conv (Linear)
input_embed.proj                        — Input concatenation projection (Linear)
time_embed.time_mlp.0, .2              — Time embedding MLPs (Linear)
start_time_embed.time_mlp.0, .2        — Start time embedding MLPs (Linear)
text_fusion_linears.{i}.0              — Text fusion projection (Linear)
norm_out.linear                        — Final adaptive norm (Linear)
proj_out                               — Final output projection (Linear)
```

**Why we only target the transformer block layers:**

The `LlamaDecoderLayer` blocks are where the model's core learned representations live. They process the audio latents with self-attention and feed-forward transformations — this is where the model "understands" music structure, timbre, rhythm, etc.

The other Linear layers (time embedding, input projection, text fusion, output projection) are:
- **Small**: They have far fewer parameters than the transformer blocks
- **Interface layers**: They handle input/output format conversion, not learned musical knowledge
- **Sensitive**: Modifying the output projection (`proj_out`) or time embedding could destabilize the flow matching dynamics

That said, there's a case for adding `proj_out` to `modules_to_save` (fully trainable, not LoRA) if fine-tuning results seem flat. This would let the output head fully adapt while keeping the transformer blocks parameter-efficient.

### The PEFT Wrapping Step

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

dit = get_peft_model(dit, lora_config)
```

What `get_peft_model` does to the DiT internally:

1. **Scans** all named modules in the DiT
2. For each `nn.Linear` whose name ends with one of the `target_modules` strings, it **replaces** that Linear with a `LoraLinear` module
3. The `LoraLinear` contains:
   - The original frozen weight matrix `W` (e.g., 2048×2048)
   - A trainable `lora_A` matrix (e.g., 2048×16 for r=16)
   - A trainable `lora_B` matrix (e.g., 16×2048 for r=16)
   - Its forward computes: `output = W @ x + (lora_B @ lora_A @ x) * (alpha / r)`
4. **Freezes** all original parameters (`requires_grad = False`)
5. Returns a `PeftModel` wrapper that has the same forward signature as the original DiT

This means `dit(x=..., cond=..., text=..., time=..., ...)` still works exactly the same — the PeftModel delegates to the underlying DiT forward method.

### Wrapping in CFM After LoRA

After `get_peft_model`, the DiT is now a `PeftModel`. We pass it to CFM just like before:

```python
model = CFM(transformer=dit, ...)
```

CFM stores it as `self.transformer` and calls `self.transformer(x=..., cond=..., ...)` in its forward and sample methods. The PeftModel's `__call__` delegates to the original DiT's `forward()`, which works identically because LoRA doesn't change the forward signature.

**One important detail**: CFM accesses `transformer.dim` in its constructor:

```python
class CFM(nn.Module):
    def __init__(self, transformer, ...):
        self.transformer = transformer
        dim = transformer.dim          # ← This needs to work
```

After PEFT wrapping, `dit.dim` may not be directly accessible because the PeftModel wrapper doesn't automatically expose arbitrary attributes. We need to handle this. There are two approaches:

1. **Access via the base model**: `dit.base_model.model.dim` (fragile, depends on PEFT internals)
2. **Set it on the PeftModel**: `dit.dim = dit.base_model.model.dim` after wrapping
3. **Pass `dim` separately**: Refactor CFM to take `dim` as a parameter

Approach 2 is simplest. In practice, PEFT's `PeftModel` does forward `__getattr__` calls to the base model for attributes it doesn't have, so `dit.dim` should work. But it's worth being aware of this in case of issues.

### Full Entry Point Code

```python
# train/train_lora.py

from importlib.resources import files
import json
import os
import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from prefigure.prefigure import get_all_args

from model import CFM, DiT
from model.trainer_lora import LoRATrainer

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"


def main():
    args = get_all_args("config/lora_default.ini")

    with open(args.model_config) as f:
        model_config = json.load(f)

    # ---- Step 1: Build DiT with random weights ----
    dit = DiT(**model_config["model"], max_frames=args.max_frames)

    # ---- Step 2: Load pretrained weights ----
    # Download from HuggingFace Hub (same as infer/infer_utils.py)
    if args.pretrained_ckpt_path:
        ckpt_path = args.pretrained_ckpt_path
    else:
        repo_id = "ASLP-lab/DiffRhythm-1_2"
        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename="cfm_model.pt", cache_dir="./pretrained"
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # The checkpoint has CFM-level keys (transformer.xxx).
    # We need to strip the "transformer." prefix to load into the bare DiT.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "ema_model_state_dict" in checkpoint:
        # Prefer EMA weights — they're higher quality
        state_dict = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
    else:
        state_dict = checkpoint

    # Strip "transformer." prefix to match DiT's own state dict keys
    dit_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("transformer."):
            dit_state_dict[k.replace("transformer.", "", 1)] = v

    dit.load_state_dict(dit_state_dict, strict=False)
    print(f"Loaded pretrained DiT weights from {ckpt_path}")

    # ---- Step 3: Apply LoRA ----
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    dit = get_peft_model(dit, lora_config)
    dit.print_trainable_parameters()

    # ---- Step 4: Wrap in CFM ----
    model = CFM(
        transformer=dit,
        num_channels=model_config["model"]["mel_dim"],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=args.cond_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ---- Step 5: Create trainer and start training ----
    trainer = LoRATrainer(
        model,
        args,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=f"ckpts/{args.exp_name}",
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="diffrhythm-lora",
        wandb_run_name=args.exp_name,
        last_per_steps=args.last_per_steps,
        batch_size=args.batch_size,
    )

    trainer.train(resumable_with_seed=args.resumable_with_seed)


if __name__ == "__main__":
    main()
```

**Key differences from `train/train.py`:**

| Line | Original | LoRA Version | Why |
|------|----------|-------------|-----|
| Config file | `config/default.ini` | `config/lora_default.ini` | Different hyperparameters |
| After DiT creation | (nothing) | Load pretrained weights | Need base knowledge |
| After loading | (nothing) | `get_peft_model(dit, lora_config)` | Inject LoRA adapters |
| Trainer class | `Trainer` | `LoRATrainer` | Different optimizer/checkpoint/EMA behavior |
| Trainer args | Includes EMA, bnb_optimizer | Simpler, no EMA | Not needed for LoRA |

---

## 6. The LoRA Trainer: `model/trainer_lora.py`

### Why We Can't Reuse the Existing Trainer

The existing `Trainer` in `model/trainer.py` has these behaviors that are wrong for LoRA:

1. **Optimizer over all parameters** (line 143):
   ```python
   self.optimizer = AdamW(model.parameters(), lr=learning_rate)
   ```
   This allocates Adam states (momentum + variance = 2 floats per parameter) for **all** 1B parameters, even though 99.2% are frozen. That's ~8GB of wasted VRAM.

2. **EMA model** (line 112):
   ```python
   self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
   ```
   This creates a full copy of all model parameters for EMA averaging. That's another ~2GB for the EMA shadow weights. With LoRA's tiny adapter size, the improvement from EMA wouldn't be noticeable and the memory cost is disproportionate.

3. **Checkpoint saving** (line 190-203): Saves the full `model.state_dict()` which would include both frozen base weights and LoRA weights — redundant and wasteful (multi-GB files vs. ~50MB for just the LoRA adapter).

4. **Checkpoint loading** (line 205-255): Expects full model state dicts and EMA state dicts that don't exist in our LoRA checkpoint format.

5. **Scheduler** (line 153-163): The warmup+linear-decay schedule is designed for 100+ epoch runs from scratch. LoRA fine-tuning is typically much shorter and benefits from a different schedule.

### Constructor Changes

Here's what changes and why:

```python
class LoRATrainer:
    def __init__(
        self,
        model: CFM,
        args,
        epochs,
        learning_rate,
        num_warmup_updates=1000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        wandb_project="diffrhythm-lora",
        wandb_run_name="lora_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
    ):
```

Compared to the original `Trainer.__init__`, we've **removed**:
- `bnb_optimizer` — 8-bit AdamW isn't necessary when we only have ~8M trainable params
- `ema_kwargs` — no EMA
- `noise_scheduler` — unused in the original too
- `duration_predictor` — unused in the original too
- `use_style_prompt` — not trainer-level config
- `grad_ckpt` — handled separately (see gradient checkpointing section)

### The Optimizer: Why Filtering Parameters Matters

**Original code (model/trainer.py, line 143):**
```python
self.optimizer = AdamW(model.parameters(), lr=learning_rate)
```

**LoRA version:**
```python
self.optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
)
```

**What `filter(lambda p: p.requires_grad, ...)` does:**

PyTorch's `model.parameters()` yields every parameter in the model. After `get_peft_model`, only the LoRA adapter matrices have `requires_grad=True`. The `filter` keeps only those.

**Why this matters — the memory arithmetic:**

AdamW maintains two state tensors per parameter: first moment (momentum) and second moment (variance), each the same size as the parameter itself.

| Scenario | Params in Optimizer | Optimizer State Size |
|----------|--------------------|--------------------|
| **Without filter** (all params) | ~1,000,000,000 | ~8 GB (2 × 4 bytes × 1B) |
| **With filter** (LoRA only, r=16) | ~8,000,000 | ~64 MB (2 × 4 bytes × 8M) |

That's a **125x reduction** in optimizer memory. On a 24GB GPU (e.g., RTX 3090/4090), this is the difference between fitting the model and running out of VRAM.

PyTorch won't error if you pass frozen parameters to the optimizer. It will silently create Adam states for them, allocate the memory, and compute (zero) gradient updates every step. The training would work correctly but you'd waste 8GB of VRAM for no benefit.

### Why We Don't Need EMA

**What EMA does in the original trainer:**

```python
# Line 112: create EMA
self.ema_model = EMA(model, include_online_model=False)

# Line 317: update after each step
self.ema_model.update()
```

EMA maintains an exponentially smoothed copy of the model weights. It helps smooth out training noise and generally produces slightly better final weights. The EMA model (not the online model) is what gets used for inference in the published checkpoints.

**Why it's unnecessary for LoRA:**

1. **Few parameters**: With only ~8M trainable parameters, the training dynamics are much smoother. There's less noise to average out.
2. **Short training runs**: LoRA fine-tuning typically runs for thousands of steps, not millions. EMA's benefit increases with training length.
3. **Memory cost**: EMA creates a full shadow copy. For the original 1B model, this means duplicating 1B parameters (even though most are frozen). The `ema_pytorch` library doesn't know about LoRA — it would shadow everything.
4. **Complexity cost**: EMA with LoRA requires either: (a) EMA over just the LoRA params (custom code), or (b) EMA over the whole model (wasteful). Neither is worth the effort for typical LoRA fine-tuning.

### Learning Rate and Scheduler Choices

**Original training config:**
```ini
learning_rate = 7.5e-5
num_warmup_updates = 20
epochs = 110
```

**LoRA config:**
```ini
learning_rate = 2e-4
num_warmup_updates = 100
epochs = 10
```

**Why a higher learning rate:**

LoRA's adapter matrices are initialized as:
- `lora_A`: Random Gaussian (Kaiming uniform by default)
- `lora_B`: Zeros

This means at initialization, every LoRA adapter contributes **exactly zero** to the output (B × A × x = 0 × anything = 0). The model starts as the exact pretrained model.

With a low learning rate (like 7.5e-5), the adapters would barely move from zero in the first few hundred steps. The gradients flowing through the frozen base model are large (because the base model is large), but the LoRA update magnitude is `learning_rate × gradient`. With such a small LR, adaptation would be extremely slow.

A learning rate of 2e-4 to 5e-4 provides a stronger learning signal that pushes the adapters away from zero faster. The `alpha / r` scaling in LoRA (32/16 = 2.0 in our config) also affects the effective learning rate — a higher alpha means each adapter update has a larger effect on the output.

**Why a different scheduler:**

The original uses a warmup-then-linear-decay schedule designed for 110 epochs of from-scratch training. For LoRA, we want:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Warmup: 100 steps to let the optimizer states stabilize
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
# Cosine decay: smooth decay to near-zero
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
```

**Why cosine instead of linear decay:**

- Cosine decay starts slow (maintains high LR longer at the beginning), which helps LoRA adapters move away from their zero initialization
- It ends with a smooth approach to zero, which prevents the final training steps from making noisy updates
- It's the most common schedule for fine-tuning in the LoRA literature

### The Training Loop: What Stays the Same and Why

The inner training loop is almost identical to the original. Here's why each part stays the same:

**Batch unpacking (unchanged):**
```python
text_inputs = batch["lrc"]
mel_spec = batch["latent"].permute(0, 2, 1)
mel_lengths = batch["latent_lengths"]
style_prompt = batch["prompt"]
style_prompt_lens = batch["prompt_lengths"]
start_time = batch["start_time"]
```
The data format is identical. We're fine-tuning on the same type of data (lyrics + audio latents + style prompts).

**Forward pass (unchanged):**
```python
loss, cond, pred = self.model(
    mel_spec, text=text_inputs, lens=mel_lengths,
    noise_scheduler=self.noise_scheduler,
    style_prompt=style_prompt,
    style_prompt_lens=style_prompt_lens,
    start_time=start_time
)
```
The CFM.forward() method doesn't know or care about LoRA. It calls `self.transformer(x=φ, cond=cond, text=text, time=time, ...)` which goes through the PEFT wrapper, through the DiT, through the LoRA-adapted Linear layers, and back. The loss computation (MSE between predicted and target flow, masked to the random span) is identical.

**Backward pass (unchanged):**
```python
self.accelerator.backward(loss)
```
PyTorch autograd handles the LoRA gradient computation automatically. Gradients flow back through the frozen base weights (which don't get updated because their params have `requires_grad=False`) and through the LoRA adapter matrices (which do get updated).

**Gradient clipping (unchanged):**
```python
if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
```
`clip_grad_norm_` operates on all parameters but only clips those with actual gradients (i.e., the LoRA params). Frozen params have `None` gradients and are skipped automatically.

**What's removed:**
```python
# REMOVED: EMA update
# if self.is_main:
#     self.ema_model.update()
```

### Checkpoint Saving: Three Formats

We save in three formats to maximize flexibility:

```python
def save_checkpoint(self, step, last=False):
    self.accelerator.wait_for_everyone()
    if self.is_main:
        # 1. PEFT native format (for PeftModel.from_pretrained loading)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        dit_peft = unwrapped_model.transformer
        peft_save_dir = f"{self.checkpoint_path}/lora_adapter"
        if last:
            peft_save_dir = f"{self.checkpoint_path}/lora_adapter_last"
        else:
            peft_save_dir = f"{self.checkpoint_path}/lora_adapter_step_{step}"
        dit_peft.save_pretrained(peft_save_dir)

        # 2. Training state (for resuming training)
        training_state = dict(
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            step=step,
        )
        state_name = "training_state_last.pt" if last else f"training_state_{step}.pt"
        self.accelerator.save(training_state, f"{self.checkpoint_path}/{state_name}")

        # 3. Standalone LoRA weights (for manual loading)
        from peft import get_peft_model_state_dict
        lora_weights = get_peft_model_state_dict(dit_peft)
        weights_name = "lora_weights_last.pt" if last else f"lora_weights_{step}.pt"
        torch.save(lora_weights, f"{self.checkpoint_path}/{weights_name}")
```

**Why three formats:**

1. **PEFT native format** (`adapter_config.json` + `adapter_model.safetensors`): This is what `PeftModel.from_pretrained()` expects. It stores both the LoRA config and weights. This is the standard way to distribute and load LoRA adapters.

2. **Training state** (optimizer + scheduler + step): Separated from the model weights because:
   - Optimizer states are only needed for resuming training, not for inference
   - They're much larger than the LoRA weights themselves (~64MB for Adam states vs ~32MB for LoRA weights)
   - Keeping them separate means the PEFT adapter directory stays clean and loadable without the training baggage

3. **Standalone LoRA weights** (just a PyTorch state dict): A simple fallback for manual loading with `load_state_dict(weights, strict=False)`. Useful if you want to load the weights without the PEFT library or in a custom way.

**Contrast with the original trainer's checkpoint format:**

The original saves everything in one file:
```python
checkpoint = dict(
    model_state_dict=model.state_dict(),          # ~4GB (full model in fp32)
    optimizer_state_dict=optimizer.state_dict(),   # ~8GB (Adam states for all params)
    ema_model_state_dict=ema_model.state_dict(),   # ~4GB (EMA shadow)
    scheduler_state_dict=scheduler.state_dict(),   # tiny
    step=step,                                     # tiny
)
```
Total: ~16GB per checkpoint. Our LoRA checkpoint is ~100MB total across all three formats.

### Checkpoint Loading: Resuming LoRA Training

```python
def load_checkpoint(self):
    if not os.path.exists(self.checkpoint_path):
        return 0

    # Find latest PEFT adapter
    adapter_dirs = sorted(
        [d for d in os.listdir(self.checkpoint_path) if d.startswith("lora_adapter")],
        key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_path, x))
    )
    if not adapter_dirs:
        return 0

    latest_adapter = adapter_dirs[-1]
    adapter_path = os.path.join(self.checkpoint_path, latest_adapter)

    # Load LoRA weights into the PEFT model
    unwrapped_model = self.accelerator.unwrap_model(self.model)
    dit_peft = unwrapped_model.transformer
    from peft import set_peft_model_state_dict
    adapter_weights = torch.load(
        os.path.join(adapter_path, "adapter_model.safetensors"),
        map_location="cpu"
    )
    set_peft_model_state_dict(dit_peft, adapter_weights)

    # Find and load corresponding training state
    # ... (find matching training_state file by step number)
    training_state = torch.load(training_state_path, map_location="cpu")
    self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
    self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
    step = training_state["step"]

    print(f"Resumed LoRA training from step {step}")
    return step
```

**Why we unwrap the model:**

After `accelerator.prepare()`, the model is wrapped in DDP (DistributedDataParallel) or FSDP. We need to get the underlying model to access the PEFT API. The chain is:
```
DDP wrapper → CFM → PeftModel(DiT) → DiT
         ↑                 ↑
  accelerator.unwrap_model   unwrapped.transformer
```

### Full Trainer Code

```python
# model/trainer_lora.py

from __future__ import annotations

import os
import gc
from tqdm import tqdm
import wandb

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataset.dataset import DiffusionDataset
from torch.utils.data import DataLoader

from model import CFM
from model.utils import exists, default


class LoRATrainer:
    def __init__(
        self,
        model: CFM,
        args,
        epochs,
        learning_rate,
        num_warmup_updates=1000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        wandb_project="diffrhythm-lora",
        wandb_run_name="lora_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
    ):
        self.args = args

        # Same DDP config as original — no unused parameters because
        # all model components participate in forward pass
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

        logger = "wandb" if wandb.api.api_key else None

        self.accelerator = Accelerator(
            log_with=logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
        )

        if logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "lora_rank": getattr(args, 'lora_rank', None),
                    "lora_alpha": getattr(args, 'lora_alpha', None),
                },
            )

        self.model = model

        # NO EMA — not needed for LoRA fine-tuning (see rationale above)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/lora_default")

        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Optimizer: ONLY over trainable (LoRA) parameters
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        self.get_dataloader()
        self.get_scheduler()

        self.model, self.optimizer, self.scheduler, self.train_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_dataloader
            )

    def get_scheduler(self):
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = int(total_steps - warmup_steps)

        warmup = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(decay_steps, 1), eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    def get_dataloader(self):
        dd = DiffusionDataset(
            self.args.file_path, self.args.max_frames, self.args.min_frames,
            self.args.sampling_rate, self.args.downsample_rate,
            self.accelerator.state.mixed_precision.replace("no", "fp32")
        )
        self.train_dataloader = DataLoader(
            dataset=dd,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=dd.custom_collate_fn,
            persistent_workers=True,
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            dit_peft = unwrapped_model.transformer

            # 1. Save PEFT adapter (adapter_config.json + adapter_model.safetensors)
            if last:
                adapter_dir = f"{self.checkpoint_path}/lora_adapter_last"
            else:
                adapter_dir = f"{self.checkpoint_path}/lora_adapter_step_{step}"
            dit_peft.save_pretrained(adapter_dir)

            # 2. Save training state for resuming
            training_state = dict(
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if last:
                self.accelerator.save(training_state, f"{self.checkpoint_path}/training_state_last.pt")
            else:
                self.accelerator.save(training_state, f"{self.checkpoint_path}/training_state_{step}.pt")

            print(f"Saved LoRA checkpoint at step {step}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()

        # Look for latest adapter checkpoint
        has_last = os.path.exists(f"{self.checkpoint_path}/lora_adapter_last")
        if has_last:
            adapter_dir = f"{self.checkpoint_path}/lora_adapter_last"
            state_path = f"{self.checkpoint_path}/training_state_last.pt"
        else:
            adapter_dirs = sorted(
                [d for d in os.listdir(self.checkpoint_path)
                 if d.startswith("lora_adapter_step_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            if not adapter_dirs:
                return 0
            adapter_dir = f"{self.checkpoint_path}/{adapter_dirs[-1]}"
            step_num = adapter_dirs[-1].split("_")[-1]
            state_path = f"{self.checkpoint_path}/training_state_{step_num}.pt"

        # Load LoRA adapter weights
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        dit_peft = unwrapped_model.transformer

        adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            adapter_weights = load_file(adapter_file)
        else:
            adapter_weights = torch.load(
                os.path.join(adapter_dir, "adapter_model.bin"),
                map_location="cpu"
            )
        set_peft_model_state_dict(dit_peft, adapter_weights)

        # Load training state
        step = 0
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location="cpu")
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
            step = training_state["step"]
            del training_state
            gc.collect()

        print(f"Resumed LoRA checkpoint from step {step}")
        return step

    def train(self, resumable_with_seed: int = None):
        train_dataloader = self.train_dataloader

        start_step = self.load_checkpoint()
        global_step = start_step

        if resumable_with_seed and resumable_with_seed > 0:
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(
                train_dataloader, num_batches=skipped_batch
            )
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()

            if resumable_with_seed and resumable_with_seed > 0 and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    smoothing=0.15,
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["lrc"]
                    mel_spec = batch["latent"].permute(0, 2, 1)
                    mel_lengths = batch["latent_lengths"]
                    style_prompt = batch["prompt"]
                    style_prompt_lens = batch["prompt_lengths"]
                    start_time = batch["start_time"]

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths,
                        style_prompt=style_prompt,
                        style_prompt_lens=style_prompt_lens,
                        start_time=start_time,
                    )

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # NO self.ema_model.update() — not using EMA

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]},
                        step=global_step,
                    )

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()
```

---

## 7. The LoRA Config File: `config/lora_default.ini`

```ini
[DEFAULTS]

# Experiment name — checkpoint directory will be ckpts/{exp_name}/
exp_name = diffrhythm-lora-test

# ---- Data ----
file_path = "dataset/train.scp"
batch_size = 4
max_frames = 2048
min_frames = 512
sampling_rate = 44100
downsample_rate = 2048

# ---- LoRA hyperparameters ----
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05

# ---- Training ----
epochs = 10
learning_rate = 2e-4
num_warmup_updates = 100
save_per_updates = 500
last_per_steps = 500
grad_accumulation_steps = 1
max_grad_norm = 1.0

# ---- CFG dropout (same as original for consistency) ----
audio_drop_prob = 0.3
cond_drop_prob = 0.0
style_drop_prob = 0.1
lrc_drop_prob = 0.1

# ---- Pretrained model ----
# Leave empty to auto-download from HuggingFace Hub
pretrained_ckpt_path = ""
model_config = config/diffrhythm-1b.json

# ---- Resumable ----
resumable_with_seed = 666
reset_lr = 0
```

**Key differences from `config/default.ini`:**

| Parameter | Original | LoRA | Rationale |
|-----------|----------|------|-----------|
| `batch_size` | 8 | 4 | Base model VRAM is still large; lower batch size trades throughput for VRAM headroom. Increase if you have the memory. |
| `epochs` | 110 | 10 | LoRA converges much faster with far fewer trainable params |
| `learning_rate` | 7.5e-5 | 2e-4 | LoRA adapters need higher LR to move away from zero initialization |
| `num_warmup_updates` | 20 | 100 | Slightly longer warmup to stabilize the fresh Adam states on the LoRA params |
| `save_per_updates` | 5000 | 500 | Save more frequently — LoRA training is shorter so we want finer checkpoints |
| `min_frames` | 10 | 512 | Higher minimum to avoid degenerate short samples during fine-tuning |
| New: `lora_rank` | — | 16 | LoRA rank |
| New: `lora_alpha` | — | 32 | LoRA scaling factor |
| New: `lora_dropout` | — | 0.05 | LoRA dropout |
| New: `pretrained_ckpt_path` | — | "" | Path to pretrained weights |

---

## 8. The Launch Script: `scripts/train_lora.sh`

```bash
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD

accelerate launch --config-file config/accelerate_config.yaml \
    train/train_lora.py \
    --model-config config/diffrhythm-1b.json \
    --batch-size 4 \
    --max-frames 2048 \
    --min-frames 512 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --learning-rate 2e-4 \
    --num-warmup-updates 100 \
    --epochs 10 \
    --save-per-updates 500 \
    --last-per-steps 500 \
    --audio-drop-prob 0.3 \
    --style-drop-prob 0.1 \
    --lrc-drop-prob 0.1 \
    --grad-accumulation-steps 1 \
    --exp-name diffrhythm-lora-test \
    --file-path "dataset/train.scp" \
    --resumable-with-seed 666
```

This is structurally identical to `scripts/train.sh` — it uses `accelerate launch` with the same Accelerate config (DDP, mixed precision, GPU setup). The only differences are:
- Points to `train/train_lora.py` instead of `train/train.py`
- Includes LoRA-specific CLI args (`--lora-rank`, `--lora-alpha`, `--lora-dropout`)
- Uses different default hyperparameters

---

## 9. LoRA Inference: How to Use the Trained Adapter

After training, you'll have a directory like:
```
ckpts/diffrhythm-lora-test/
├── lora_adapter_last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── lora_adapter_step_500/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── training_state_last.pt
└── training_state_500.pt
```

### Option A: Load with PEFT (keeps adapter separate)

```python
from peft import PeftModel
from model import DiT, CFM

# 1. Build base DiT and load pretrained weights (same as training script)
dit = DiT(**model_config["model"], max_frames=max_frames)
checkpoint = torch.load("pretrained/cfm_model.pt", map_location="cpu")
# ... strip transformer. prefix and load as shown in the entry point ...
dit.load_state_dict(dit_state_dict, strict=False)

# 2. Load LoRA adapter on top
dit = PeftModel.from_pretrained(dit, "ckpts/diffrhythm-lora-test/lora_adapter_last")

# 3. Wrap in CFM and use for inference
model = CFM(transformer=dit, num_channels=64, max_frames=max_frames)
model = model.half().to(device).eval()

# 4. Use model.sample() exactly as the original inference code does
```

### Option B: Merge and unload (zero overhead, no PEFT dependency at inference)

```python
# After loading the adapter (step 2 above):
dit = dit.merge_and_unload()  # Merges LoRA weights into base weights permanently

# Now dit is a regular DiT with no PEFT wrapper
# Continue as normal
model = CFM(transformer=dit, num_channels=64, max_frames=max_frames)
```

**When to use which:**
- **Option A**: During development/experimentation, when you want to easily swap adapters
- **Option B**: For deployment, when you want the simplest possible inference with zero overhead

### Modifying the Existing Inference Script

To use a LoRA adapter with the existing `infer/infer.py`, you'd modify `infer/infer_utils.py`'s `prepare_model()` function. The changes are minimal — add the PEFT loading after the base model loading:

```python
def prepare_model(max_frames, device, lora_adapter_path=None):
    # ... existing code to build and load base CFM model ...

    if lora_adapter_path:
        from peft import PeftModel
        cfm.transformer = PeftModel.from_pretrained(
            cfm.transformer, lora_adapter_path
        )
        # Optional: merge for zero overhead
        # cfm.transformer = cfm.transformer.merge_and_unload()

    return cfm, tokenizer, muq, vae
```

---

## 10. Design Decisions Deep Dive

### Why the Dataset Stays Unchanged

The `DiffusionDataset` class in `dataset/dataset.py` loads three things per sample:
1. Lyrics tokens (`.pt` file with time alignment)
2. VAE-encoded audio latents (`.pt` file)
3. Style prompt embedding (`.pt` file)

None of these change for LoRA fine-tuning. The data format, loading logic, random cropping, lyrics alignment, and collation are all independent of whether the model being trained is full-parameter or LoRA. You just point `file_path` to your fine-tuning dataset's `.scp` file.

The only scenario where you'd need a modified dataset is if your fine-tuning data has a different format (e.g., raw audio instead of pre-computed latents). But that would require a new preprocessing pipeline, not a LoRA-specific change.

### Why CFM.forward() Stays Unchanged

Let's trace through `CFM.forward()` to see why it doesn't need changes:

```python
def forward(self, inp, text, style_prompt, ...):
    # 1. Sample noise x0 ~ N(0,I)
    x0 = torch.randn_like(x1)

    # 2. Sample time step t ~ sigmoid(N(0,1))
    time = torch.nn.functional.sigmoid(torch.normal(...))

    # 3. Interpolate: φ = (1-t)*x0 + t*x1
    φ = (1 - t) * x0 + t * x1
    flow = x1 - x0

    # 4. Create condition (zero out masked region)
    cond = torch.where(mask, zeros, x1)

    # 5. Random CFG dropout
    drop_audio_cond = random() < self.audio_drop_prob
    drop_text = random() < self.lrc_drop_prob
    drop_prompt = random() < self.style_drop_prob

    # 6. Call transformer ← THIS IS WHERE LORA MATTERS
    pred = self.transformer(
        x=φ, cond=cond, text=text, time=time,
        drop_audio_cond=drop_audio_cond, drop_text=drop_text,
        drop_prompt=drop_prompt, style_prompt=style_prompt,
        start_time=start_time
    )

    # 7. Compute loss
    loss = F.mse_loss(pred, flow, reduction="none")
    loss = loss[rand_span_mask]
    return loss.mean(), cond, pred
```

Step 6 calls `self.transformer(...)` which is our LoRA-wrapped DiT. The PeftModel wrapper intercepts this call and routes it through the LoRA-adapted forward method. The output shape and semantics are identical to the original DiT output. Steps 1–5 and 7 are pure math operations that don't involve any model parameters.

The flow matching training objective (predicting the velocity field `x1 - x0` from the noised interpolation `φ`) is agnostic to whether the transformer is LoRA-adapted or not. It's just asking "given this noisy input and conditions, what's the velocity?" — the same question, answered by a slightly different model.

### Why Gradient Clipping Works Without Changes

```python
self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
```

This iterates over all parameters, computes the total gradient norm across those that have gradients, and scales them down if the norm exceeds `max_grad_norm`. Frozen parameters have `None` gradients (not zero tensors — actually `None`). PyTorch's `clip_grad_norm_` skips parameters with `None` gradients.

So the gradient norm is computed over only the LoRA parameters, and clipping affects only them. This is correct behavior — we want to clip based on the gradients that actually contribute to the optimizer step.

### Why Gradient Checkpointing is Trickier with PEFT

The original trainer has a `grad_ckpt` flag that's passed through to the model. Gradient checkpointing trades compute for memory by not storing intermediate activations during the forward pass, recomputing them during backward.

With PEFT, there's a subtlety: gradient checkpointing must be applied to the **base model** (the DiT), not the PEFT wrapper. PEFT provides a helper:

```python
dit = get_peft_model(dit, lora_config)

# Enable gradient checkpointing on the base model
dit.enable_input_require_grads()  # Required for PEFT + grad checkpointing
dit.base_model.model.gradient_checkpointing_enable()
```

The `enable_input_require_grads()` call is necessary because gradient checkpointing requires input tensors to have `requires_grad=True` to properly recompute activations during backward. Without it, the recomputation would fail silently and produce incorrect gradients.

For simplicity, the LoRA trainer code above omits gradient checkpointing. To add it, insert those two lines after `get_peft_model()` in the entry point.

### VRAM Budget Comparison

For a single GPU with fp16 mixed precision:

| Component | Full Training | LoRA Training | Savings |
|-----------|--------------|---------------|---------|
| Model weights (fp16) | ~2 GB | ~2 GB | Same — base model is still loaded |
| Optimizer states | ~8 GB | ~64 MB | **125x** reduction |
| Gradients | ~2 GB | ~16 MB | **125x** reduction |
| Activations (bs=4) | ~4 GB | ~4 GB | Same — forward pass is identical |
| EMA shadow | ~2 GB | 0 | Eliminated |
| **Total** | **~18 GB** | **~6 GB** | **~3x** reduction |

The main VRAM savings come from the optimizer states and the elimination of EMA. The model weights and activations are the same because the full base model still needs to be in memory and the forward pass is identical (LoRA adds a small branch, not a separate pass).

With gradient checkpointing enabled, activations drop from ~4GB to ~1GB at the cost of ~30% slower training.

---

## 11. Summary of All Changes

### New Files

| File | Lines (approx) | Purpose |
|------|----------------|---------|
| `train/train_lora.py` | ~80 | Entry point: load pretrained → apply LoRA → create trainer |
| `model/trainer_lora.py` | ~250 | Trainer: filtered optimizer, no EMA, PEFT checkpoints, cosine scheduler |
| `config/lora_default.ini` | ~35 | Hyperparameters: higher LR, fewer epochs, LoRA-specific params |
| `scripts/train_lora.sh` | ~20 | Launch script with LoRA-appropriate defaults |

### Unchanged Files

| File | Why Unchanged |
|------|---------------|
| `model/cfm.py` | Flow matching math is model-agnostic |
| `model/dit.py` | LoRA wraps externally, DiT code untouched |
| `model/modules.py` | Building blocks unchanged |
| `model/utils.py` | Helpers are generic |
| `dataset/dataset.py` | Same data format for fine-tuning |
| `config/diffrhythm-1b.json` | Same architecture |
| `config/accelerate_config.yaml` | Same distributed/mixed-precision setup |

### New Dependency

```
peft>=0.7.0
```

Add to `requirements.txt` or install with `pip install peft`.

### Checkpoint Output Structure

```
ckpts/diffrhythm-lora-test/
├── lora_adapter_step_500/
│   ├── adapter_config.json        (~1 KB — LoRA config for reproducibility)
│   └── adapter_model.safetensors  (~32 MB — just the A and B matrices)
├── lora_adapter_last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── training_state_500.pt          (~64 MB — optimizer + scheduler state)
└── training_state_last.pt
```

Compare to full training checkpoints: `model_{step}.pt` at ~4-16 GB each.
