# LoRA Fine-Tuning: A Code-Level Guide

This guide covers how LoRA (Low-Rank Adaptation) fine-tuning works from a code perspective using HuggingFace's `peft` library, and then maps those patterns onto the DiffRhythm codebase specifically.

---

## Table of Contents

1. [The PEFT Library](#1-the-peft-library)
2. [Core API: LoraConfig and get_peft_model](#2-core-api-loraconfig-and-get_peft_model)
3. [Choosing Target Modules](#3-choosing-target-modules)
4. [The Training Loop with LoRA](#4-the-training-loop-with-lora)
5. [Saving and Loading LoRA Weights](#5-saving-and-loading-lora-weights)
6. [Merging LoRA Weights Back Into the Base Model](#6-merging-lora-weights-back-into-the-base-model)
7. [Full Minimal Example](#7-full-minimal-example)
8. [Applying LoRA to DiffRhythm](#8-applying-lora-to-diffrhythm)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Useful Reference Links](#10-useful-reference-links)

---

## 1. The PEFT Library

**PEFT** (Parameter-Efficient Fine-Tuning) is the HuggingFace library that implements LoRA and other adapter methods. Install it with:

```bash
pip install peft
```

The two imports you'll use most often:

```python
from peft import LoraConfig, get_peft_model, PeftModel
```

| Import | Purpose |
|---|---|
| `LoraConfig` | Defines the LoRA hyperparameters (rank, alpha, target modules, etc.) |
| `get_peft_model` | Wraps an existing `nn.Module` with LoRA layers, freezing the base weights |
| `PeftModel` | Used to load a saved LoRA adapter back onto a base model |

PEFT works by **wrapping** your existing PyTorch model. It doesn't require your model to be a HuggingFace `PreTrainedModel` — any `nn.Module` works. This is important for DiffRhythm since `DiT` and `CFM` are plain PyTorch modules.

---

## 2. Core API: LoraConfig and get_peft_model

### LoraConfig

This object defines *what* gets adapted and *how*:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                          # rank of the low-rank matrices
    lora_alpha=32,                 # scaling factor (effective scale = alpha / r)
    target_modules=["q_proj", "v_proj"],  # which Linear layers to adapt
    lora_dropout=0.05,             # dropout on the LoRA path
    bias="none",                   # "none", "all", or "lora_only"
    task_type=TaskType.FEATURE_EXTRACTION,  # or None for custom models
)
```

**Key parameters explained:**

- **`r` (rank):** The dimension of the low-rank matrices A and B. Lower = fewer params, higher = more capacity. Common values: 4, 8, 16, 32, 64. Start with 16.
- **`lora_alpha`:** A scaling constant. The LoRA output is scaled by `alpha / r`. A common pattern is to set `lora_alpha = 2 * r` so the effective scale is 2.0. In practice, many people just set `lora_alpha = r` (scale of 1.0) or leave alpha at 32 regardless of r.
- **`target_modules`:** A list of strings. PEFT searches the model's named modules and applies LoRA to any `nn.Linear` whose name **contains** one of these strings. You can also pass `"all-linear"` to target every linear layer.
- **`lora_dropout`:** Dropout applied to the LoRA branch during training. 0.0–0.1 is typical.
- **`bias`:** Whether to also train biases. `"none"` means no bias training (most common). `"all"` trains all biases. `"lora_only"` trains only the biases of LoRA-adapted layers.
- **`task_type`:** Hint for PEFT about the model architecture. For custom models that aren't standard HF models, you can use `TaskType.FEATURE_EXTRACTION` or just omit it / set `None`.

### get_peft_model

This function takes your model + config and returns a wrapped version:

```python
from peft import get_peft_model

model = YourModel(...)          # any nn.Module
model = get_peft_model(model, lora_config)
```

What `get_peft_model` does internally:

1. **Iterates** over all named modules in the model.
2. For each `nn.Linear` whose name matches a string in `target_modules`, it **replaces** that layer with a `LoraLinear` wrapper that contains the frozen original weight plus trainable low-rank matrices A and B.
3. **Freezes** all original parameters (`requires_grad = False`).
4. Leaves the LoRA matrices and any specified biases as trainable.

After this call, you can verify what's trainable:

```python
model.print_trainable_parameters()
# Output: trainable params: 3,407,872 || all params: 1,073,741,824 || trainable%: 0.3174
```

---

## 3. Choosing Target Modules

### Inspecting Your Model

Before configuring LoRA you need to know the names of the layers inside your model. The easiest way:

```python
# Print all named modules
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name, module.in_features, module.out_features)
```

For a model using `LlamaDecoderLayer` (like DiffRhythm), you'll see patterns like:

```
transformer.transformer_blocks.0.self_attn.q_proj  2048 2048
transformer.transformer_blocks.0.self_attn.k_proj  2048 2048
transformer.transformer_blocks.0.self_attn.v_proj  2048 2048
transformer.transformer_blocks.0.self_attn.o_proj  2048 2048
transformer.transformer_blocks.0.mlp.gate_proj     2048 8192
transformer.transformer_blocks.0.mlp.up_proj       2048 8192
transformer.transformer_blocks.0.mlp.down_proj     8192 2048
```

### Common Target Module Strategies

| Strategy | `target_modules` | Trainable % | Notes |
|---|---|---|---|
| Attention only (minimal) | `["q_proj", "v_proj"]` | ~0.15% | Classic LoRA paper approach |
| Full attention | `["q_proj", "k_proj", "v_proj", "o_proj"]` | ~0.3% | More expressive |
| Attention + MLP | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | ~0.8% | Most common for serious fine-tuning |
| Everything | `"all-linear"` | ~1-2% | PEFT auto-targets all `nn.Linear` layers |

The string matching is substring-based and applies across the entire model hierarchy. So `"q_proj"` will match `transformer_blocks.0.self_attn.q_proj`, `transformer_blocks.1.self_attn.q_proj`, etc.

### modules_to_save

There's also a `modules_to_save` parameter for layers you want to **fully train** (not apply LoRA to, but unfreeze entirely):

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    modules_to_save=["proj_out", "norm_out"],  # these get fully unfrozen
)
```

This is useful for output heads, final norms, or embedding layers that may need full adaptation.

---

## 4. The Training Loop with LoRA

The training loop barely changes. The LoRA-wrapped model behaves identically to the original — same forward signature, same outputs. The only differences are practical:

```python
import torch
from torch.optim import AdamW

# 1. Build model and wrap with LoRA
model = YourModel(...)
model = get_peft_model(model, lora_config)

# 2. Create optimizer — only trainable (LoRA) params
#    filter(lambda p: p.requires_grad, ...) is the key pattern
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # LoRA typically uses higher LR than full fine-tuning
)

# 3. Standard training loop — nothing special
model.train()
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Important practical differences from full fine-tuning:**

- **Learning rate:** LoRA adapters typically use a higher learning rate than full fine-tuning (e.g., `1e-4` to `5e-4` instead of `7.5e-5`). The adapters are initialized near zero so they need a stronger signal.
- **Optimizer memory:** Since only ~0.3% of parameters are trainable, the optimizer states (momentum, variance in Adam) are proportionally smaller. This is a huge VRAM saving.
- **No EMA needed:** EMA (Exponential Moving Average) is typically unnecessary for LoRA fine-tuning since you're training so few parameters and for fewer steps.
- **Gradient checkpointing:** Still works normally if needed. Apply it to the base model *before* wrapping with LoRA.

### Using with HuggingFace Accelerate

PEFT integrates seamlessly with Accelerate (which DiffRhythm already uses):

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp16")

model = YourModel(...)
model = get_peft_model(model, lora_config)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = get_scheduler(...)

# Accelerate preparation works as normal
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)
```

---

## 5. Saving and Loading LoRA Weights

### Saving (only the adapter weights)

```python
# Save just the LoRA adapter — typically < 50MB vs multi-GB full model
model.save_pretrained("path/to/lora_adapter")
```

This creates a directory with:
```
path/to/lora_adapter/
├── adapter_config.json    # The LoraConfig you used
└── adapter_model.safetensors  # Only the LoRA A, B matrices (tiny)
```

### Loading (adapter onto base model)

```python
from peft import PeftModel

# 1. Recreate the base model (with frozen pretrained weights)
base_model = YourModel(...)
base_model.load_state_dict(torch.load("pretrained_weights.pt"))

# 2. Load the LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "path/to/lora_adapter")
```

### Manual saving/loading (alternative)

If you prefer manual control (useful for custom checkpoint formats like DiffRhythm's):

```python
# Save: extract only the LoRA state dict
lora_state_dict = {
    k: v for k, v in model.named_parameters() if v.requires_grad
}
torch.save(lora_state_dict, "lora_weights.pt")

# Load: apply to a LoRA-wrapped model
model = get_peft_model(base_model, lora_config)
lora_weights = torch.load("lora_weights.pt")
model.load_state_dict(lora_weights, strict=False)  # strict=False ignores frozen params
```

### Using get_peft_model_state_dict (recommended)

PEFT provides a dedicated helper:

```python
from peft import get_peft_model_state_dict, set_peft_model_state_dict

# Save
state_dict = get_peft_model_state_dict(model)
torch.save(state_dict, "lora_weights.pt")

# Load
lora_weights = torch.load("lora_weights.pt")
set_peft_model_state_dict(model, lora_weights)
```

---

## 6. Merging LoRA Weights Back Into the Base Model

After fine-tuning, you can merge the LoRA weights permanently into the base model so there's zero inference overhead:

```python
# Merge LoRA weights into the base model
model = model.merge_and_unload()

# Now model is a regular nn.Module with no LoRA wrapper
# Save the full merged model
torch.save(model.state_dict(), "merged_model.pt")
```

This is useful when:
- You want to deploy without the PEFT dependency
- You want zero inference overhead (no adapter computation)
- You want to use the fine-tuned model as a new base for further LoRA training

---

## 7. Full Minimal Example

Here's a complete standalone example showing the entire LoRA fine-tuning lifecycle:

```python
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from peft import LoraConfig, get_peft_model, PeftModel

# ---- 1. Define a toy transformer model ----
class ToyTransformer(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8):
        super().__init__()
        self.embed = nn.Linear(64, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "q_proj": nn.Linear(dim, dim),
                "k_proj": nn.Linear(dim, dim),
                "v_proj": nn.Linear(dim, dim),
                "o_proj": nn.Linear(dim, dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
                "norm": nn.LayerNorm(dim),
            }))
        self.head = nn.Linear(dim, 64)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            q = layer["q_proj"](x)
            k = layer["k_proj"](x)
            v = layer["v_proj"](x)
            attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
            x = x + layer["o_proj"](attn @ v)
            x = x + layer["ffn"](layer["norm"](x))
        return self.head(x)

# ---- 2. Configure and apply LoRA ----
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = ToyTransformer(dim=256, depth=4)
print(f"Total params before LoRA: {sum(p.numel() for p in model.parameters()):,}")

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Example output: trainable params: 73,728 || all params: 2,227,520 || trainable%: 3.31

# ---- 3. Training ----
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

# Dummy data
dataset = TensorDataset(torch.randn(1000, 32, 64), torch.randn(1000, 32, 64))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model.train()
for epoch in range(3):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---- 4. Save the adapter ----
model.save_pretrained("my_lora_adapter")

# ---- 5. Load adapter onto a fresh base model ----
fresh_base = ToyTransformer(dim=256, depth=4)
# (load pretrained weights into fresh_base here if applicable)
loaded_model = PeftModel.from_pretrained(fresh_base, "my_lora_adapter")
loaded_model.eval()

# ---- 6. Optional: merge and unload for deployment ----
merged_model = loaded_model.merge_and_unload()
torch.save(merged_model.state_dict(), "merged_model.pt")
```

---

## 8. Applying LoRA to DiffRhythm

Now let's map everything above onto this specific codebase. The DiffRhythm architecture is:

```
CFM (Conditional Flow Matching wrapper)
└── transformer: DiT
    ├── time_embed: TimestepEmbedding
    ├── text_embed: TextEmbedding
    ├── input_embed: InputEmbedding
    ├── transformer_blocks: ModuleList of 16 LlamaDecoderLayer
    │   └── each block contains:
    │       ├── self_attn.q_proj  (2048 → 2048)
    │       ├── self_attn.k_proj  (2048 → 2048)
    │       ├── self_attn.v_proj  (2048 → 2048)
    │       ├── self_attn.o_proj  (2048 → 2048)
    │       ├── mlp.gate_proj     (2048 → 8192)
    │       ├── mlp.up_proj       (2048 → 8192)
    │       └── mlp.down_proj     (8192 → 2048)
    ├── text_fusion_linears: ModuleList of 8 Linear+SiLU
    ├── norm_out: AdaLayerNormZero_Final
    └── proj_out: Linear(2048 → 64)
```

### Step-by-step: What to Change

#### A. After building the model, wrap the DiT with LoRA

The LoRA should target the `DiT` (the inner transformer), **not** the outer `CFM` wrapper. This is because the CFM wrapper is just flow-matching logic — the learnable capacity lives in the DiT.

```python
from peft import LoraConfig, get_peft_model

# Build model as the existing code does
dit = DiT(**model_config["model"], max_frames=args.max_frames)

# Load pretrained weights into the DiT
pretrained = torch.load("path/to/pretrained_dit.pt", map_location="cpu")
dit.load_state_dict(pretrained, strict=False)

# Configure LoRA targeting the LlamaDecoderLayer attention + MLP
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
)

# Wrap the DiT (not the CFM) with LoRA
dit = get_peft_model(dit, lora_config)
dit.print_trainable_parameters()

# Then wrap in CFM as before
model = CFM(
    transformer=dit,
    num_channels=model_config["model"]["mel_dim"],
    audio_drop_prob=args.audio_drop_prob,
    cond_drop_prob=args.cond_drop_prob,
    style_drop_prob=args.style_drop_prob,
    lrc_drop_prob=args.lrc_drop_prob,
    max_frames=args.max_frames,
)
```

> **Why target DiT and not CFM?** The CFM wrapper has no trainable layers of its own. If you wrapped CFM, `get_peft_model` would still find the same Linear layers inside, but the module hierarchy paths would be longer (e.g., `transformer.transformer_blocks.0...`) and saving/loading would include the CFM wrapper in the PEFT checkpoint structure. Targeting DiT directly keeps things cleaner. However, targeting CFM would also work functionally — PEFT searches recursively.

#### B. Modify the optimizer to only use trainable parameters

In `model/trainer.py`, the optimizer currently does:

```python
self.optimizer = AdamW(model.parameters(), lr=learning_rate)
```

Change to:

```python
self.optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
)
```

This ensures the optimizer only tracks the LoRA parameters, dramatically reducing memory.

#### C. Adjust the learning rate

The existing config uses `learning_rate = 7.5e-5`. For LoRA fine-tuning, increase this:

```ini
learning_rate = 2e-4   # or even 5e-4
```

LoRA adapters are initialized near zero (A is random, B is zero), so they need a stronger learning rate signal to move away from the identity function.

#### D. Simplify the scheduler

The current warmup+decay scheduler is designed for long full-training runs. For LoRA fine-tuning (typically much shorter), a cosine schedule or constant-with-warmup often works better:

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
)
```

Or simply use the existing `get_constant_scheduler` method that's already in the Trainer.

#### E. Modify checkpoint saving/loading

The current save/load in `trainer.py` saves the full model state dict. For LoRA, you want to save only the adapter:

**Saving:**
```python
def save_checkpoint(self, step, last=False):
    self.accelerator.wait_for_everyone()
    if self.is_main:
        # Get just the LoRA weights from the DiT inside the CFM
        dit_model = self.accelerator.unwrap_model(self.model).transformer
        lora_state_dict = {
            k: v for k, v in dit_model.named_parameters() if v.requires_grad
        }
        checkpoint = dict(
            lora_state_dict=lora_state_dict,
            optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            step=step,
        )
        save_name = "model_last.pt" if last else f"model_{step}.pt"
        self.accelerator.save(checkpoint, f"{self.checkpoint_path}/{save_name}")

        # Also save in PEFT format for easy loading later
        dit_model.save_pretrained(f"{self.checkpoint_path}/lora_adapter_step_{step}")
```

**Loading:**
```python
def load_checkpoint(self):
    # ... find latest checkpoint file ...
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load LoRA weights
    dit_model = self.accelerator.unwrap_model(self.model).transformer
    dit_model.load_state_dict(checkpoint["lora_state_dict"], strict=False)

    # Restore optimizer and scheduler
    if "optimizer_state_dict" in checkpoint:
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and not self.reset_lr:
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("step", 0)
```

#### F. Disable or simplify EMA

The current code uses `ema_pytorch` to maintain an EMA copy of the model. For LoRA fine-tuning you have two options:

1. **Disable EMA entirely** (simplest, recommended for LoRA):
   ```python
   # In trainer __init__, skip EMA setup
   # self.ema_model = EMA(model, ...)  # comment out
   ```

2. **EMA only over LoRA params** (if you really want it):
   ```python
   # This requires more custom work — not worth it for most LoRA use cases
   ```

#### G. Gradient clipping still works as-is

The existing gradient clipping code is already fine:

```python
if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
```

This will only affect parameters with gradients (i.e., the LoRA params), so no changes needed.

#### H. Inference with the LoRA adapter

For inference, load the base model then apply the adapter:

```python
from peft import PeftModel

# Build base DiT and load pretrained weights
dit = DiT(**model_config["model"], max_frames=args.max_frames)
dit.load_state_dict(torch.load("pretrained_dit.pt"), strict=False)

# Load LoRA adapter
dit = PeftModel.from_pretrained(dit, "ckpts/my_lora_run/lora_adapter_step_5000")

# Optional: merge for zero-overhead inference
dit = dit.merge_and_unload()

# Wrap in CFM and use normally
model = CFM(transformer=dit, ...)
model.eval()
output = model.sample(...)
```

### Summary of Files to Modify

| File | Change |
|---|---|
| `train/train.py` | Add `LoraConfig` + `get_peft_model` after DiT creation, load pretrained base weights |
| `model/trainer.py` | Filter optimizer params, adjust LR, modify save/load for LoRA, disable EMA |
| `config/default.ini` | Higher learning rate, fewer epochs, adjusted warmup |
| `infer/` scripts | Add `PeftModel.from_pretrained()` or `merge_and_unload()` before inference |

### Estimated Parameter Counts

With `r=16` targeting attention + MLP on the DiffRhythm 1B model:

| Component | Parameters |
|---|---|
| Base model (frozen) | ~1,000,000,000 |
| LoRA adapters (trainable) | ~8,000,000 (~0.8%) |
| Optimizer states (Adam) | ~16,000,000 (2x LoRA params) |

Compared to full fine-tuning where optimizer states alone would be ~2B parameters (2x the model).

---

## 9. Common Pitfalls

### Don't forget to load pretrained weights *before* applying LoRA

```python
# WRONG: apply LoRA to a randomly initialized model
model = DiT(...)
model = get_peft_model(model, lora_config)  # LoRA on random weights = useless

# RIGHT: load pretrained weights first, then apply LoRA
model = DiT(...)
model.load_state_dict(torch.load("pretrained.pt"))
model = get_peft_model(model, lora_config)  # LoRA on top of good pretrained weights
```

### Don't pass all parameters to the optimizer

```python
# WRONG: wastes memory on optimizer states for frozen params
optimizer = AdamW(model.parameters(), lr=1e-4)

# RIGHT: only trainable params
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

PyTorch won't error on the wrong version — it will just silently allocate Adam states for frozen parameters, wasting VRAM.

### Watch out for model wrapping order with Accelerate

```python
# CORRECT order:
dit = get_peft_model(dit, lora_config)    # 1. LoRA first
model = CFM(transformer=dit, ...)          # 2. Then wrap in your model
model = accelerator.prepare(model)         # 3. Accelerate last

# WRONG: preparing before LoRA will cause issues
model = accelerator.prepare(model)
model = get_peft_model(model, lora_config)  # Don't do this
```

### strict=False when loading partial state dicts

When loading LoRA weights into a LoRA-wrapped model, always use `strict=False` because the state dict only contains the LoRA keys, not the frozen base model keys:

```python
model.load_state_dict(lora_weights, strict=False)
```

### Saving with Accelerate's unwrap

When saving, always unwrap the model first to get past the DDP/FSDP wrapper:

```python
unwrapped = accelerator.unwrap_model(model)
unwrapped.transformer.save_pretrained("lora_adapter")
```

---

## 10. Useful Reference Links

- **PEFT Documentation:** https://huggingface.co/docs/peft
- **PEFT GitHub:** https://github.com/huggingface/peft
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **PEFT with custom models:** https://huggingface.co/docs/peft/developer_guides/custom_models
- **F5-TTS (DiffRhythm ancestor, has LoRA examples):** https://github.com/SWivid/F5-TTS
