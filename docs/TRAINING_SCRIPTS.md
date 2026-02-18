# DiffRhythm Training Scripts

This document provides a detailed explanation of every script and configuration file involved in training the DiffRhythm model. DiffRhythm uses **Conditional Flow Matching (CFM)** with a **Diffusion Transformer (DiT)** backbone, trained via HuggingFace Accelerate for distributed GPU support.

---

## Table of Contents

- [File Overview](#file-overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Entry Point: train/train.py](#entry-point-traintrainpy)
- [Training Orchestrator: model/trainer.py](#training-orchestrator-modeltrainerpy)
  - [Initialization](#initialization)
  - [DataLoader Setup](#dataloader-setup)
  - [Learning Rate Scheduler](#learning-rate-scheduler)
  - [Training Loop](#training-loop)
  - [Checkpointing](#checkpointing)
- [Conditional Flow Matching: model/cfm.py](#conditional-flow-matching-modelcfmpy)
  - [Training Forward Pass](#training-forward-pass)
  - [Loss Computation](#loss-computation)
- [Diffusion Transformer: model/dit.py](#diffusion-transformer-modelditpy)
  - [TextEmbedding](#textembedding)
  - [InputEmbedding](#inputembedding)
  - [DiT Transformer](#dit-transformer)
- [Dataset: dataset/dataset.py](#dataset-datasetdatasetpy)
  - [Data Format](#data-format)
  - [Sample Loading](#sample-loading)
  - [Collation](#collation)
- [Utility Functions: model/utils.py](#utility-functions-modelutilspy)
- [Configuration Files](#configuration-files)
  - [Default Training Config: config/default.ini](#default-training-config-configdefaultini)
  - [Model Architecture Config: config/diffrhythm-1b.json](#model-architecture-config-configdiffrhythm-1bjson)
  - [Accelerate Config: config/accelerate_config.yaml](#accelerate-config-configaccelerate_configyaml)
- [Launch Script: scripts/train.sh](#launch-script-scriptstrainsh)
- [How to Launch Training](#how-to-launch-training)
- [Training Workflow Summary](#training-workflow-summary)

---

## File Overview

| File | Purpose |
|------|---------|
| `train/train.py` | Main training entry point |
| `model/trainer.py` | Training loop, optimizer, scheduler, checkpointing |
| `model/cfm.py` | Conditional Flow Matching model (loss + sampling) |
| `model/dit.py` | Diffusion Transformer backbone |
| `model/modules.py` | Reusable neural network building blocks |
| `model/utils.py` | Helper functions (masking, tokenization, seeding) |
| `model/__init__.py` | Exports `CFM`, `DiT`, and `Trainer` |
| `dataset/dataset.py` | PyTorch Dataset and DataLoader collation |
| `config/default.ini` | Default training hyperparameters |
| `config/diffrhythm-1b.json` | Model architecture specification |
| `config/accelerate_config.yaml` | HuggingFace Accelerate distributed training config |
| `scripts/train.sh` | Shell script to launch training |

---

## Architecture at a Glance

```
scripts/train.sh
  └── train/train.py                      (entry point)
        ├── config/default.ini            (hyperparameters via prefigure)
        ├── config/diffrhythm-1b.json     (model architecture)
        └── model/trainer.py              (Trainer class)
              ├── dataset/dataset.py      (DiffusionDataset)
              ├── model/cfm.py            (CFM wrapper)
              │     └── model/dit.py      (DiT transformer)
              │           └── model/modules.py (building blocks)
              └── model/utils.py          (helpers)
```

---

## Entry Point: `train/train.py`

This is the main script that wires everything together. It:

1. **Loads configuration** from `config/default.ini` using the `prefigure` library, which merges INI defaults with command-line overrides.
2. **Reads the model architecture** from a JSON file (e.g., `config/diffrhythm-1b.json`).
3. **Instantiates the model**: creates a `DiT` transformer, wraps it in a `CFM` module with classifier-free guidance drop probabilities.
4. **Creates a `Trainer`** instance with all training hyperparameters.
5. **Starts training** with optional resumable seed for reproducible dataset shuffling.

Key code flow:

```python
args = get_all_args("config/default.ini")      # load config + CLI overrides
model_config = json.load(open(args.model_config))

model = CFM(
    transformer=DiT(**model_config["model"], max_frames=args.max_frames),
    audio_drop_prob=args.audio_drop_prob,
    cond_drop_prob=args.cond_drop_prob,
    style_drop_prob=args.style_drop_prob,
    lrc_drop_prob=args.lrc_drop_prob,
    max_frames=args.max_frames
)

trainer = Trainer(model, args, ...)
trainer.train(resumable_with_seed=args.resumable_with_seed)
```

Environment variables `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are both set to `"1"` to avoid thread contention during distributed training.

---

## Training Orchestrator: `model/trainer.py`

The `Trainer` class (adapted from [F5-TTS](https://github.com/SWivid/F5-TTS)) manages the full training lifecycle.

### Initialization

The constructor sets up:

| Component | Details |
|-----------|---------|
| **Accelerator** | HuggingFace Accelerate with DDP (no unused parameters), optional WandB logging |
| **EMA Model** | Exponential Moving Average via `ema_pytorch.EMA`, maintained on the main process only |
| **Optimizer** | `AdamW` (default) or `bitsandbytes.AdamW8bit` for memory-efficient training |
| **DataLoader** | Built from `DiffusionDataset` |
| **LR Scheduler** | Warmup + linear decay (see below) |

All four components (model, optimizer, scheduler, dataloader) are passed through `accelerator.prepare()` for distributed training.

### DataLoader Setup

```python
def get_dataloader(self):
    dd = DiffusionDataset(
        file_path, max_frames, min_frames,
        sampling_rate, downsample_rate, precision
    )
    self.train_dataloader = DataLoader(
        dataset=dd,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dd.custom_collate_fn,
        persistent_workers=True
    )
```

- Uses 4 worker processes with persistent workers and pinned memory for efficient GPU transfer.
- A custom collation function handles variable-length padding.

### Learning Rate Scheduler

A two-phase `SequentialLR` scheduler:

1. **Warmup phase**: `LinearLR` ramps from `1e-8` to `1.0` over `num_warmup_updates * num_processes` steps.
2. **Decay phase**: `LinearLR` decays from `1.0` to `1e-8` over the remaining steps.

Total steps are calculated as: `(len(dataloader) * epochs) / grad_accumulation_steps`.

An alternative `get_constant_scheduler()` method is also available for constant learning rate experiments.

### Training Loop

The `train()` method implements:

1. **Checkpoint loading** — resumes from the latest checkpoint if available.
2. **Resumable epoch/batch skipping** — when `resumable_with_seed > 0`, calculates the exact epoch and batch to resume from and uses `accelerator.skip_first_batches()`.
3. **Per-batch training step**:
   - Extracts `lrc` (lyrics tokens), `latent` (VAE-encoded audio), `prompt` (style embedding), and `start_time` from the batch.
   - Calls `model.forward()` which returns `(loss, cond, pred)`.
   - Backward pass with gradient accumulation via `accelerator.accumulate()`.
   - Gradient clipping when `max_grad_norm > 0`.
   - Optimizer and scheduler step.
   - EMA model update (main process only).
   - Logs `loss` and `lr` to WandB.
4. **Checkpoint saving**:
   - Regular checkpoints every `save_per_updates * grad_accumulation_steps` steps.
   - "Last" checkpoint every `last_per_steps` steps.
   - Final checkpoint at end of training.

### Checkpointing

**Saved state** (`save_checkpoint`):
- Model state dict
- Optimizer state dict
- EMA model state dict
- Scheduler state dict
- Current global step

Checkpoints are saved to `ckpts/{exp_name}/` as either `model_{step}.pt` or `model_last.pt`.

**Loading** (`load_checkpoint`):
- Finds the latest checkpoint (`model_last.pt` preferred, otherwise highest step number).
- Uses flexible shape matching — filters out parameters whose shapes don't match, enabling architecture changes between runs.
- Optionally resets the LR scheduler via the `reset_lr` flag.

---

## Conditional Flow Matching: `model/cfm.py`

The `CFM` class wraps the transformer backbone and implements the flow matching training objective.

### Training Forward Pass

The `forward()` method implements the flow matching training procedure:

1. **Create masks**: Generate a random span mask via `mask_from_frac_lengths()` with fraction range `(0.7, 1.0)` of the sequence length. This determines which portion of the audio is to be predicted (infilled).
2. **Sample noise**: `x0 ~ N(0, I)` (Gaussian noise).
3. **Sample time step**: `t ~ sigmoid(N(0, 1))` — a logit-normal distribution biased toward intermediate time steps.
4. **Interpolate**: Compute the noised sample `φ = (1-t) * x0 + t * x1` where `x1` is the ground-truth mel latent.
5. **Compute flow target**: `flow = x1 - x0` (the velocity field).
6. **Condition masking**: Zero out the masked span in the condition audio: `cond = where(mask, 0, x1)`.
7. **Classifier-free guidance dropout**: Randomly drop each conditioning signal independently:
   - `drop_audio_cond` with probability `audio_drop_prob`
   - `drop_text` with probability `lrc_drop_prob`
   - `drop_prompt` with probability `style_drop_prob`
8. **Predict flow**: Pass through the DiT transformer.
9. **Compute loss**: MSE between predicted and target flow, masked to the random span only.

### Loss Computation

```python
loss = F.mse_loss(pred, flow, reduction="none")
loss = loss[rand_span_mask]
return loss.mean(), cond, pred
```

The loss is only computed over the masked (predicted) region, consistent with the infilling objective.

---

## Diffusion Transformer: `model/dit.py`

The `DiT` class is the transformer backbone that predicts the flow (velocity field) during training.

### TextEmbedding

Embeds integer token IDs into continuous representations:
- Lookup embedding: `nn.Embedding(text_num_embeds + 1, text_dim)`
- Optional positional encoding via sinusoidal embeddings
- Optional `ConvNeXtV2Block` layers for contextual text processing

When `drop_text=True` (classifier-free guidance), all text tokens are zeroed.

### InputEmbedding

Combines all conditioning signals into a single input:
- Concatenates: noised audio `x`, masked condition audio `cond`, text embeddings, style embedding (broadcast), and time embedding (broadcast)
- Projects through a linear layer to the transformer dimension
- Adds convolutional position embeddings

### DiT Transformer

| Component | Details |
|-----------|---------|
| **Time embedding** | `TimestepEmbedding` for diffusion time step `t` |
| **Start time embedding** | `TimestepEmbedding` for normalized start position in the song |
| **Duration embedding** | `TimestepEmbedding` for song duration (only when `max_frames=6144`) |
| **Transformer blocks** | `LlamaDecoderLayer` from HuggingFace Transformers (16 layers for 1B config) |
| **Rotary embeddings** | `LlamaRotaryEmbedding` for relative position encoding |
| **Text fusion** | Linear layers inject text information into the first `depth // 2` transformer layers |
| **Output norm** | `AdaLayerNormZero_Final` with time conditioning |
| **Output projection** | Linear layer to `mel_dim` (64 channels) |

The forward pass:
1. Embeds time step, start time, and duration into a combined conditioning vector `c = t + s_t + d_t`.
2. Embeds text tokens.
3. Combines all inputs via `InputEmbedding`.
4. Passes through Llama transformer blocks with rotary position embeddings.
5. Injects text residuals into the first half of the layers.
6. Applies final adaptive layer norm and projects to mel dimension.

---

## Dataset: `dataset/dataset.py`

### Data Format

Training data is specified via a `.scp` file (e.g., `dataset/train.scp`). Each line has the format:

```
utt_id|lrc_path|latent_path|style_path
```

Example:
```
2626046476|dataset/lrc/2626046476.pt|dataset/latent/2626046476.pt|dataset/style/2626046476.pt
```

The three `.pt` files contain:
- **lrc** (`lrc_path`): A dict with `time` (list of timestamps) and `lrc` (list of token sequences) — lyrics aligned with audio.
- **latent** (`latent_path`): VAE-encoded audio latent representation, shape `[1, d, t]` where `d` is the mel dimension and `t` is the number of frames.
- **style** (`style_path`): Style prompt embedding, shape `[1, d]` — a fixed-size representation of a reference audio clip.

### Sample Loading

The `get_triple()` method processes a single training sample:

1. Loads the LRC, latent, and style tensors from `.pt` files.
2. **Random cropping**: Selects a random start frame within the latent, crops to `max_frames` length, and computes a normalized start time.
3. **Lyrics alignment**: Filters lyrics to the cropped time window and maps token positions to audio frames based on timestamps.
4. Converts tensors to the configured precision (fp16/bf16/fp32).

Special token IDs used for lyrics:
- `0`: Pad token
- `1`: Comma token
- `2`: Period token
- `355`: Start token

The `__getitem__` method includes error handling with fallback — if a sample fails to load or is shorter than `min_frames`, it randomly selects another sample.

### Collation

The `custom_collate_fn()` batches variable-length samples:
- **Latents**: Padded to `max_frames` with zeros.
- **Prompts**: Padded to the max prompt length within the batch.
- **LRC tokens**: Already fixed-length (`max_frames`), stacked directly.
- Returns a dict with tensors for `prompt`, `lrc`, `latent`, `start_time`, and their corresponding length tensors.

---

## Utility Functions: `model/utils.py`

| Function | Purpose |
|----------|---------|
| `seed_everything(seed)` | Sets random seeds across Python, NumPy, and PyTorch for reproducibility |
| `exists(v)` | Returns `True` if `v` is not `None` |
| `default(v, d)` | Returns `v` if it exists, otherwise `d` |
| `lens_to_mask(t, length)` | Converts a tensor of lengths to a boolean mask of shape `[b, n]` |
| `mask_from_start_end_indices(seq_len, start, end, max_frames)` | Creates a boolean mask from start/end index pairs |
| `mask_from_frac_lengths(seq_len, frac_lengths, max_frames)` | Creates random span masks given fractional lengths — used in training for infilling |
| `list_str_to_tensor(text)` | UTF-8 byte tokenization of strings |
| `list_str_to_idx(text, vocab_char_map)` | Character/vocabulary-based tokenization |
| `get_tokenizer(dataset_name, tokenizer)` | Factory for tokenizers: `"pinyin"`, `"char"`, `"byte"`, or `"custom"` |
| `convert_char_to_pinyin(text_list)` | Converts Chinese characters to Pinyin using `pypinyin` and `jieba` |
| `repetition_found(text, length, tolerance)` | Detects repetitive patterns in text (data quality filter) |

---

## Configuration Files

### Default Training Config: `config/default.ini`

Parsed by `prefigure` — any value can be overridden from the command line.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exp_name` | `diffrhythm-test` | Experiment name (used for checkpoint directory) |
| `batch_size` | `8` | Batch size per GPU |
| `max_frames` | `3000` | Maximum sequence length in frames |
| `min_frames` | `10` | Minimum sample length (shorter samples are skipped) |
| `num_workers` | `4` | DataLoader worker count |
| `seed` | `42` | Random seed |
| `epochs` | `110` | Total training epochs |
| `learning_rate` | `7.5e-5` | Peak learning rate |
| `num_warmup_updates` | `20` | Warmup steps |
| `save_per_updates` | `5000` | Checkpoint interval (steps) |
| `last_per_steps` | `5000` | "Last" checkpoint interval (steps) |
| `grad_accumulation_steps` | `1` | Gradient accumulation steps |
| `max_grad_norm` | `1.0` | Max gradient norm for clipping |
| `file_path` | `dataset/train.scp` | Path to training data list |
| `audio_drop_prob` | `0.3` | CFG dropout for audio conditioning |
| `cond_drop_prob` | `0.0` | CFG dropout for general conditioning |
| `style_drop_prob` | `0.1` | CFG dropout for style prompt |
| `lrc_drop_prob` | `0.1` | CFG dropout for lyrics |
| `reset_lr` | `0` | Whether to reset LR when resuming |
| `resumable_with_seed` | `666` | Seed for reproducible dataset shuffling on resume |
| `downsample_rate` | `2048` | Audio downsample rate |
| `grad_ckpt` | `0` | Enable gradient checkpointing (0 = off) |
| `sampling_rate` | `44100` | Audio sampling rate in Hz |

### Model Architecture Config: `config/diffrhythm-1b.json`

Defines the DiT transformer architecture:

```json
{
    "model_type": "diffrhythm",
    "model": {
        "dim": 2048,
        "depth": 16,
        "heads": 32,
        "ff_mult": 4,
        "text_dim": 512,
        "conv_layers": 4,
        "mel_dim": 64,
        "text_num_embeds": 363
    }
}
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 2048 | Transformer hidden dimension |
| `depth` | 16 | Number of Llama transformer layers |
| `heads` | 32 | Number of attention heads |
| `ff_mult` | 4 | Feed-forward intermediate size multiplier |
| `text_dim` | 512 | Text embedding dimension |
| `conv_layers` | 4 | Number of ConvNeXtV2 blocks in text encoder |
| `mel_dim` | 64 | Mel spectrogram / latent channels |
| `text_num_embeds` | 363 | Vocabulary size for lyrics tokens |

### Accelerate Config: `config/accelerate_config.yaml`

HuggingFace Accelerate configuration for distributed training:

| Setting | Value | Description |
|---------|-------|-------------|
| `distributed_type` | `MULTI_GPU` | Distributed training strategy |
| `mixed_precision` | `fp16` | Mixed precision mode |
| `num_processes` | `1` | Number of GPUs (adjust for multi-GPU) |
| `num_machines` | `1` | Number of machines |
| `main_process_port` | `12342` | Port for distributed communication |

---

## Launch Script: `scripts/train.sh`

A convenience shell script that launches training via HuggingFace Accelerate:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD

accelerate launch --config-file config/accelerate_config.yaml \
    train/train.py \
    --model-config config/diffrhythm-1b.json \
    --batch-size 6 \
    --max-frames 2048 \
    --min-frames 512 \
    --cond-drop-prob 0.2 \
    --style-drop-prob 0.2 \
    --lrc-drop-prob 0.2 \
    --audio-drop-prob 1.0 \
    --reset-lr 1 \
    --epochs 1000 \
    --resumable-with-seed 666 \
    --grad-accumulation-steps 1 \
    --grad-ckpt 0 \
    --exp-name diffrhythm-test \
    --file-path "dataset/train.scp"
```

Note that command-line arguments override the defaults from `config/default.ini`. The script sets `audio_drop_prob=1.0` (always drop audio conditioning), which is a more aggressive CFG strategy than the default of `0.3`.

---

## How to Launch Training

1. **Prepare your data** — generate the `.scp` file and corresponding `.pt` files for lyrics, latents, and style prompts (see preprocessing documentation).

2. **Configure Accelerate** — edit `config/accelerate_config.yaml` for your hardware setup (number of GPUs, mixed precision, etc.), or run `accelerate config` interactively.

3. **Set up WandB** (optional) — export your `WANDB_API_KEY` environment variable for experiment tracking. If not set, WandB logging is silently disabled.

4. **Launch training**:
   ```bash
   bash scripts/train.sh
   ```

   Or manually:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$PWD
   accelerate launch --config-file config/accelerate_config.yaml \
       train/train.py \
       --model-config config/diffrhythm-1b.json \
       --batch-size 8 \
       --epochs 110
   ```

5. **Resume training** — simply relaunch with the same `--exp-name`. The trainer automatically loads the latest checkpoint from `ckpts/{exp_name}/`. Use `--reset-lr 1` to reset the learning rate schedule on resume.

---

## Training Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│                    Configuration                         │
│  default.ini + CLI args + diffrhythm-1b.json            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Model Construction                      │
│  DiT(dim=2048, depth=16, heads=32, ...)                 │
│  └── wrapped in CFM(drop_probs, max_frames)             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Trainer Setup                          │
│  Accelerator → EMA → Optimizer (AdamW) → Scheduler      │
│  DataLoader(DiffusionDataset) → prepare()               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Training Loop                          │
│                                                          │
│  for epoch in range(epochs):                            │
│    for batch in dataloader:                             │
│      ┌──────────────────────────────────────────┐       │
│      │ 1. Sample noise x0 ~ N(0,I)              │       │
│      │ 2. Sample time t ~ sigmoid(N(0,1))        │       │
│      │ 3. Interpolate φ = (1-t)*x0 + t*x1       │       │
│      │ 4. Target flow = x1 - x0                  │       │
│      │ 5. Random span mask for infilling         │       │
│      │ 6. CFG dropout (audio, text, style)       │       │
│      │ 7. DiT predicts flow from φ               │       │
│      │ 8. Loss = MSE(pred, flow) on masked span  │       │
│      └──────────────────────────────────────────┘       │
│      backward → clip gradients → optimizer step         │
│      scheduler step → EMA update → log to WandB        │
│      save checkpoint periodically                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Dependencies

| Library | Role |
|---------|------|
| PyTorch | Core deep learning framework |
| HuggingFace Accelerate | Distributed training, mixed precision, gradient accumulation |
| HuggingFace Transformers | Llama architecture (decoder layers, rotary embeddings) |
| ema-pytorch | Exponential Moving Average of model weights |
| prefigure | INI + CLI configuration parsing |
| torchdiffeq | ODE solver for inference (not used during training) |
| WandB | Experiment tracking and logging (optional) |
| bitsandbytes | 8-bit AdamW optimizer (optional, memory-efficient) |
