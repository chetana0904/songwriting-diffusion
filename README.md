# DiffRhythm -- LoRA Fine-Tuning

Fine-tune the [DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) music generation model on your own data using LoRA adapters.

**Note:** This is experimental, partially vibecoded work.

For the original project documentation (inference, Docker, model versions, citation), see [ORIGINAL_README.md](ORIGINAL_README.md).

## Prepare Your Data

Place your audio files (`.mp3`, `.wav`, `.flac`) and optional `.lrc` lyrics files in a directory, then run:

```bash
python preprocess/preprocess_data.py --input-dir /path/to/songs --output-dir dataset
```

This produces VAE latents, style embeddings, and tokenized lyrics for each track, plus a `train.scp` manifest. Audio without a matching `.lrc` file is treated as instrumental.

Full details on input format, options, and troubleshooting: [preprocess/PREPROCESSING.md](preprocess/PREPROCESSING.md)

## Train

### NVIDIA GPU

```bash
bash scripts/train_lora.sh
```

Or run directly with custom arguments:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD

accelerate launch --config-file config/accelerate_config.yaml \
    train/train_lora.py \
    --file-path "dataset/train.scp" \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 2e-4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --exp-name my-lora-run
```

### Apple Silicon (MPS)

Use the MPS accelerate config and set the fallback flag:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD

PYTORCH_ENABLE_MPS_FALLBACK=1 accelerate launch \
    --config-file config/accelerate_config_mps.yaml \
    train/train_lora.py \
    --file-path "dataset/train.scp" \
    --batch-size 1 \
    --epochs 10 \
    --exp-name my-lora-run
```

### Key Parameters

All parameters can be set in `config/lora_default.ini` or overridden on the command line.

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `lora_rank` | 16 | Rank of LoRA adapter matrices. Higher = more capacity, more memory. |
| `lora_alpha` | 32 | LoRA scaling factor. Effective LR scales with `alpha / rank`. |
| `lora_dropout` | 0.05 | Dropout on LoRA layers. |
| `learning_rate` | 2e-4 | Higher than full training because adapters start at zero. |
| `epochs` | 10 | LoRA converges faster than full training. |
| `batch_size` | 4 | Reduce to 1 if you run out of memory. |
| `pretrained_ckpt_path` | (empty) | Path to base checkpoint. Auto-downloads from HuggingFace if empty. |

The pretrained DiffRhythm-1.2 checkpoint is downloaded automatically on the first run (~2 GB).

## Checkpoints

Training saves adapter checkpoints to `ckpts/{exp_name}/`:

```
ckpts/my-lora-run/
  lora_adapter_last/
    adapter_config.json          # LoRA config (needed to reload)
    adapter_model.safetensors    # Adapter weights (~46 MB)
  training_state_last.pt         # Optimizer + scheduler state (for resuming)
```

Compare this to a full model checkpoint at ~4-16 GB. Training automatically resumes from the latest checkpoint if one exists.

## Inference with a LoRA Adapter

### Option A: Load adapter (keeps it separate)

```python
from peft import PeftModel
from model import DiT, CFM

# Build base model and load pretrained weights
cfm = CFM(transformer=DiT(**model_config["model"], max_frames=2048), ...)
# ... load pretrained checkpoint into cfm ...

# Apply LoRA adapter
cfm.transformer = PeftModel.from_pretrained(
    cfm.transformer, "ckpts/my-lora-run/lora_adapter_last"
)
```

### Option B: Merge into base weights (zero overhead at inference)

```python
cfm.transformer = cfm.transformer.merge_and_unload()
```

After merging, the model behaves as a regular DiT with no PEFT dependency.

## Project Structure

```
train/train_lora.py          Entry point: loads base model, applies LoRA, launches training
model/trainer_lora.py        Trainer: filtered optimizer, cosine schedule, PEFT checkpoints
config/lora_default.ini      Default hyperparameters
config/accelerate_config.yaml        Accelerate config (NVIDIA)
config/accelerate_config_mps.yaml    Accelerate config (Apple Silicon)
scripts/train_lora.sh        Launch script
preprocess/preprocess_data.py        Data preprocessing pipeline
preprocess/PREPROCESSING.md          Preprocessing documentation
```

## License

Apache 2.0. See [ORIGINAL_README.md](ORIGINAL_README.md) for the full license text and disclaimer.
