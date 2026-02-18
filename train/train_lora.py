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

    cfm_kwargs = dict(
        num_channels=model_config["model"]["mel_dim"],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=args.cond_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames,
    )

    # Build a full CFM(DiT) and load pretrained weights into it directly.
    # This avoids manually stripping the "transformer." prefix from checkpoint
    # keys — they already match CFM's state dict structure.
    cfm_pretrained = CFM(
        transformer=DiT(**model_config["model"], max_frames=args.max_frames),
        **cfm_kwargs,
    )

    if args.pretrained_ckpt_path:
        ckpt_path = args.pretrained_ckpt_path
    else:
        ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-1_2",
            filename="cfm_model.pt",
            cache_dir="./pretrained",
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "ema_model_state_dict" in checkpoint:
        state_dict = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cfm_pretrained.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {ckpt_path}")

    # Extract the DiT (now carrying pretrained weights) and apply LoRA.
    # Must happen before re-wrapping in CFM so adapter key paths stay clean.
    dit = cfm_pretrained.transformer
    del cfm_pretrained

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    dit = get_peft_model(dit, lora_config)
    dit.print_trainable_parameters()

    # Wrap the LoRA-adapted DiT in a fresh CFM
    model = CFM(transformer=dit, **cfm_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

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
