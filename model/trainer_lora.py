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
                    "lora_rank": getattr(args, "lora_rank", None),
                    "lora_alpha": getattr(args, "lora_alpha", None),
                },
            )

        self.model = model

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/lora_default")

        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        self.get_dataloader()
        self.get_scheduler()

        self.model, self.optimizer, self.scheduler, self.train_dataloader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_dataloader
            )
        )

    def get_scheduler(self):
        warmup_steps = self.num_warmup_updates * self.accelerator.num_processes
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
        precision = self.accelerator.state.mixed_precision.replace("no", "fp32")
        dd = DiffusionDataset(
            self.args.file_path,
            self.args.max_frames,
            self.args.min_frames,
            self.args.sampling_rate,
            self.args.downsample_rate,
            precision,
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

            # PEFT adapter (adapter_config.json + adapter_model.safetensors)
            if last:
                adapter_dir = f"{self.checkpoint_path}/lora_adapter_last"
            else:
                adapter_dir = f"{self.checkpoint_path}/lora_adapter_step_{step}"
            dit_peft.save_pretrained(adapter_dir)

            # Training state for resuming (optimizer + scheduler + step)
            training_state = dict(
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if last:
                self.accelerator.save(
                    training_state, f"{self.checkpoint_path}/training_state_last.pt"
                )
            else:
                self.accelerator.save(
                    training_state, f"{self.checkpoint_path}/training_state_{step}.pt"
                )

            print(f"Saved LoRA checkpoint at step {step}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()

        # Prefer "last" checkpoint, otherwise find the latest numbered one
        has_last = os.path.exists(f"{self.checkpoint_path}/lora_adapter_last")
        if has_last:
            adapter_dir = f"{self.checkpoint_path}/lora_adapter_last"
            state_path = f"{self.checkpoint_path}/training_state_last.pt"
        else:
            adapter_dirs = sorted(
                [
                    d
                    for d in os.listdir(self.checkpoint_path)
                    if d.startswith("lora_adapter_step_")
                ],
                key=lambda x: int(x.split("_")[-1]),
            )
            if not adapter_dirs:
                return 0
            adapter_dir = f"{self.checkpoint_path}/{adapter_dirs[-1]}"
            step_num = adapter_dirs[-1].split("_")[-1]
            state_path = f"{self.checkpoint_path}/training_state_{step_num}.pt"

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
                map_location="cpu",
            )
        set_peft_model_state_dict(dit_peft, adapter_weights)

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

            if (
                resumable_with_seed
                and resumable_with_seed > 0
                and epoch == skipped_epoch
            ):
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch + 1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch + 1}/{self.epochs}",
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
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
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
