from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultralytics.utils.loss import v8DetectionLoss

from mdra.data.paired_dataset import PairedM3FDDataset, paired_collate_fn
from mdra.data.edge_targets import build_dra_supervision, masked_l1_loss
from mdra.engine.metrics import evaluate_detector
from mdra.models.baselines import build_b_model, load_b_checkpoint_model
from mdra.utils.env_utils import collect_env_info, format_env_report
from mdra.utils.io_utils import save_json, save_yaml, write_text
from mdra.utils.path_utils import safe_mkdir, unique_experiment_dir


RESULT_COLUMNS = [
    "epoch",
    "train_box_loss",
    "train_cls_loss",
    "train_dfl_loss",
    "train_dra_loss",
    "precision",
    "recall",
    "mAP50",
    "mAP75",
    "mAP50_95",
    "lr",
    "epoch_seconds",
    "gpu_peak_memory_gib",
]


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(value: str | int) -> torch.device:
    text = str(value).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text.startswith("cuda:"):
        index = int(text.split(":", 1)[1])
    else:
        index = int(text)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but torch.cuda.is_available() is False")
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is outside available range")
    return torch.device(f"cuda:{index}")


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_b_dataloaders(
    config: dict[str, Any],
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    split_dir = Path(config["split_dir"])
    common = {
        "data_root": config["data_root"],
        "input_mode": str(config.get("input_mode", config["variant"])),
        "imgsz": int(config["imgsz"]),
        "vis_dir": config.get("vis_dir"),
        "ir_dir": config.get("ir_dir"),
        "label_dir": config.get("label_dir"),
    }
    train_dataset = PairedM3FDDataset(
        split_file=split_dir / "train.txt",
        augment=True,
        hflip_prob=float(config.get("hflip_prob", 0.5)),
        max_samples=max_train_samples,
        **common,
    )
    val_dataset = PairedM3FDDataset(
        split_file=split_dir / "val.txt",
        augment=False,
        hflip_prob=0.0,
        max_samples=max_val_samples,
        **common,
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    workers = int(config["workers"])
    batch = int(config["batch"])
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": str(config.get("device", "cpu")) != "cpu",
        "collate_fn": paired_collate_fn,
        "worker_init_fn": _worker_seed,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def build_b_eval_dataloader(
    config: dict[str, Any],
    *,
    split_name: str = "val",
    max_samples: int | None = None,
) -> DataLoader:
    """Build a deterministic val/test loader without changing the training loader contract."""
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'")
    split_file = Path(config["split_dir"]) / f"{split_name}.txt"
    dataset = PairedM3FDDataset(
        data_root=config["data_root"],
        split_file=split_file,
        input_mode=str(config.get("input_mode", config["variant"])),
        imgsz=int(config["imgsz"]),
        vis_dir=config.get("vis_dir"),
        ir_dir=config.get("ir_dir"),
        label_dir=config.get("label_dir"),
        augment=False,
        hflip_prob=0.0,
        max_samples=max_samples,
    )
    workers = int(config["workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch"]),
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=str(config.get("device", "cpu")) != "cpu",
        collate_fn=paired_collate_fn,
        worker_init_fn=_worker_seed,
        persistent_workers=workers > 0,
    )


def _git_version(project_root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _optimizer(model: torch.nn.Module, config: dict[str, Any]):
    name = str(config["optimizer"]).lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=float(config["lr0"]), weight_decay=float(config["weight_decay"]))
    return SGD(
        model.parameters(),
        lr=float(config["lr0"]),
        momentum=float(config.get("momentum", 0.937)),
        weight_decay=float(config["weight_decay"]),
        nesterov=True,
    )


def _scheduler(optimizer, config: dict[str, Any]) -> LambdaLR:
    epochs = max(int(config["epochs"]), 1)
    lrf = float(config["lrf"])

    def cosine(epoch: int) -> float:
        return lrf + (1.0 - lrf) * (1.0 + math.cos(math.pi * epoch / epochs)) / 2.0

    return LambdaLR(optimizer, lr_lambda=cosine)


def _move_loss_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "img": batch["img"].to(device, non_blocking=True),
        "cls": batch["cls"].to(device, non_blocking=True),
        "bboxes": batch["bboxes"].to(device, non_blocking=True),
        "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
    }


class BExperimentTrainer:
    """Unified B1-B5 trainer with explicit MDRA checkpoints and resume semantics."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        project_root: str | Path,
        output_root: str | Path,
        resume: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.resume_path = Path(resume).expanduser().resolve() if resume else None
        seed_everything(int(config["seed"]), bool(config.get("deterministic", True)))

        if self.resume_path:
            model, checkpoint = load_b_checkpoint_model(self.resume_path)
            self.config = dict(checkpoint["config"])
            self.run_dir = Path(checkpoint["run_dir"]).resolve()
            self.start_epoch = int(checkpoint["epoch"]) + 1
            self.best_fitness = float(checkpoint.get("best_fitness", -1.0))
            self.no_improve_epochs = int(checkpoint.get("no_improve_epochs", 0))
            self.pretrained_report = dict(checkpoint.get("pretrained_report", {}))
            self._resume_checkpoint = checkpoint
        else:
            self.config = dict(config)
            self.run_dir = unique_experiment_dir(
                self.output_root / "b_experiments", str(self.config["experiment_id"])
            )
            self.start_epoch = 0
            self.best_fitness = -1.0
            self.no_improve_epochs = 0
            model, self.pretrained_report = build_b_model(
                str(self.config["variant"]),
                nc=int(self.config["nc"]),
                class_names=list(self.config["class_names"]),
                pretrained=self.config.get("pretrained"),
                loss_gains=self.config,
            )
            self._resume_checkpoint = None

        self.device = resolve_device(self.config["device"])
        self.amp_enabled = bool(self.config.get("amp", True)) and self.device.type == "cuda"
        self.model = model.to(self.device)
        self.criterion = v8DetectionLoss(self.model)
        self.optimizer = _optimizer(self.model, self.config)
        self.scheduler = _scheduler(self.optimizer, self.config)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        if self._resume_checkpoint:
            self.optimizer.load_state_dict(self._resume_checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(self._resume_checkpoint["scheduler_state"])
            scaler_state = self._resume_checkpoint.get("scaler_state")
            if scaler_state:
                self.scaler.load_state_dict(scaler_state)

        self.train_loader, self.val_loader = build_b_dataloaders(self.config)
        self.accumulate = max(1, math.ceil(int(self.config["effective_batch"]) / int(self.config["batch"])))
        self.weights_dir = safe_mkdir(self.run_dir / "weights")
        self.results_path = self.run_dir / "results.csv"
        self._prepare_run_files()

    def _prepare_run_files(self) -> None:
        if self.resume_path:
            resume_dir = unique_experiment_dir(self.output_root / "resume_logs", self.run_dir.name + "_resume")
            save_json(
                {
                    "status": "requested",
                    "checkpoint": str(self.resume_path),
                    "run_dir": str(self.run_dir),
                    "start_epoch": self.start_epoch,
                    "requested_at": time.time(),
                },
                resume_dir / "request.json",
            )
            self.resume_log_dir = resume_dir
            return
        self.resume_log_dir = None
        environment = collect_env_info()
        save_json(environment, self.run_dir / "environment.json")
        write_text(self.run_dir / "environment.txt", format_env_report(environment) + "\n")
        save_yaml(self.config, self.run_dir / "resolved_config.yaml")
        save_json(self.pretrained_report, self.run_dir / "pretrained_report.json")
        save_json(
            {
                "variant": self.config["variant"],
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "strides": [float(value) for value in self.model.model[-1].stride.cpu().tolist()],
                "code_version": _git_version(self.project_root),
            },
            self.run_dir / "model_summary.json",
        )
        with self.results_path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()

    def _checkpoint(self, epoch: int) -> dict[str, Any]:
        return {
            "format": "mdra_b_experiment_v1",
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "no_improve_epochs": self.no_improve_epochs,
            "run_dir": str(self.run_dir),
            "config": self.config,
            "pretrained_report": self.pretrained_report,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }

    def _inference_checkpoint(self, epoch: int) -> dict[str, Any]:
        """Create a detector-only checkpoint with DRA parameters and configuration removed."""
        inference_config = dict(self.config)
        inference_config["source_training_variant"] = self.config["variant"]
        inference_config["variant"] = "early_fusion_p2"
        inference_config["input_mode"] = "early_fusion_p2"
        inference_model, _ = build_b_model(
            "early_fusion_p2",
            nc=int(inference_config["nc"]),
            class_names=list(inference_config["class_names"]),
            pretrained=None,
            loss_gains=inference_config,
        )
        source_state = self.model.state_dict()
        inference_state = inference_model.state_dict()
        shared = {
            key: value.detach().cpu()
            for key, value in source_state.items()
            if key in inference_state and value.shape == inference_state[key].shape
        }
        missing = sorted(set(inference_state) - set(shared))
        if missing:
            raise RuntimeError(f"cannot strip DRA checkpoint; missing detector tensors: {missing[:10]}")
        inference_model.load_state_dict(shared, strict=True)
        return {
            "format": "mdra_detector_inference_v1",
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "run_dir": str(self.run_dir),
            "config": inference_config,
            "source_training_variant": self.config["variant"],
            "dra_head_removed": True,
            "model_state": inference_model.state_dict(),
        }

    def _save_checkpoint(self, epoch: int, improved: bool) -> None:
        checkpoint = self._checkpoint(epoch)
        torch.save(checkpoint, self.weights_dir / "last.pt")
        if improved:
            torch.save(checkpoint, self.weights_dir / "best.pt")
            if getattr(self.model, "dra_head", None) is not None:
                torch.save(self._inference_checkpoint(epoch), self.weights_dir / "best_inference.pt")

    def _apply_warmup(self, epoch: int, step: int) -> None:
        """Linearly warm the base learning rate during the configured opening epochs."""
        warmup_epochs = float(self.config.get("warmup_epochs", 0.0))
        warmup_steps = int(round(warmup_epochs * len(self.train_loader)))
        global_step = epoch * len(self.train_loader) + step
        if warmup_steps <= 0 or global_step >= warmup_steps:
            return
        progress = (global_step + 1) / warmup_steps
        factor = 0.1 + 0.9 * progress
        base_lr = float(self.config["lr0"])
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * factor

    def _train_epoch(self, epoch: int) -> tuple[dict[str, float], float]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        sums = torch.zeros(4, dtype=torch.float64)
        batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{self.config['epochs']}", leave=False)
        for step, batch in enumerate(progress):
            self._apply_warmup(epoch, step)
            loss_batch = _move_loss_batch(batch, self.device)
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                predictions = self.model(loss_batch["img"])
                dra_loss = loss_batch["img"].new_zeros(())
                if isinstance(predictions, dict):
                    det_predictions = predictions["det_preds"]
                    dra_prediction = predictions["dra_pred"]
                    supervision = build_dra_supervision(
                        loss_batch["img"],
                        {**batch, "bboxes": loss_batch["bboxes"], "batch_idx": loss_batch["batch_idx"]},
                        output_size=tuple(dra_prediction.shape[-2:]),
                        mode=str(getattr(self.model, "dra_mode")),
                        bbox_expansion=float(self.config["dra_bbox_expansion"]),
                        edge_fusion=str(self.config["dra_edge_fusion"]),
                        edge_alpha=float(self.config["dra_edge_alpha"]),
                    )
                    dra_loss = masked_l1_loss(dra_prediction, supervision.target, supervision.loss_mask)
                else:
                    det_predictions = predictions
                loss, loss_items = self.criterion(det_predictions, loss_batch)
                loss = loss + float(self.config.get("dra_lambda", 0.0)) * dra_loss * loss_batch["img"].shape[0]
                scaled_loss = loss / self.accumulate
            self.scaler.scale(scaled_loss).backward()
            should_step = (step + 1) % self.accumulate == 0 or step + 1 == len(self.train_loader)
            if should_step:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            sums[:3] += loss_items.detach().cpu().double()
            sums[3] += float(dra_loss.detach())
            batches += 1
            means = sums / batches
            progress.set_postfix(
                box=f"{means[0]:.3f}", cls=f"{means[1]:.3f}",
                dfl=f"{means[2]:.3f}", dra=f"{means[3]:.3f}"
            )
        peak = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3) if self.device.type == "cuda" else 0.0
        )
        means = sums / max(batches, 1)
        return {
            "box": float(means[0]), "cls": float(means[1]), "dfl": float(means[2]),
            "dra": float(means[3]),
        }, peak

    def _append_result(
        self,
        epoch: int,
        losses: dict[str, float],
        metrics: dict[str, Any],
        epoch_seconds: float,
        peak_memory: float,
    ) -> None:
        row = {
            "epoch": epoch + 1,
            "train_box_loss": losses["box"],
            "train_cls_loss": losses["cls"],
            "train_dfl_loss": losses["dfl"],
            "train_dra_loss": losses["dra"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "mAP50": metrics["mAP50"],
            "mAP75": metrics["mAP75"],
            "mAP50_95": metrics["mAP50_95"],
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "gpu_peak_memory_gib": peak_memory,
        }
        with self.results_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writerow(row)
        save_json(metrics, self.run_dir / "last_validation_metrics.json", overwrite=True)

    def train(self) -> dict[str, Any]:
        total_epochs = int(self.config["epochs"])
        patience = int(self.config.get("patience", 0))
        if self.start_epoch >= total_epochs:
            raise RuntimeError(
                f"checkpoint epoch {self.start_epoch} already reached configured total epochs {total_epochs}"
            )
        start_time = time.time()
        completed_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()
            losses, peak_memory = self._train_epoch(epoch)
            metrics = evaluate_detector(
                self.model,
                self.val_loader,
                device=self.device,
                class_names=list(self.config["class_names"]),
                conf_thres=float(self.config["conf_thres"]),
                iou_thres=float(self.config["iou_thres"]),
                amp=self.amp_enabled,
            )
            fitness = float(metrics["mAP50_95"])
            improved = fitness > self.best_fitness
            if improved:
                self.best_fitness = fitness
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1
            self._append_result(epoch, losses, metrics, time.time() - epoch_start, peak_memory)
            self.scheduler.step()
            self._save_checkpoint(epoch, improved)
            completed_epoch = epoch
            print(
                f"epoch={epoch + 1} P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"mAP50={metrics['mAP50']:.4f} mAP75={metrics['mAP75']:.4f} "
                f"mAP50-95={metrics['mAP50_95']:.4f}"
            )
            if patience > 0 and self.no_improve_epochs >= patience:
                print(f"early stopping after {self.no_improve_epochs} epochs without improvement")
                break

        best_path = self.weights_dir / "best.pt"
        last_path = self.weights_dir / "last.pt"
        if best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(best_checkpoint["model_state"], strict=True)
        final_metrics = evaluate_detector(
            self.model,
            self.val_loader,
            device=self.device,
            class_names=list(self.config["class_names"]),
            conf_thres=float(self.config["conf_thres"]),
            iou_thres=float(self.config["iou_thres"]),
            amp=self.amp_enabled,
            plot=True,
            save_dir=self.run_dir / "validation_plots",
        )
        summary = {
            "status": "completed",
            "variant": self.config["variant"],
            "run_dir": str(self.run_dir),
            "start_epoch": self.start_epoch + 1,
            "completed_epoch": completed_epoch + 1,
            "best_fitness": self.best_fitness,
            "final_metrics": final_metrics,
            "best_checkpoint": str(best_path) if best_path.is_file() else None,
            "last_checkpoint": str(last_path),
            "results_csv": str(self.results_path),
            "training_seconds": time.time() - start_time,
            "resume_from": str(self.resume_path) if self.resume_path else None,
        }
        save_json(summary, self.run_dir / "run_summary.json", overwrite=(self.run_dir / "run_summary.json").exists())
        if self.resume_log_dir:
            save_json(summary, self.resume_log_dir / "summary.json")
        return summary


def dry_run_b_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run one real paired batch through forward, loss, and backward without training epochs."""
    dry = dict(config)
    dry.update({"workers": 0, "batch": min(int(config["batch"]), 2), "imgsz": 64})
    device = resolve_device(dry["device"])
    train_loader, _ = build_b_dataloaders(dry, max_train_samples=2, max_val_samples=2)
    model, report = build_b_model(
        str(dry["variant"]),
        nc=int(dry["nc"]),
        class_names=list(dry["class_names"]),
        pretrained=dry.get("pretrained"),
        loss_gains=dry,
    )
    model = model.to(device).train()
    batch = next(iter(train_loader))
    loss_batch = _move_loss_batch(batch, device)
    with torch.cuda.amp.autocast(enabled=bool(dry.get("amp", True)) and device.type == "cuda"):
        output = model(loss_batch["img"])
        criterion = v8DetectionLoss(model)
        dra_loss = loss_batch["img"].new_zeros(())
        if isinstance(output, dict):
            det_output = output["det_preds"]
            dra_prediction = output["dra_pred"]
            supervision = build_dra_supervision(
                loss_batch["img"],
                {**batch, "bboxes": loss_batch["bboxes"], "batch_idx": loss_batch["batch_idx"]},
                output_size=tuple(dra_prediction.shape[-2:]),
                mode=str(getattr(model, "dra_mode")),
                bbox_expansion=float(dry["dra_bbox_expansion"]),
                edge_fusion=str(dry["dra_edge_fusion"]),
                edge_alpha=float(dry["dra_edge_alpha"]),
            )
            dra_loss = masked_l1_loss(dra_prediction, supervision.target, supervision.loss_mask)
        else:
            det_output = output
        loss, items = criterion(det_output, loss_batch)
        loss = loss + float(dry.get("dra_lambda", 0.0)) * dra_loss * loss_batch["img"].shape[0]
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run loss is not finite")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("dry-run backward did not produce finite gradients")
    return {
        "status": "passed",
        "variant": dry["variant"],
        "device": str(device),
        "input_shape": list(loss_batch["img"].shape),
        "feature_levels": len(det_output),
        "feature_shapes": [list(tensor.shape) for tensor in det_output],
        "loss": float(loss.detach()),
        "dra_loss": float(dra_loss.detach()),
        "loss_items": [float(value) for value in items.detach()],
        "backward": "passed",
        "gpu_peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
        "model": report,
    }
