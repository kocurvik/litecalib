"""Train a LiteCalib network (the FoV-field method, or the radial / division ablations).

    python train/train.py --config configs/ms_aux_field_wsd24_k5_vitl_opg.yaml
    python train/train.py --config configs/ms_aux_field_wsd24_k5_vitl_opg.yaml --submitit   # on SLURM
"""

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = str(Path(__file__).resolve().parent)
for _p in (str(ROOT), THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data
import dense as dense_lib
import slurm
from litecalib import fit as fit_lib
from litecalib import geometry
from litecalib import model as model_lib
from loss import division_param_loss, profile_loss

RAD2DEG = 180 / math.pi
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_{0}/dinov2_{0}_pretrain.pth"


class Dist:
    """The distributed environment, or a stand-in for single-process runs."""

    def __init__(self, cuda: bool | None = None, backend: str | None = None):
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.enabled = self.world_size > 1
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        if self.enabled and not dist.is_initialized():
            if self.cuda:
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend=backend or ("nccl" if self.cuda else "gloo"),
                rank=self.rank,
                world_size=self.world_size,
            )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def print(self, *a, **kw) -> None:
        if self.is_main:
            print(*a, **kw, flush=True)

    def shutdown(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def _fetch(url: str, ckpt: Path, flag: str) -> Path:
    """Download `url` to `ckpt` unless it is already there."""
    if ckpt.exists():
        return ckpt
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {ckpt}", flush=True)
    part = ckpt.with_suffix(ckpt.suffix + ".part")
    try:
        import requests

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        part.replace(ckpt)
    except Exception as e:
        raise RuntimeError(
            f"could not download {url} ({e}). Fetch it on a machine with internet and pass "
            f"{flag} <path> (expected at {ckpt})."
        ) from e
    return ckpt


def fmt_duration(seconds: float) -> str:
    """A compact duration for the end of a log line: `7h20m`, `12m30s`, `45s`."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


def fmt_finish(seconds: float) -> str:
    """When `seconds` from now falls, in local time, with the weekday."""
    return time.strftime("%a %H:%M", time.localtime(time.time() + max(0.0, seconds)))


def dinov2_checkpoint(backbone: str, path: str | None = None) -> Path:
    """Locate (and, if need be, download) DINOv2's released weights for `backbone`."""
    if path:
        return Path(path).expanduser()
    home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    name = f"dinov2_{backbone}_pretrain.pth"
    return _fetch(DINOV2_URL.format(backbone), home / "hub" / "checkpoints" / name, "--dinov2_ckpt")


def build_model(args, device) -> tuple[torch.nn.Module, dict]:
    """The network `args` describe, initialized from DINOv2 (``--init dinov2``) or a checkpoint."""
    net = model_lib.RadialProfileNet(
        backbone=args.backbone,
        k=args.K,
        placement=args.placement,
        layers=args.decoder_layers,
        q_dim=args.q_dim,
        num_heads=args.attn_heads,
        init_vfov_deg=args.init_vfov,
        profile=args.profile,
        dense_head=args.dense_loss_weight > 0,
        dense_dim=args.dense_dim,
        dense_repr=args.dense_repr,
        field_repr=args.field_repr,
    )
    info = {"init": args.init or f"{args.backbone}-random"}
    if args.init == "dinov2":
        info.update(net.load_dinov2_backbone(dinov2_checkpoint(args.backbone, args.dinov2_ckpt)))
    elif args.init:
        state = torch.load(Path(args.init).expanduser(), map_location="cpu", weights_only=False)
        info["missing"] = list(net.load_state_dict(state.get("model", state), strict=False).missing_keys)
    return net.to(device), info


def division_terms(args, pred, batch, loss, metrics):
    """The direct-regression ablation's loss: the profile loss weighted, plus the parameter term."""
    loss = args.div_profile_weight * loss
    if args.div_param_weight > 0:
        if "image" in batch:
            h, w = batch["image"].shape[-2:]
        else:
            h, w = batch["target_size"][0].tolist()
        log_f_gt, kappa_gt = fit_lib.division_labels(
            batch["theta"], int(h), int(w), args.placement, model_lib.REGRESSION_CAMS[args.profile]
        )
        p_loss, p_metrics = division_param_loss(
            pred, log_f_gt, kappa_gt, batch["valid"], args.div_kappa_weight
        )
        loss = loss + args.div_param_weight * p_loss
        metrics.update(p_metrics)
    return loss


def build_loaders(args, dist_env: Dist):
    """Train loader (all ranks) and per-dataset val loaders (main rank only)."""
    common = dict(
        k=args.K,
        placement=args.placement,
        check_invariants=not args.no_check_invariants,
        resolution=args.resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        edge_divisible_by=args.edge_divisible_by,
        augmentations=args.augmentations,
        seed=args.seed,
        profile="field" if args.profile == "field" else "radial",
        edit_prob=args.edit_prob,
        edit_crop=args.edit_crop,
        knot_targets=not getattr(args, "dense_targets_only", False),
        field_repr=args.field_repr,
    )
    common["augment_after_resize"] = not args.augment_before_resize
    mismatches = [] if args.gpu_pipeline else data.augmentation_mismatches(args.augmentations)
    if mismatches and not args.allow_augmentation_mismatch:
        raise SystemExit(
            "the installed albumentations silently ignores AnyCalib's augmentation settings:\n  "
            + "\n  ".join(mismatches)
            + "\n\nFix the environment (pip install 'albumentations==1.4.*'), or pass "
            "--allow_augmentation_mismatch to train anyway, or --augmentations identity."
        )
    members = [(g, part) for g, d in enumerate(args.data) for part in d.split("+")]
    dirs = [data.resolve_dataset_dir(part, args.paths) for _, part in members]
    train_sets = [
        data.build_dataset(data.stage_store(d) if d.name in args.no_preload else d, "train",
                           flip_prob=args.flip_prob, limit=args.limit_train,
                           jpeg_bytes=args.gpu_pipeline, preload=d.name not in args.no_preload, **common)
        for d in dirs
    ]
    train_set = train_sets[0] if len(train_sets) == 1 else data.MergedDataset(
        train_sets, None, groups=[g for g, _ in members])
    train_loader = data.make_loader(
        train_set, args.batch_size, "train", args.workers, args.prefetch,
        distributed=dist_env.enabled, samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
    )

    val_loaders = {}
    if dist_env.is_main and not args.no_val:
        for d in dirs:
            val = data.build_dataset(
                d, "val", flip_prob=0.0, limit=args.limit_val, to_closest_train_size=True,
                preload=True, **common,
            )
            val_loaders[val.name] = data.make_loader(
                val, args.val_batch_size, "val", args.workers, args.prefetch
            )
    return train_set, train_loader, val_loaders


def branch_path(ckpt: Path, epoch: int) -> Path:
    """Where `--branch_epochs` puts the full state of `epoch`: `<stem>_ep<epoch>_full.pt` beside `ckpt`."""
    return ckpt.with_name(f"{ckpt.stem}_ep{epoch}_full{ckpt.suffix}")


BRANCH_SHARED = ("backbone", "data", "seed", "lr", "backbone_lr", "weight_decay",
                 "batch_size", "accum", "samples_per_dataset", "warmup_steps",
                 "warmup_start_factor", "lr_schedule", "profile", "K", "field_repr")


def keep_path(ckpt: Path, epoch: int) -> Path:
    """Where `--keep_epochs` puts the snapshot of `epoch`: `<stem>_ep<epoch>.pt` beside `ckpt`."""
    return ckpt.with_name(f"{ckpt.stem}_ep{epoch}{ckpt.suffix}")


def best_path(ckpt: Path) -> Path:
    """Where the best-validation weights go: `<stem>_best.pt` beside `ckpt`."""
    return ckpt.with_name(f"{ckpt.stem}_best{ckpt.suffix}")


def val_score(val_stats: dict) -> float | None:
    """The knot L1 (deg) pooled over every val set, weighted by its sample count (None without validation)."""
    n = sum(v["n"] for v in val_stats.values())
    return sum(v["l1_deg"] * v["n"] for v in val_stats.values()) / n if n else None


def best_epoch(history: list[dict]) -> dict | None:
    """The epoch with the lowest `val_score` in a run's history (the first, on a tie)."""
    scored = [(s, h["train"]["epoch"]) for h in history
              if (s := val_score(h.get("val") or {})) is not None]
    if not scored:
        return None
    score, epoch = min(scored, key=lambda t: t[0])
    return {"epoch": epoch, "val_l1_deg": score}


def lr_lambda(args, total_steps: int | None = None):
    """Warmup, then ``--lr_schedule``: ``step`` (multiply by `lr_gamma` at each milestone) or ``wsd``
    (warmup-stable-decay: the peak lr until the last `decay_epochs`, then a ``1 - sqrt`` cool-down to 0)."""
    if args.lr_schedule == "wsd":
        assert total_steps is not None, "wsd needs the run's length in steps"
        decay_steps = round(args.decay_epochs * total_steps / args.epochs)
        decay_start = total_steps - decay_steps

    def fn(step: int) -> float:
        warm = min(1.0, (step + 1) / max(1, args.warmup_steps))
        warm = args.warmup_start_factor + (1 - args.warmup_start_factor) * warm
        if args.lr_schedule == "wsd":
            if step < decay_start or not decay_steps:
                return warm
            return warm * max(0.0, 1.0 - math.sqrt((step - decay_start) / decay_steps))
        decay = args.lr_gamma ** sum(step >= m for m in args.lr_milestones)
        return warm * decay

    return fn


@torch.no_grad()
def validate(net, loaders: dict, args, device, amp_dtype) -> dict:
    """Knot L1 and FoV error of `net` on each validation loader."""
    net.eval()
    out = {}
    for name, loader in loaders.items():
        totals = {"l1_deg": 0.0, "fov_err": [], "n": 0, "invalid": 0}
        for batch in loader:
            batch = data.to_device(batch, device)
            omega = data.quadrature_weights_for(batch, args.K, args.placement, device, args.profile)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = net(batch["image"])
            pred = {k: v.float() for k, v in pred.items() if isinstance(v, torch.Tensor)}
            keep = batch["valid"]
            n = int(keep.sum())
            totals["invalid"] += int((~keep).sum())
            if n == 0:
                continue
            err = (pred["theta"] - batch["theta"])[keep].abs()
            totals["l1_deg"] += float(RAD2DEG * (omega * err).sum(dim=-1).sum())
            totals["fov_err"] += (RAD2DEG * 2 * (pred["Theta"] - batch["Theta"])[keep].abs()).tolist()
            totals["n"] += n
        n = max(1, totals["n"])
        fov = torch.tensor(totals["fov_err"]) if totals["fov_err"] else torch.zeros(1)
        out[name] = {
            "n": totals["n"],
            "invalid": totals["invalid"],
            "l1_deg": totals["l1_deg"] / n,
            "fov_err_deg_median": float(fov.median()),
            "fov_err_deg_mean": float(fov.mean()),
        }
    net.train()
    return out


def run(args) -> None:
    """One training run (one DDP rank)."""
    args.device = slurm.device_or_cpu(args.device)
    dist_env = Dist(cuda=args.device == "cuda")
    torch.manual_seed(args.seed + dist_env.rank)
    device = torch.device(f"cuda:{dist_env.local_rank}" if args.device == "cuda" else args.device)
    amp_dtype = None
    if args.amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    train_set, train_loader, val_loaders = build_loaders(args, dist_env)
    pipeline = None
    if args.gpu_pipeline:
        from gpu_pipeline import GpuPipeline

        pipeline = GpuPipeline(augment=args.augmentations != "identity", seed=args.seed + dist_env.rank,
                               device=device)
    net, init_info = build_model(args, device)
    regression = args.profile in model_lib.REGRESSION_CAMS
    dist_env.print(
        f"model: {args.backbone}, "
        + (f"FoV field, {args.K} x {args.K} points ({geometry.field_num_values(args.K)} outputs)"
           if args.profile == "field"
           else f"division:1 (f, kappa), rendered at K={args.K} knots" if regression
           else f"radial profile, K={args.K}")
        + f", {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters"
        + f"\ninit: {init_info}"
        f"\ntrain: {data.summarize(train_set)}, camera models {data.cam_id_counts(train_set)}"
        + (f"\nepoch: {args.samples_per_dataset} views drawn afresh from each of "
           f"{len(args.data)} training shares ({', '.join(args.data)}) = {len(train_loader)} steps"
           if args.samples_per_dataset else "")
        + f"\nval: {', '.join(f'{k} ({len(v.dataset)})' for k, v in val_loaders.items()) or 'disabled'}"
        f"\npipeline: {'gpu' if pipeline is not None else 'cpu'}"
        f"\ndevice: {device}, world size {dist_env.world_size}, amp {amp_dtype}, "
        f"effective batch {args.batch_size * dist_env.world_size * max(1, args.accum)}"
    )

    if dist_env.enabled:
        net = DistributedDataParallel(net, device_ids=[dist_env.local_rank] if device.type == "cuda" else None)
    core = net.module if isinstance(net, DistributedDataParallel) else net

    optimizer = torch.optim.AdamW(core.param_groups(args.lr, args.backbone_lr, args.weight_decay))
    accum = max(1, args.accum)
    total_steps = len(train_loader) // accum * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args, total_steps))

    start_epoch, step, history = 1, 0, []
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    resuming = bool(args.resume and ckpt_path and ckpt_path.exists())
    if args.branch_from and not resuming:
        state = torch.load(Path(args.branch_from), map_location="cpu", weights_only=False)
        assert "optimizer" in state, f"{args.branch_from} has no optimizer state to branch from"
        theirs = state["args"]
        differ = {k: (theirs.get(k), getattr(args, k)) for k in BRANCH_SHARED
                  if theirs.get(k) != getattr(args, k)}
        assert not differ, f"--branch_from {args.branch_from} was trained differently: {differ}"
        decay_start = total_steps - round(args.decay_epochs * total_steps / args.epochs)
        assert state["step"] <= decay_start, (
            f"--branch_from at step {state['step']}, after this run's decay starts ({decay_start})")
        core.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, step, history = state["epoch"] + 1, state["step"], state["history"]
        dist_env.print(f"branched from {args.branch_from} at epoch {state['epoch']} (step {step}); "
                       f"cool-down over the last {args.decay_epochs} of {args.epochs} epochs")
    if resuming:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        core.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        else:
            for _ in range(state["step"]):
                scheduler.step()
        start_epoch, step, history = state["epoch"] + 1, state["step"], state["history"]
        dist_env.print(f"resumed from {ckpt_path} at epoch {state['epoch']} (step {step})")

    out_path = Path(args.out) if args.out else None
    steps_per_epoch = len(train_loader)

    def save(epoch: int) -> None:
        if not (dist_env.is_main and ckpt_path):
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": core.state_dict(), "epoch": epoch, "step": step, "history": history,
                 "args": vars(args), "init": init_info}

        def write(path: Path, st: dict) -> None:
            tmp = path.with_name(path.name + ".partial")
            torch.save(st, tmp)
            tmp.replace(path)

        if epoch in args.keep_epochs:
            write(keep_path(ckpt_path, epoch), state)
        best = best_epoch(history) if args.keep_best else None
        if best is not None and best["epoch"] == epoch:
            write(best_path(ckpt_path), {**state, "best": best})
            dist_env.print(f"  new best val L1 {best['val_l1_deg']:.3f} deg -> {best_path(ckpt_path)}")
        if not args.no_optimizer_state:
            state["optimizer"] = optimizer.state_dict()
            state["scheduler"] = scheduler.state_dict()
        if epoch in args.branch_epochs:
            write(branch_path(ckpt_path, epoch), state)
        write(ckpt_path, state)

    def write_history() -> None:
        if dist_env.is_main and out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"args": vars(args), "init": init_info, "history": history,
                                            "best": best_epoch(history)}, indent=2, default=str),
                                encoding="utf-8")

    epoch_walls: list[float] = []
    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        data.set_epoch(train_loader, epoch)
        sums = {"loss": 0.0, "l1_deg": 0.0, "fov_err_deg": 0.0}
        if args.dense_loss_weight > 0:
            sums["dense_deg"] = 0.0
        if regression and args.div_param_weight > 0:
            sums.update(f_err_pct=0.0, kappa_err=0.0)
        t0, n_batches, invalid = time.time(), 0, 0
        bar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}", mininterval=30,
                   disable=not dist_env.is_main or args.no_progress)
        optimizer.zero_grad(set_to_none=True)
        t_data, t_last = 0.0, time.time()
        for i, batch in enumerate(bar):
            t_data += time.time() - t_last
            batch = data.to_device(batch, device)
            if pipeline is not None:
                batch["image"] = pipeline(batch)
            omega = data.quadrature_weights_for(batch, args.K, args.placement, device, args.profile)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = net(batch["image"])
            pred = {k: v.float() for k, v in pred.items() if isinstance(v, torch.Tensor)}
            loss, metrics = profile_loss(
                pred, batch["theta"], omega, batch["valid"],
                nll=step >= args.nll_after and not regression,
                lambda_theta=args.lambda_theta,
                theta_max_gt=batch["Theta"],
            )
            if regression:
                loss = division_terms(args, pred, batch, loss, metrics)
            if args.dense_loss_weight > 0:
                d_loss, d_metrics = dense_lib.dense_loss(
                    pred["tangent_coords"], batch, *batch["image"].shape[-2:], repr=args.dense_repr
                )
                loss = loss + args.dense_loss_weight * d_loss
                metrics.update(d_metrics)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(core.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
            n_batches += 1
            invalid += len(batch["valid"]) - metrics["n"]
            sums["loss"] += float(loss.detach())
            if metrics["n"]:
                for key in sums:
                    if key != "loss":
                        sums[key] += metrics[key]
            t_last = time.time()
            if dist_env.is_main and args.log_every and step % args.log_every == 0 and (i + 1) % accum == 0:
                elapsed = time.time() - t0
                eta_epoch = elapsed / (i + 1) * (steps_per_epoch - (i + 1))
                dist_env.print(
                    f"  step {step:6d}  loss {float(loss.detach()):8.4f}  L1 {metrics['l1_deg']:6.3f} deg  "
                    f"FoV {metrics['fov_err_deg']:6.3f} deg  "
                    + (f"dense {metrics['dense_deg']:6.3f} deg  " if "dense_deg" in metrics else "")
                    + (f"df {metrics['f_err_pct']:5.2f}%  dkappa {metrics['kappa_err']:6.4f}  "
                       if "f_err_pct" in metrics else "")
                    + f"lr {scheduler.get_last_lr()[-1]:.2e}  "
                    f"[{i + 1}/{steps_per_epoch}, epoch eta {fmt_duration(eta_epoch)}, "
                    f"data wait {100 * t_data / max(elapsed, 1e-9):.0f}%]"
                )

        train_stats = {k: v / max(1, n_batches) for k, v in sums.items()}
        epoch_s = time.time() - t0
        train_stats.update(epoch=epoch, step=step, invalid=invalid, time_s=round(epoch_s, 1),
                           data_wait=round(t_data / max(epoch_s, 1e-9), 3))
        val_stats = validate(core, val_loaders, args, device, amp_dtype) if val_loaders else {}
        history.append({"train": train_stats, "val": val_stats})
        dist_env.print(
            f"epoch {epoch}: train loss {train_stats['loss']:.4f}  L1 {train_stats['l1_deg']:.3f} deg  "
            f"FoV {train_stats['fov_err_deg']:.3f} deg  "
            + (f"dense {train_stats['dense_deg']:.3f} deg  " if "dense_deg" in train_stats else "")
            + f"({train_stats['time_s'] / 60:.1f} min, {100 * train_stats['data_wait']:.0f}% waiting for data"
            + (f", {invalid} invalid samples" if invalid else "") + ")"
            + "".join(
                f"\n  val {name}: L1 {v['l1_deg']:.3f} deg  FoV {v['fov_err_deg_median']:.3f} deg (median)"
                f"  [{v['n']} samples" + (f", {v['invalid']} invalid]" if v["invalid"] else "]")
                for name, v in val_stats.items()
            )
        )
        if epoch == start_epoch and device.type == "cuda":
            dist_env.print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        save(epoch)
        wall = time.time() - t0
        epoch_walls.append(wall)
        train_stats["wall_s"] = round(wall, 1)
        write_history()
        left = args.epochs - epoch
        if left > 0:
            recent = epoch_walls[-3:]
            eta_run = left * sum(recent) / len(recent)
            dist_env.print(f"  epoch wall {fmt_duration(wall)} | {left} epoch(s) left, "
                           f"run eta {fmt_duration(eta_run)} (finish ~{fmt_finish(eta_run)})")
        dist_env.barrier()

    dist_env.print(f"done. {'checkpoint ' + str(ckpt_path) if ckpt_path else 'no checkpoint saved'}")
    dist_env.shutdown()


def submit(args, argv: list[str], parser: argparse.ArgumentParser):
    """Submit this script to SLURM with the same arguments."""
    if args.init == "dinov2":
        print(f"backbone init: {dinov2_checkpoint(args.backbone, args.dinov2_ckpt)}")
    elif args.init and not Path(args.init).expanduser().exists():
        raise SystemExit(f"--init {args.init}: no such checkpoint (run from the repository root?)")
    batch = args.batch_size * args.gpus * args.accum
    return slurm.submit(
        args, argv, parser, module="train", module_dir=Path(THIS_DIR),
        extra_note=f"effective batch size {batch} ({args.batch_size} per GPU x "
                   f"{args.gpus} GPU(s) x {args.accum} accumulation)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML whose keys set the defaults of any flag below (CLI still wins)")
    g = p.add_argument_group("data")
    g.add_argument("--data", nargs="+", default=["openpano_v2_gen"],
                   help="dataset directories, or names under the dataset roots (paths.py); `a+b` draws "
                        "the stores a and b as one share of --samples_per_dataset")
    g.add_argument("--samples_per_dataset", type=int, default=None,
                   help="an epoch is this many views drawn afresh from EACH training share; default: "
                        "every view once per epoch")
    g.add_argument("--no_preload", nargs="+", default=[], metavar="DATASET",
                   help="training datasets (by directory name) read per view instead of loaded into RAM; "
                        "the store is first copied to the job's local scratch if there is one")
    g.add_argument("--paths", default=None, help="machine in configs/paths (default: $LITECALIB_PATHS or local)")
    g.add_argument("--resolution", type=int, default=102_400, help="H*W of the network input")
    g.add_argument("--aspect_ratio", type=float, nargs=2, default=(0.5, 2.0))
    g.add_argument("--edge_divisible_by", type=int, default=None,
                   help="what each side of the network input must be a multiple of (default: 14)")
    g.add_argument("--augmentations", default="geocalib", help="siclib augmentation preset")
    g.add_argument("--gpu_pipeline", action="store_true",
                   help="decode, resize and augment the training batch on the GPU (train/gpu_pipeline.py)")
    g.add_argument("--augment_before_resize", action="store_true",
                   help="AnyCalib's order: augment the original image, then resize (slower)")
    g.add_argument("--allow_augmentation_mismatch", action="store_true",
                   help="train even if the installed albumentations ignores AnyCalib's settings")
    g.add_argument("--flip_prob", type=float, default=0.5, help="random h/v flip")
    g.add_argument("--edit_prob", type=float, default=0.0,
                   help="probability of an edited training view (AnyCalib's crop + change_pixel_ar: "
                        "c off center, fy != fx); --profile field only")
    g.add_argument("--edit_crop", type=float, default=0.5,
                   help="an edited view loses up to this fraction of each side's length")
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--prefetch", type=int, default=2)
    g.add_argument("--limit_train", type=int, default=None, help="smoke test: first N samples")
    g.add_argument("--limit_val", type=int, default=None)
    g.add_argument("--no_val", action="store_true")
    g.add_argument("--no_check_invariants", action="store_true",
                   help="skip the centered-c / square-pixel assertion on unedited views")
    g = p.add_argument_group("model")
    g.add_argument("--backbone", default="vitl14", choices=list(model_lib.BACKBONES))
    g.add_argument("--K", type=int, default=geometry.DEFAULT_K,
                   help="field: points per grid side (odd); radial / division: knots")
    g.add_argument("--profile", default="field", choices=list(model_lib.PROFILES),
                   help="field: the FoV field at a K x K grid (the method); radial: a monotone angle "
                        "profile along the diagonal; division: division:1's (f, k1) regressed directly")
    g.add_argument("--field_repr", default="fov", choices=list(geometry.FIELD_REPRS),
                   help="field outputs: fov, (atan2(X, Z), atan2(Y, Z)), or logmap, AnyCalib's log map")
    g.add_argument("--placement", default=geometry.DEFAULT_PLACEMENT, choices=list(geometry.KNOT_PLACEMENTS))
    g.add_argument("--decoder_layers", type=int, default=2)
    g.add_argument("--q_dim", type=int, default=None, help="default: the backbone's width")
    g.add_argument("--attn_heads", type=int, default=8)
    g.add_argument("--init_vfov", type=float, default=60.0, help="vFoV the heads start at")
    g.add_argument("--init", default=None,
                   help="'dinov2' for DINOv2's released weights, or a checkpoint; default: random")
    g.add_argument("--dinov2_ckpt", default=None, help="explicit path for --init dinov2")
    g = p.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--batch_size", type=int, default=24, help="per GPU")
    g.add_argument("--val_batch_size", type=int, default=24)
    g.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    g.add_argument("--lr", type=float, default=6e-5, help="decoder / heads")
    g.add_argument("--backbone_lr", type=float, default=6e-6)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--clip_grad", type=float, default=1.0)
    g.add_argument("--warmup_steps", type=int, default=1000)
    g.add_argument("--warmup_start_factor", type=float, default=1e-3)
    g.add_argument("--lr_schedule", default="step", choices=["step", "wsd"],
                   help="step: AnyCalib's (--lr_milestones, --lr_gamma). wsd: warmup-stable-decay, "
                        "the peak lr until the last --decay_epochs, then a 1 - sqrt cool-down to 0")
    g.add_argument("--lr_milestones", type=int, nargs="*", default=(10_000, 30_000))
    g.add_argument("--lr_gamma", type=float, default=0.3)
    g.add_argument("--decay_epochs", type=float, default=0.0, help="wsd: the cool-down's length in epochs")
    g.add_argument("--lambda_theta", type=float, default=0.0, help="radial: weight of the |Theta| term")
    g.add_argument("--dense_loss_weight", type=float, default=0.0,
                   help="> 0 adds the training-only dense branch and AnyCalib's l1-z1 loss on the GT ray "
                        "at every input pixel (train/dense.py), with this weight")
    g.add_argument("--dense_repr", default="logmap", choices=list(geometry.FIELD_REPRS),
                   help="what the dense branch predicts per pixel: logmap (AnyCalib's) or fov")
    g.add_argument("--dense_dim", type=int, default=256, help="the dense branch's width")
    g.add_argument("--div_profile_weight", type=float, default=1.0,
                   help="--profile division: weight of the profile loss on the knots rendered from (f, kappa)")
    g.add_argument("--div_param_weight", type=float, default=0.0,
                   help="--profile division: weight of |d log f| + div_kappa_weight |d kappa|")
    g.add_argument("--div_kappa_weight", type=float, default=1.0)
    g.add_argument("--nll_after", type=int, default=2000,
                   help="steps of weighted L1 before switching to the Laplace NLL")
    g.add_argument("--no_amp", dest="amp", action="store_false", help="disable mixed precision")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("runtime")
    g.add_argument("--tag", default="run", help="run name; used in the default output paths")
    g.add_argument("--out", default=None, help="default: results/train_<backbone>_<data>_<tag>.json")
    g.add_argument("--ckpt", default=None,
                   help="default: checkpoints/<backbone>_<data>_<tag>.pt; '' to skip saving")
    g.add_argument("--keep_epochs", type=int, nargs="*", default=[],
                   help="also keep these epochs' weights as <ckpt stem>_ep<N>.pt")
    g.add_argument("--no_keep_best", dest="keep_best", action="store_false",
                   help="do not keep the best-validation epoch's weights as <ckpt stem>_best.pt")
    g.add_argument("--resume", action="store_true", help="continue from --ckpt if it exists")
    g.add_argument("--branch_epochs", type=int, nargs="*", default=[],
                   help="also keep these epochs' full state as <ckpt stem>_ep<N>_full.pt, the start of a "
                        "--branch_from cool-down")
    g.add_argument("--branch_from", default=None,
                   help="start from a --branch_epochs snapshot and train on under this run's schedule (wsd)")
    g.add_argument("--no_optimizer_state", action="store_true", help="save only the weights")
    g.add_argument("--device", default="cuda")
    g.add_argument("--ddp_gpus", type=int, default=0, help="GPUs to train on (0 = every visible one)")
    g.add_argument("--log_every", type=int, default=100, help="steps between progress lines")
    g.add_argument("--no_progress", action="store_true", help="no tqdm bar (cleaner logs)")
    slurm.add_arguments(p, job_name="litecalib", log_dir="logs/train_submit/%t/%j")
    return p


def parse(
    argv: list[str] | None = None, parser: argparse.ArgumentParser | None = None,
) -> tuple[argparse.Namespace, list[str], argparse.ArgumentParser]:
    """Parse the command line over the ``--config`` defaults, check it, and fill in the output paths."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    if args.config:
        conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        unknown = set(conf) - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        parser.set_defaults(**conf)
        args = parser.parse_args(argv)
    if args.profile == "field" and (args.K < 3 or args.K % 2 == 0):
        parser.error(f"--profile field needs an odd --K >= 3, got {args.K}")
    if args.profile in model_lib.REGRESSION_CAMS and args.div_profile_weight == args.div_param_weight == 0:
        parser.error(f"--profile {args.profile} with both loss weights at 0 trains nothing")
    if args.edge_divisible_by is None:
        args.edge_divisible_by = model_lib.PATCH_SIZE
    elif args.edge_divisible_by % model_lib.PATCH_SIZE:
        parser.error(f"--edge_divisible_by must be a multiple of {model_lib.PATCH_SIZE}")
    if args.samples_per_dataset is not None and args.samples_per_dataset * len(args.data) < args.batch_size:
        parser.error("--samples_per_dataset x datasets is less than one batch")
    unknown = sorted(set(args.no_preload) - {Path(p).name for d in args.data for p in d.split("+")})
    if unknown:
        parser.error(f"--no_preload {unknown}: not among --data {args.data}")
    if not 0.0 <= args.edit_prob <= 1.0 or not 0.0 <= args.edit_crop < 1.0:
        parser.error("--edit_prob must be in [0, 1] and --edit_crop in [0, 1)")
    if args.edit_prob and args.profile != "field" and not getattr(args, "dense_targets_only", False):
        parser.error("--edit_prob: edited views have an off-center c and fy != fx; use --profile field")
    if args.dense_repr != "logmap" and not args.dense_loss_weight > 0:
        parser.error("--dense_repr is the dense branch's: give --dense_loss_weight")
    if args.lr_schedule != "wsd" and (args.decay_epochs or args.branch_from):
        parser.error("--decay_epochs / --branch_from are for --lr_schedule wsd")
    if args.branch_epochs and (args.no_optimizer_state or args.ckpt == ""):
        parser.error("--branch_epochs keeps the full state: it needs --ckpt and the optimizer state")
    data_tag = "+".join(Path(d).name for d in args.data)
    stem = f"{args.backbone}_{data_tag}_{args.tag}"
    if args.out is None:
        args.out = str(ROOT / "results" / f"train_{stem}.json")
    if args.ckpt is None:
        args.ckpt = str(ROOT / "checkpoints" / f"{stem}.pt")
    return args, argv, parser


def world_size_for(args) -> int:
    """How many GPU workers to start: every visible GPU, unless asked for fewer."""
    if args.device != "cuda" or not torch.cuda.is_available():
        return 1
    n = torch.cuda.device_count()
    return n if args.ddp_gpus <= 0 else min(n, args.ddp_gpus)


def _worker(rank: int, world: int, args) -> None:
    """One DDP rank. Module-level so that `spawn` can pickle it."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    run(args)


def launch(args) -> None:
    """Run `run(args)` on every GPU of this node, then remove whatever `--no_preload` staged on scratch."""
    try:
        world = world_size_for(args)
        if world <= 1:
            run(args)
            return
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sock:
                sock.bind(("", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        print(f"starting {world} DDP workers on this node (port {os.environ['MASTER_PORT']})", flush=True)
        torch.multiprocessing.spawn(_worker, args=(world, args), nprocs=world, join=True)
    finally:
        data.clear_stage()


def main(argv: list[str] | None = None) -> None:
    args, argv, parser = parse(argv)
    if args.submitit:
        submit(args, argv, parser)
        return
    launch(args)


if __name__ == "__main__":
    main()
