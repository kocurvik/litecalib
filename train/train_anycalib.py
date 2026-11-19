"""Train AnyCalib's network (dense FoV field) on the same data path as LiteCalib (released as
`anycalib_litecalib_data_vitl`; AnyCalib_ours in the paper).

    python train/train_anycalib.py --config configs/ms_ac_wsd24_vitl_opg.yaml
"""

import sys
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = str(Path(__file__).resolve().parent)
for _p in (str(ROOT), THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data
import slurm
import train as base
from dense import field_loss, gt_rays
from litecalib import model as model_lib
from litecalib.anycalib.model.anycalib_pretrained import AnyCalib

ARCH = "anycalib"

UNSUPPORTED = (
    "K", "profile", "placement", "decoder_layers", "q_dim", "attn_heads", "init_vfov",
    "lambda_theta", "div_profile_weight", "div_param_weight", "div_kappa_weight", "nll_after",
    "dense_loss_weight", "branch_from",
)


class AnyCalibField(AnyCalib):
    """Upstream's `AnyCalib` with `forward` cut before the calibrator: the field only."""

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.head(self.decoder(self.backbone(image)))
        b, _, h, w = image.shape
        return {
            "rays": out["rays"].permute(0, 2, 3, 1).reshape(b, h * w, 3),
            "tangent_coords": out["tangent_coords"].permute(0, 2, 3, 1).reshape(b, h * w, 2),
        }


def build_model(args, device) -> tuple[AnyCalibField, dict]:
    """AnyCalib's network, its backbone initialized from DINOv2's release (``--init dinov2``) or a checkpoint."""
    net = model_lib.build_anycalib(args.backbone, args.dense_dim, cls=AnyCalibField,
                                   sdpa=not args.no_sdpa)
    info = {"arch": ARCH, "init": args.init or "random", "backbone": args.backbone,
            "dense_dim": args.dense_dim, "sdpa": not args.no_sdpa}
    if args.init == "dinov2":
        ckpt = base.dinov2_checkpoint(args.backbone, args.dinov2_ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        net.backbone.model.load_state_dict(state, strict=True)
        info.update(ckpt=str(ckpt), loaded=len(state))
    elif args.init:
        state = torch.load(Path(args.init).expanduser(), map_location="cpu", weights_only=False)
        net.load_state_dict(state.get("model", state), strict=True)
        info["ckpt"] = args.init
    return net.to(device), info


def param_groups(net: torch.nn.Module, lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
    """siclib's `pack_lr_parameters` with `lr_scaling: [[0.1, [backbone]]]`: two lr groups, weight decay on every
    trainable parameter.
    """
    groups = {True: [], False: []}
    for n, p in net.named_parameters():
        if p.requires_grad:
            groups["backbone" in n].append(p)
    return [{"params": groups[True], "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"},
            {"params": groups[False], "lr": lr, "weight_decay": weight_decay, "name": "head"}]


@torch.no_grad()
def validate(net, loaders: dict, device, amp_dtype) -> dict:
    """Mean / median angular error of the dense field on each validation loader."""
    net.eval()
    out = {}
    for name, loader in loaders.items():
        errs, loss_sum, n = [], 0.0, 0
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            h, w = image.shape[-2:]
            rays, valid = gt_rays(batch, h, w, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = net(image)
            pred = {k: v.float() for k, v in pred.items()}
            loss, m = field_loss(pred, rays, valid)
            errs.append(m["ang_err_deg"].cpu())
            loss_sum += float(loss) * len(image)
            n += len(image)
        e = torch.cat(errs) if errs else torch.zeros(1)
        out[name] = {"n": n, "loss": loss_sum / max(1, n),
                     "ang_err_deg_mean": float(e.mean()), "ang_err_deg_median": float(e.median())}
    net.train()
    return out


def val_score(val_stats: dict) -> float | None:
    """What `_best.pt` minimizes: the mean angular error (upstream's `best_key`), pooled over every val set by
    its image count.
    """
    n = sum(v["n"] for v in val_stats.values())
    return sum(v["ang_err_deg_mean"] * v["n"] for v in val_stats.values()) / n if n else None


def best_epoch(history: list[dict]) -> dict | None:
    """The epoch with the lowest `val_score`."""
    scored = [(s, h["train"]["epoch"]) for h in history
              if (s := val_score(h.get("val") or {})) is not None]
    if not scored:
        return None
    score, epoch = min(scored, key=lambda t: t[0])
    return {"epoch": epoch, "val_ang_err_deg": score}


def run(args) -> None:
    """One training run (one DDP rank)."""
    args.device = slurm.device_or_cpu(args.device)
    dist_env = base.Dist(cuda=args.device == "cuda")
    torch.manual_seed(args.seed + dist_env.rank)
    device = torch.device(f"cuda:{dist_env.local_rank}" if args.device == "cuda" else args.device)
    amp_dtype = None
    if args.amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    train_set, train_loader, val_loaders = base.build_loaders(args, dist_env)
    pipeline = None
    if args.gpu_pipeline:
        from gpu_pipeline import GpuPipeline

        pipeline = GpuPipeline(augment=args.augmentations != "identity",
                               seed=args.seed + dist_env.rank, device=device)
    net, init_info = build_model(args, device)
    dist_env.print(
        f"model: AnyCalib (DINOv2 {args.backbone} + light DPT, {args.dense_dim} wide + convex "
        f"tangent head; attention {'upstream' if args.no_sdpa else 'sdpa'}), "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters "
        f"({sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.1f} M trainable)"
        f"\nloss: l1-z1 (dense, every valid pixel)"
        f"\ninit: {init_info}"
        f"\ntrain: {data.summarize(train_set)}, camera models {data.cam_id_counts(train_set)}"
        + (f"\nepoch: {args.samples_per_dataset} views drawn afresh from each of "
           f"{len(getattr(train_set, 'datasets', [train_set]))} training sets = {len(train_loader)} steps"
           if args.samples_per_dataset else "")
        + (f"\nnot preloaded (each view read from disk when sampled): {', '.join(args.no_preload)}"
           if args.no_preload else "")
        + f"\nval: {', '.join(f'{k} ({len(v.dataset)})' for k, v in val_loaders.items()) or 'disabled'}"
        f"\npipeline: {'gpu (decode + resize + augment)' if pipeline is not None else 'cpu'}"
        f"\ndevice: {device}, world size {dist_env.world_size}, amp {amp_dtype}, "
        f"effective batch {args.batch_size * dist_env.world_size * max(1, args.accum)}"
    )

    if dist_env.enabled:
        net = DistributedDataParallel(net, device_ids=[dist_env.local_rank] if device.type == "cuda" else None)
    core = net.module if isinstance(net, DistributedDataParallel) else net

    optimizer = torch.optim.AdamW(param_groups(core, args.lr, args.backbone_lr, args.weight_decay))
    total_steps = len(train_loader) // max(1, args.accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, base.lr_lambda(args, total_steps))

    start_epoch, step, history = 1, 0, []
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if args.resume and ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        core.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        else:
            dist_env.print(f"{ckpt_path} has no optimizer state: the moments restart, the weights continue")
            for _ in range(state["step"]):
                scheduler.step()
        start_epoch, step, history = state["epoch"] + 1, state["step"], state["history"]
        dist_env.print(f"resumed from {ckpt_path} at epoch {state['epoch']} (step {step})")

    out_path = Path(args.out) if args.out else None
    steps_per_epoch = len(train_loader)
    accum = max(1, args.accum)

    def write(path: Path, st: dict) -> None:
        tmp = path.with_name(path.name + ".partial")
        torch.save(st, tmp)
        tmp.replace(path)

    def save(epoch: int) -> None:
        if not (dist_env.is_main and ckpt_path):
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"arch": ARCH, "model": core.state_dict(), "epoch": epoch, "step": step,
                 "history": history, "args": vars(args), "init": init_info}
        if epoch in args.keep_epochs:
            write(base.keep_path(ckpt_path, epoch), state)
        best = best_epoch(history) if args.keep_best else None
        if best is not None and best["epoch"] == epoch:
            write(base.best_path(ckpt_path), {**state, "best": best})
            dist_env.print(f"  new best val angular error {best['val_ang_err_deg']:.3f} deg "
                           f"-> {base.best_path(ckpt_path)}")
        if not args.no_optimizer_state:
            state["optimizer"] = optimizer.state_dict()
            state["scheduler"] = scheduler.state_dict()
        if epoch in args.branch_epochs:
            write(base.branch_path(ckpt_path, epoch), state)
        write(ckpt_path, state)

    def write_history() -> None:
        if dist_env.is_main and out_path:
            import json

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"arch": ARCH, "args": vars(args), "init": init_info, "history": history,
                 "best": best_epoch(history)}, indent=2, default=str), encoding="utf-8")

    epoch_walls: list[float] = []
    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        data.set_epoch(train_loader, epoch)
        sums = {"loss": 0.0, "ang_err_deg": 0.0}
        t0, n_batches = time.time(), 0
        bar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}", mininterval=30,
                   disable=not dist_env.is_main or args.no_progress)
        optimizer.zero_grad(set_to_none=True)
        t_data, t_last = 0.0, time.time()
        for i, batch in enumerate(bar):
            t_data += time.time() - t_last
            batch = data.to_device(batch, device)
            if pipeline is not None:
                batch["image"] = pipeline(batch)
            h, w = batch["image"].shape[-2:]
            rays, valid = gt_rays(batch, h, w, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = net(batch["image"])
            pred = {k: v.float() for k, v in pred.items()}
            loss, metrics = field_loss(pred, rays, valid)
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
            ang = float(metrics["ang_err_deg"].mean())
            sums["loss"] += float(loss.detach())
            sums["ang_err_deg"] += ang
            t_last = time.time()
            if dist_env.is_main and args.log_every and step % args.log_every == 0 and (i + 1) % accum == 0:
                elapsed = time.time() - t0
                eta_epoch = elapsed / (i + 1) * (steps_per_epoch - (i + 1))
                dist_env.print(
                    f"  step {step:6d}  loss {float(loss.detach()):8.5f}  ang {ang:6.3f} deg  "
                    f"lr {scheduler.get_last_lr()[-1]:.2e}  [{i + 1}/{steps_per_epoch}, epoch eta "
                    f"{base.fmt_duration(eta_epoch)}, data wait {100 * t_data / max(elapsed, 1e-9):.0f}%]"
                )

        train_stats = {k: v / max(1, n_batches) for k, v in sums.items()}
        epoch_s = time.time() - t0
        train_stats.update(epoch=epoch, step=step, time_s=round(epoch_s, 1),
                           data_wait=round(t_data / max(epoch_s, 1e-9), 3))
        val_stats = validate(core, val_loaders, device, amp_dtype) if val_loaders else {}
        history.append({"train": train_stats, "val": val_stats})
        dist_env.print(
            f"epoch {epoch}: train loss {train_stats['loss']:.5f}  ang {train_stats['ang_err_deg']:.3f} deg  "
            f"({epoch_s / 60:.1f} min, {100 * train_stats['data_wait']:.0f}% waiting for data)"
            + "".join(f"\n  val {name}: ang {v['ang_err_deg_mean']:.3f} deg mean, "
                      f"{v['ang_err_deg_median']:.3f} median  [{v['n']} samples]"
                      for name, v in val_stats.items())
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
            dist_env.print(f"  epoch wall {base.fmt_duration(wall)} | {left} epoch(s) left, "
                           f"run eta {base.fmt_duration(eta_run)} (finish ~{base.fmt_finish(eta_run)})")
        dist_env.barrier()

    dist_env.print(f"done. {'checkpoint ' + str(ckpt_path) if ckpt_path else 'no checkpoint saved'}")
    dist_env.shutdown()


def build_parser():
    """`train.py`'s parser with AnyCalib's defaults and ``--no_sdpa``."""
    p = base.build_parser()
    p.description = __doc__
    p.set_defaults(init="dinov2", tag="ac", job_name="anycalib", profile="radial")
    p.set_defaults(dense_targets_only=True)
    p.add_argument("--no_sdpa", action="store_true",
                   help="upstream's attention as is (without xformers: the full attention matrix)")
    return p


def parse(argv: list[str] | None = None):
    """`train.parse`, refusing the options AnyCalib's network does not use."""
    parser = build_parser()
    args, argv, parser = base.parse(argv, parser)
    changed = [d for d in UNSUPPORTED if getattr(args, d) != parser.get_default(d)]
    if changed:
        parser.error(f"{', '.join('--' + d for d in changed)}: not used by AnyCalib's network")
    return args, argv, parser


def _worker(rank: int, world: int, args) -> None:
    """One DDP rank."""
    import os

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    run(args)


def launch(args) -> None:
    """train.py's `launch`, with this script's `run` (the staging is cleaned up either way)."""
    import os
    import socket

    try:
        world = base.world_size_for(args)
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
        if args.init == "dinov2":
            print(f"backbone init: {base.dinov2_checkpoint(args.backbone, args.dinov2_ckpt)}")
        elif args.init and not Path(args.init).expanduser().exists():
            raise SystemExit(f"--init {args.init}: no such checkpoint")
        slurm.submit(args, argv, parser, module="train_anycalib", module_dir=Path(THIS_DIR),
                     extra_note=f"effective batch size {args.batch_size * args.gpus * args.accum}")
        return
    launch(args)


if __name__ == "__main__":
    main()
