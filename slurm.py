"""Submitting a script to SLURM from inside itself (``--submitit``), through `submitit`."""

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

__all__ = ["add_arguments", "strip_flags", "submit", "run_module", "FLAGS"]

FLAGS = (
    "submitit", "gpus", "cpus_per_task", "mem_gb", "timeout_min", "partition",
    "account", "job_name", "slurm_log_dir", "conda_env", "exclude", "hpc", "afterok",
)
# EDIT FOR YOUR CLUSTER: the defaults below and the HPC profiles are examples from the clusters the
# paper's runs used. Set your SLURM account and partition names here (or pass --account / --partition).
DEFAULTS = {
    "gpus": 1, "cpus_per_task": None, "mem_gb": 64, "timeout_min": 480,
    "partition": "gpu", "account": None,
}
CPUS_PER_GPU = 16
# --hpc <name>: partition renames ("gpu" / "short" as used in the configs -> the cluster's names), CPUs
# per GPU and, for aarch64 GPU nodes, a separate conda installation whose python the job runs.
HPC = {
    "default": {"partitions": {}, "cpus_per_gpu": CPUS_PER_GPU},
    "example_arm": {
        "partitions": {"gpu": "gpu_short", "short": "cpu_short"},
        "cpus_per_gpu": 72,
        "arm_partitions": ("gpu_short", "gpu_medium", "gpu_long"),
        "arm_conda": "~/miniforge3-aarch64",
    },
}


def add_arguments(
    parser: argparse.ArgumentParser, job_name: str, log_dir: str, **overrides
) -> argparse.ArgumentParser:
    """Add the `--submitit` group, with this cluster's defaults."""
    d = {**DEFAULTS, **overrides}
    g = parser.add_argument_group("SLURM (--submitit)")
    g.add_argument("--submitit", action="store_true",
                   help="submit this script to SLURM instead of running it here")
    g.add_argument("--gpus", type=int, default=d["gpus"], help="GPUs on the one node")
    g.add_argument("--cpus_per_task", type=int, default=d["cpus_per_task"],
                   help=f"CPUs for the job's single task (default: {CPUS_PER_GPU} per GPU)")
    g.add_argument("--mem_gb", type=int, default=d["mem_gb"])
    g.add_argument("--timeout_min", type=int, default=d["timeout_min"])
    g.add_argument("--partition", default=d["partition"])
    g.add_argument("--account", default=d["account"])
    g.add_argument("--job_name", default=job_name)
    g.add_argument("--slurm_log_dir", default=log_dir,
                   help="submitit job folder; %%t is the run tag, %%j the job id")
    g.add_argument("--conda_env", default=None, help="conda env to activate inside the job")
    g.add_argument("--exclude", default=None,
                   help="nodes the job must not run on (e.g. a node with a bad GPU)")
    g.add_argument("--afterok", default=None, metavar="JOB[:JOB...]",
                   help="start only once these jobs have completed successfully; if one fails, "
                   "this job is cancelled rather than left waiting")
    g.add_argument("--hpc", choices=sorted(HPC), default="default",
                   help="the cluster submitted to: its partition names, CPUs per GPU and, on "
                   "aarch64 GPU nodes, the job's interpreter")
    return parser


def strip_flags(argv: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """Drop the SLURM-only flags from an argv, leaving what the job should run."""
    takes_value, flagless = set(), set()
    for action in parser._actions:
        if action.dest not in FLAGS:
            continue
        (flagless if action.nargs == 0 else takes_value).update(action.option_strings)
    out, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        name = token.split("=", 1)[0]
        if name in flagless:
            continue
        if name in takes_value:
            skip = "=" not in token
            continue
        out.append(token)
    return out


def run_module(module: str, argv: list[str], search_paths: list[str]) -> None:
    """Entry point *inside* the job: set up the environment, then call `module.main(argv)`."""
    sys.path[:0] = [p for p in search_paths if p not in sys.path]
    importlib.import_module(module).main(argv)


def device_or_cpu(device: str) -> str:
    """`device`, unless it is ``cuda`` and no GPU is visible: then the CPU -- or, inside a SLURM job that was
    given GPUs, an error.
    """
    import os
    import socket

    import torch

    if device != "cuda" or torch.cuda.is_available():
        return device
    gpus = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE")
    if os.environ.get("SLURM_JOB_ID") and gpus and gpus != "0":
        host = socket.gethostname().split(".")[0]
        raise RuntimeError(
            f"SLURM job {os.environ['SLURM_JOB_ID']} on {host} was given GPU(s) {gpus} but CUDA sees "
            f"none (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}): a bad GPU on "
            f"the node, not a CPU run; resubmit with --exclude {host}"
        )
    print("no CUDA device visible; falling back to the CPU", flush=True)
    return "cpu"


def memory_parameters(mem_gb: int, gpus: int) -> dict:
    """`--mem_gb` (the job's total) as a submitit parameter: per GPU for GPU jobs."""
    if gpus > 0:
        return {"slurm_mem_per_gpu": f"{mem_gb * 1024 // gpus}M"}
    return {"mem_gb": mem_gb}


def submit(
    args: argparse.Namespace,
    argv: list[str],
    parser: argparse.ArgumentParser,
    module: str,
    module_dir: Path,
    tag: str = "",
    extra_note: str = "",
    cluster: str | None = None,
):
    """Submit `module.main(argv)` to SLURM with the resources in `args`."""
    import submitit

    job_argv = strip_flags(argv, parser)
    log_dir = args.slurm_log_dir.replace("%t", tag or getattr(args, "tag", "") or module)
    search_paths = [str(module_dir), str(ROOT)]
    setup = [
        'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"',
        f'export PYTHONPATH="{":".join(search_paths)}:$PYTHONPATH"',
    ]
    hpc_name = getattr(args, "hpc", None) or "default"
    hpc = HPC[hpc_name]
    partition = hpc["partitions"].get(args.partition, args.partition)
    executor_kw = {}
    if partition in hpc.get("arm_partitions", ()):
        prefix = Path(hpc["arm_conda"]).expanduser() / "envs" / (args.conda_env or "litecalib")
        setup[:0] = [f'export CONDA_PREFIX="{prefix}"', 'export PATH="$CONDA_PREFIX/bin:$PATH"']
        executor_kw["slurm_python"] = str(prefix / "bin" / "python")
    elif args.conda_env:
        setup.insert(0, f"conda activate {args.conda_env}")
    slurm_extra = {}
    if partition:
        slurm_extra["slurm_partition"] = partition
    if args.account:
        slurm_extra["slurm_account"] = args.account
    if getattr(args, "exclude", None):
        slurm_extra["slurm_exclude"] = args.exclude
    if getattr(args, "afterok", None):
        slurm_extra["slurm_additional_parameters"] = {
            "dependency": f"afterok:{args.afterok}", "kill-on-invalid-dep": "yes"}

    cpus = args.cpus_per_task or hpc["cpus_per_gpu"] * max(1, args.gpus)
    executor = submitit.AutoExecutor(folder=log_dir, cluster=cluster, **executor_kw)
    executor.update_parameters(
        slurm_job_name=args.job_name,
        nodes=1,
        gpus_per_node=args.gpus,
        tasks_per_node=1,
        cpus_per_task=cpus,
        **memory_parameters(args.mem_gb, args.gpus),
        timeout_min=args.timeout_min,
        stderr_to_stdout=True,
        slurm_setup=setup,
        **slurm_extra,
    )
    job = executor.submit(run_module, module, job_argv, search_paths)
    print(
        f"submitted job {job.job_id} ({hpc_name}, {partition}): 1 node x {args.gpus} GPU(s), "
        f"1 task x {cpus} CPUs, {args.mem_gb} GB, {args.timeout_min} min"
        + (f"\n  {extra_note}" if extra_note else "")
        + f"\n  log:  {log_dir.replace('%j', str(job.job_id))}/{job.job_id}_0_log.out"
        f"\n  args: {' '.join(job_argv)}"
    )
    return job
