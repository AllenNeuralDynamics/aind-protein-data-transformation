"""Submit ``ZeissCompressionJob`` partitions to SLURM via the REST API.

This is a debug-oriented launcher for the CZI -> Zarr -> S3 ``compress_data``
job defined in :mod:`aind_hcr_data_transformation.zeiss_job`. It exists to help
diagnose stalls / failures on very large (>10 TB) Zeiss datasets by giving
deep per-worker visibility:

* one independent SLURM job per partition (via ``aind_slurm_rest_v2``)
* verbose prologue diagnostics on every compute node
* periodic Python stack dumps (``py-spy`` preferred, ``faulthandler``
  + ``SIGUSR1`` fallback)
* live ``tail -F`` of every worker's stdout/stderr/pyspy log
* "no-progress" stall watchdog
* on failure, automatic ``slurmdb`` lookup (exit code, max RSS, reason)

Run this from the ``hpc`` login node where ``SLURM_USER`` and ``SLURM_TOKEN``
are available in the environment. Example::

    /allen/programs/mindscope/workgroups/omfish/carsonb/rocky_miniconda/envs/\
upload_v3/bin/python scripts/submit_via_slurm_rest.py \\
        --input-source /allen/aind/stage/Z1/.../HCR_831990_2026-05-11_00-00-00 \\
        --s3-location s3://aind-open-data-dev-u5u0i5/HCR_831990_2026-05-11 \\
        --aws-profile default \\
        --yes

Notes
-----
This script intentionally lives outside the library code: it is an
operational aid, not a feature. It does not modify
:class:`aind_hcr_data_transformation.zeiss_job.ZeissCompressionJob`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- Defaults mirroring hcr-czitile-utils/scripts/transfer_airflow.py -------
DEFAULT_MAX_PARTITIONS: int = 32
DEFAULT_CPUS_PER_TASK: int = 4
SCHEDULING_OVERHEAD_MB_PER_TILE: int = 1300
SHARD_PROCESSING_OVERHEAD_MB: int = 4400
BUFFER_MEMORY_MB: int = 1000
MIN_RAM_PER_NODE_MB: int = 24_000
MAX_RAM_PER_NODE_MB: int = 40_000
PROCESSING_SPEED_MB_PER_HOUR: int = 4200
TIME_BUFFER_MIN: int = 60

DEFAULT_SLURM_HOST: str = "http://slurm2/api"
DEFAULT_SLURM_PARTITION: str = "aind"
DEFAULT_LOG_ROOT: str = "/allen/aind/scratch/carson.berry/hpc_debug"
DEFAULT_REPO_DIR: str = (
    "/allen/aind/scratch/carson.berry/aind-hcr-data-transformation"
)
# The *worker* env needs aind_hcr_data_transformation + boto3 + zarr +
# tensorstore + awscli. The dedicated `hcr_transform` conda env was created
# specifically for this purpose so we don't disturb other environments.
# The *submitter* env (typically upload_v3) only needs aind-slurm-rest-v2;
# that's a separate concern handled by the user when they invoke this
# script.
DEFAULT_PYTHON: str = (
    "/allen/programs/mindscope/workgroups/omfish/carsonb/"
    "rocky_miniconda/envs/hcr_transform/bin/python"
)


# ---- Logging ----------------------------------------------------------------
LOG_FMT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"
log = logging.getLogger("submit_via_slurm_rest")


def _setup_logging(log_dir: Path) -> None:
    """Configure root logger to write to console and ``submitter.log``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "submitter.log"),
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(LOG_FMT, datefmt=DATE_FMT))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        root.addHandler(h)


# ---- Resource sizing --------------------------------------------------------
@dataclass
class ResourcePlan:
    """Computed SLURM resources for the run."""

    num_partitions: int
    cpus_per_task: int
    mem_per_cpu_mb: int
    time_limit_min: int
    num_tiles: int
    first_tile_mb: float


def _list_czi_tiles(input_source: Path) -> List[Path]:
    """Return sorted list of ``*.czi`` tiles inside ``<input_source>/SPIM``."""
    spim = input_source / "SPIM"
    if not spim.is_dir():
        raise FileNotFoundError(f"SPIM folder missing: {spim}")
    tiles = sorted(p for p in spim.glob("*.czi") if p.is_file())
    if not tiles:
        raise FileNotFoundError(f"No .czi tiles in {spim}")
    return tiles


def compute_resource_plan(
    tiles: List[Path],
    *,
    num_partitions_override: Optional[int],
    cpus_per_task: int,
    mem_per_cpu_mb_override: Optional[int],
    time_limit_min_override: Optional[int],
) -> ResourcePlan:
    """Mirror the resource math in ``transfer_airflow.py``.

    Parameters
    ----------
    tiles : list of Path
        All discovered CZI tiles.
    num_partitions_override : int, optional
        If set, use exactly this many partitions; else
        ``min(len(tiles), DEFAULT_MAX_PARTITIONS)``.
    cpus_per_task : int
        SLURM ``minimum_cpus`` per task.
    mem_per_cpu_mb_override : int, optional
        If set, override the auto-computed memory.
    time_limit_min_override : int, optional
        If set, override the auto-computed time limit.

    Returns
    -------
    ResourcePlan
    """
    n_tiles = len(tiles)
    num_partitions = num_partitions_override or min(
        n_tiles, DEFAULT_MAX_PARTITIONS
    )
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")

    tiles_per_partition = max(1, n_tiles // num_partitions)
    sched_overhead = tiles_per_partition * SCHEDULING_OVERHEAD_MB_PER_TILE
    estimated_mem_per_cpu = (
        sched_overhead + SHARD_PROCESSING_OVERHEAD_MB + BUFFER_MEMORY_MB
    ) // cpus_per_task
    floor_mem = MIN_RAM_PER_NODE_MB // cpus_per_task
    ceil_mem = MAX_RAM_PER_NODE_MB // cpus_per_task
    mem_per_cpu_mb = mem_per_cpu_mb_override or min(
        max(estimated_mem_per_cpu, floor_mem), ceil_mem
    )

    first_tile_mb = tiles[0].stat().st_size / (1024 * 1024)
    estimated_min = int(
        (n_tiles * first_tile_mb / 1024) / PROCESSING_SPEED_MB_PER_HOUR
        + TIME_BUFFER_MIN
    )
    time_limit_min = time_limit_min_override or estimated_min

    return ResourcePlan(
        num_partitions=num_partitions,
        cpus_per_task=cpus_per_task,
        mem_per_cpu_mb=int(mem_per_cpu_mb),
        time_limit_min=int(time_limit_min),
        num_tiles=n_tiles,
        first_tile_mb=first_tile_mb,
    )


# ---- Per-partition job settings & worker bash ------------------------------
def _build_job_settings_json(
    *,
    input_source: Path,
    s3_location: str,
    num_partitions: int,
    partition: int,
    output_directory: str,
) -> str:
    """Return the JSON string handed to ``zeiss_job --job-settings``.

    The payload matches :class:`aind_hcr_data_transformation.models.\
ZeissJobSettings` and only sets fields the worker needs; defaults from the
pydantic model fill in the rest.
    """
    payload: Dict[str, object] = {
        "input_source": str(input_source),
        "output_directory": output_directory,
        "s3_location": s3_location,
        "num_of_partitions": num_partitions,
        "partition_to_process": partition,
    }
    return json.dumps(payload)


def _shell_single_quote(value: str) -> str:
    """Wrap ``value`` in single quotes safe for embedding in bash."""
    return "'" + value.replace("'", "'\\''") + "'"


def build_worker_script(
    *,
    partition: int,
    num_partitions: int,
    job_settings_json: str,
    log_dir: Path,
    repo_dir: Path,
    python_exe: str,
    aws_profile: Optional[str],
    py_spy_interval_sec: int,
    heartbeat_sec: int = 30,
) -> str:
    """Return the bash script body submitted as the SLURM job script.

    The script:

    1. Prints a verbose diagnostic prologue (host, mem, df, mounts, env, AWS
       identity, DNS sanity).
    2. Launches a heartbeat loop and an optional ``py-spy`` periodic dumper,
       installing a ``SIGUSR1`` trap so ``scancel --signal=USR1 <jobid>``
       triggers an on-demand stack dump.
    3. Runs the worker as ``python -c "<bootstrap>"`` so that
       :mod:`faulthandler` is registered against ``SIGUSR1`` even when
       ``py-spy`` is not permitted by the kernel.
    4. Captures the worker's exit code, kills the background helpers, and
       exits with that code.
    """
    worker_out = log_dir / f"worker_{partition:03d}.out"
    worker_err = log_dir / f"worker_{partition:03d}.err"
    pyspy_log = log_dir / f"worker_{partition:03d}.pyspy.log"

    js_quoted = _shell_single_quote(job_settings_json)
    aws_export = (
        f'export AWS_PROFILE={_shell_single_quote(aws_profile)}\n'
        if aws_profile
        else "# AWS_PROFILE not set; relying on default credential chain\n"
    )

    # The python bootstrap turns SIGUSR1 into a faulthandler dump, then
    # delegates to the existing module entrypoint.
    py_bootstrap = (
        "import faulthandler, signal, runpy, sys; "
        "faulthandler.enable(); "
        "faulthandler.register(signal.SIGUSR1, chain=False); "
        "sys.argv = ['zeiss_job', '--job-settings', "
        f"{js_quoted}]; "
        "runpy.run_module("
        "'aind_hcr_data_transformation.zeiss_job', run_name='__main__')"
    )

    pyspy_block = ""
    if py_spy_interval_sec > 0:
        pyspy_block = f"""
# Periodic py-spy dumps (best-effort; needs CAP_SYS_PTRACE or ptrace_scope<=1)
(
  while kill -0 "$WORKER_PID" 2>/dev/null; do
    sleep {py_spy_interval_sec}
    if command -v py-spy >/dev/null 2>&1; then
      echo "--- py-spy dump $(date -u +%FT%TZ) pid=$WORKER_PID ---" \\
        >> {pyspy_log}
      py-spy dump --pid "$WORKER_PID" >> {pyspy_log} 2>&1 || \\
        kill -USR1 "$WORKER_PID" 2>/dev/null || true
    else
      echo "--- faulthandler USR1 $(date -u +%FT%TZ) pid=$WORKER_PID ---" \\
        >> {pyspy_log}
      kill -USR1 "$WORKER_PID" 2>/dev/null || true
    fi
  done
) &
PYSPY_PID=$!
"""

    return f"""#!/bin/bash
# Auto-generated by scripts/submit_via_slurm_rest.py
# partition={partition}/{num_partitions}
set -uo pipefail

PARTITION={partition}
NUM_PARTITIONS={num_partitions}
LOG_DIR={log_dir}
REPO_DIR={repo_dir}
PYTHON={python_exe}
WORKER_OUT={worker_out}
WORKER_ERR={worker_err}

mkdir -p "$LOG_DIR"

# Mirror everything to per-worker logs so the submitter can tail them.
exec > >(stdbuf -oL tee -a "$WORKER_OUT") \\
     2> >(stdbuf -oL tee -a "$WORKER_ERR" >&2)

echo "===== HCR compress worker p${{PARTITION}}/${{NUM_PARTITIONS}} ====="
echo "submit_time_utc=$(date -u +%FT%TZ)"
echo "hostname=$(hostname -f 2>/dev/null || hostname)"
echo "user=$(id -un) uid=$(id -u) gid=$(id -g)"
echo "slurm_job_id=${{SLURM_JOB_ID:-unknown}} step=${{SLURM_STEP_ID:-unknown}}"
echo "slurm_node_list=${{SLURM_JOB_NODELIST:-unknown}}"
echo "cwd=$(pwd)"
echo "---- ulimit ----"; ulimit -a
echo "---- free -h ----"; free -h
echo "---- nproc ----"; nproc
echo "---- uname -a ----"; uname -a
echo "---- df -h key paths ----"
df -h /allen /tmp /scratch 2>/dev/null || df -h
echo "---- network ----"
ip -brief addr 2>/dev/null || ifconfig -a 2>/dev/null || true
echo "---- mounts (allen|s3|fuse) ----"
mount | grep -Ei 'allen|s3|fuse' || true
echo "---- env (filtered, secrets masked) ----"
env | sort | grep -Ei '^(AWS|SLURM|PATH|LD_|PYTHON|HOME|USER|TMP)' \\
  | sed -E 's/(TOKEN|SECRET|KEY)=.*/\\1=***MASKED***/'
echo "---- DNS sanity ----"
getent hosts s3.amazonaws.com 2>/dev/null || \\
  nslookup s3.amazonaws.com 2>/dev/null || true
{aws_export}
echo "---- aws sts get-caller-identity ----"
if command -v aws >/dev/null 2>&1; then
  aws sts get-caller-identity 2>&1 || true
else
  echo "aws cli not on PATH"
fi
echo "---- python ----"
"$PYTHON" --version
"$PYTHON" -c "import boto3, aind_hcr_data_transformation as h; \\
print('boto3', boto3.__version__, 'pkg', h.__file__)"
echo "===== begin worker ====="

cd "$REPO_DIR"
export LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export AWS_MAX_ATTEMPTS=10
export AWS_RETRY_MODE=adaptive
# Verbose boto3 / s3transfer logging
export BOTO_LOG_LEVEL=DEBUG

# Heartbeat: prove the worker is at least scheduled even if no log lines.
(
  while true; do
    echo "[hb $(date -u +%FT%TZ)] mem=$(free -m | awk '/Mem:/{{print $3\"/\"$2\"MB\"}}')\
 load=$(awk '{{print $1\" \"$2\" \"$3}}' /proc/loadavg)\
 nproc_running=$(ps -L -p $$ --no-headers 2>/dev/null | wc -l)"
    sleep {heartbeat_sec}
  done
) &
HB_PID=$!

# Launch the worker in the background so we have a PID for py-spy / USR1.
"$PYTHON" -X faulthandler -u -c {_shell_single_quote(py_bootstrap)} &
WORKER_PID=$!
echo "worker_pid=$WORKER_PID"
{pyspy_block}

# Forward SIGTERM (slurm preempt / scancel) into the worker so it can clean up.
trap 'echo "[trap] SIGTERM received, forwarding to $WORKER_PID";\
 kill -TERM "$WORKER_PID" 2>/dev/null || true' TERM
trap 'echo "[trap] SIGUSR1 -> dumping stacks"; kill -USR1 "$WORKER_PID"\
 2>/dev/null || true' USR1

wait "$WORKER_PID"
RC=$?
echo "===== worker exit rc=$RC at $(date -u +%FT%TZ) ====="

kill "$HB_PID" 2>/dev/null || true
[ -n "${{PYSPY_PID:-}}" ] && kill "$PYSPY_PID" 2>/dev/null || true
exit $RC
"""


# ---- SLURM REST glue --------------------------------------------------------
def _make_slurm_clients(
    host: str,
) -> Tuple[object, object]:  # (SlurmApi, SlurmdbApi)
    """Build authenticated ``SlurmApi`` and ``SlurmdbApi`` clients.

    Reads ``SLURM_USER`` and ``SLURM_TOKEN`` from the environment and aborts
    with a hint about ``scontrol token`` if either is missing.
    """
    from aind_slurm_rest_v2 import ApiClient, Configuration
    from aind_slurm_rest_v2.api.slurm_api import SlurmApi
    from aind_slurm_rest_v2.api.slurmdb_api import SlurmdbApi

    user = os.environ.get("SLURM_USER")
    token = os.environ.get("SLURM_TOKEN")
    missing = [n for n, v in [("SLURM_USER", user), ("SLURM_TOKEN", token)]
               if not v]
    if missing:
        raise RuntimeError(
            f"Missing required env var(s): {', '.join(missing)}. "
            "On the hpc login node, generate a token with:\n"
            "    export SLURM_USER=$USER\n"
            "    export SLURM_TOKEN=$(scontrol token lifespan=86400 "
            "| awk -F= '{print $2}')"
        )
    config = Configuration(host=host, username=user, access_token=token)
    client = ApiClient(config)
    return SlurmApi(client), SlurmdbApi(client)


def _build_submit_request(
    *,
    partition: int,
    num_partitions: int,
    script: str,
    log_dir: Path,
    slurm_partition: str,
    plan: ResourcePlan,
    repo_dir: Path,
    python_exe: str,
):
    """Construct ``V0040JobSubmitReq`` for one partition."""
    from aind_slurm_rest_v2.models.v0040_job_desc_msg import V0040JobDescMsg
    from aind_slurm_rest_v2.models.v0040_job_submit_req import (
        V0040JobSubmitReq,
    )
    from aind_slurm_rest_v2.models.v0040_uint32_no_val import (
        V0040Uint32NoVal,
    )
    from aind_slurm_rest_v2.models.v0040_uint64_no_val import (
        V0040Uint64NoVal,
    )

    # Prepend the worker python's bin/ so the `aws` CLI shipped in that env
    # (used by aind_hcr_data_transformation.utils.sync_dir_to_s3) is on PATH
    # for the worker process running on the compute node.
    python_bin_dir = str(Path(python_exe).resolve().parent)
    hpc_env = [
        f"PATH={python_bin_dir}:/bin:/usr/bin:/usr/local/bin",
        "LD_LIBRARY_PATH=/lib:/lib64:/usr/local/lib",
        f"HOME={os.environ.get('HOME', '/tmp')}",
    ]
    # Forward AWS_* so worker shell can export them; values are visible only
    # on the slurmctld host -> compute node, not on the wire.
    for k, v in os.environ.items():
        if k.startswith("AWS_") and k not in {"AWS_SECRET_ACCESS_KEY"}:
            hpc_env.append(f"{k}={v}")

    job = V0040JobDescMsg(
        name=f"hcr_compress_p{partition:03d}_of_{num_partitions:03d}",
        partition=slurm_partition,
        environment=hpc_env,
        standard_output=str(log_dir / f"slurm_p{partition:03d}_%j.out"),
        standard_error=str(log_dir / f"slurm_p{partition:03d}_%j.err"),
        current_working_directory=str(repo_dir),
        time_limit=V0040Uint32NoVal(set=True, number=plan.time_limit_min),
        memory_per_cpu=V0040Uint64NoVal(
            set=True, number=plan.mem_per_cpu_mb
        ),
        tasks=1,
        minimum_cpus=plan.cpus_per_task,
        maximum_nodes=1,
    )
    return V0040JobSubmitReq(script=script, job=job)


# ---- Submission manifest ----------------------------------------------------
@dataclass
class SubmittedJob:
    """Record of a single submitted partition."""

    partition: int
    job_id: str
    settings_json: str
    worker_out: str
    worker_err: str
    pyspy_log: str
    slurm_out_glob: str
    slurm_err_glob: str


@dataclass
class SubmissionManifest:
    """Persisted record of a full run, used by ``--attach``."""

    run_id: str
    log_dir: str
    repo_dir: str
    input_source: str
    s3_location: str
    slurm_host: str
    slurm_partition: str
    plan: Dict[str, object]
    submitted_at_utc: str
    jobs: List[Dict[str, object]] = field(default_factory=list)


def _write_manifest(path: Path, manifest: SubmissionManifest) -> None:
    """Atomically write the run manifest to disk."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(manifest), indent=2))
    tmp.replace(path)


def _load_manifest(path: Path) -> SubmissionManifest:
    """Load a previously persisted manifest for ``--attach``."""
    raw = json.loads(path.read_text())
    return SubmissionManifest(**raw)


# ---- Tail + monitor ---------------------------------------------------------
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "DEADLINE",
}


class TailFollower:
    """Glob-based ``tail -F`` that emits new lines tagged with a prefix.

    SLURM's actual log filename is only known after the job is RUNNING (the
    ``%j`` is filled in), so we resolve via glob each tick.
    """

    def __init__(self, glob_pattern: str, tag: str) -> None:
        """Track a single file path pattern."""
        self.glob_pattern = glob_pattern
        self.tag = tag
        self._fh: Optional[object] = None
        self._path: Optional[Path] = None
        self._last_progress = time.monotonic()

    def _resolve(self) -> Optional[Path]:
        """Return the first matching path on disk, or ``None``."""
        if self._path and self._path.exists():
            return self._path
        from glob import glob as _glob

        matches = sorted(_glob(self.glob_pattern))
        if matches:
            self._path = Path(matches[0])
            return self._path
        return None

    def pump(self) -> List[str]:
        """Read any newly-appended lines; return them with the tag prefix."""
        path = self._resolve()
        if not path:
            return []
        if self._fh is None:
            try:
                self._fh = open(path, "r", encoding="utf-8", errors="replace")
            except OSError:
                return []
        out: List[str] = []
        while True:
            line = self._fh.readline()
            if not line:
                break
            out.append(f"{self.tag} {line.rstrip()}")
        if out:
            self._last_progress = time.monotonic()
        return out

    def seconds_since_progress(self) -> float:
        """Seconds since this file last produced a new line."""
        return time.monotonic() - self._last_progress

    def tail_lines(self, n: int = 50) -> List[str]:
        """Return the last ``n`` lines of the file (best-effort)."""
        path = self._resolve()
        if not path:
            return []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()[-n:]
        except OSError:
            return []


class JobMonitor(threading.Thread):
    """One thread per submitted partition: poll state + tail logs."""

    def __init__(
        self,
        *,
        sj: SubmittedJob,
        slurm_api,
        slurmdb_api,
        poll_sec: int,
        stall_threshold_sec: int,
        stop_event: threading.Event,
    ) -> None:
        """Initialise the monitor."""
        super().__init__(
            name=f"monitor-p{sj.partition:03d}", daemon=True
        )
        self.sj = sj
        self.slurm_api = slurm_api
        self.slurmdb_api = slurmdb_api
        self.poll_sec = poll_sec
        self.stall_threshold_sec = stall_threshold_sec
        self.stop_event = stop_event
        self.last_state: Optional[str] = None
        self.final_state: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.followers = [
            TailFollower(sj.slurm_out_glob, f"[p{sj.partition:03d}][slurm-out]"),
            TailFollower(sj.slurm_err_glob, f"[p{sj.partition:03d}][slurm-err]"),
            TailFollower(sj.worker_out, f"[p{sj.partition:03d}][stdout]"),
            TailFollower(sj.worker_err, f"[p{sj.partition:03d}][stderr]"),
            TailFollower(sj.pyspy_log, f"[p{sj.partition:03d}][pyspy]"),
        ]
        self._stall_announced = False

    def _poll_state(self) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Return (state, exit_code, node_list) for the job."""
        try:
            resp = self.slurm_api.slurm_v0040_get_job(job_id=self.sj.job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("p%03d: get_job failed: %s", self.sj.partition, exc)
            return None, None, None
        jobs = getattr(resp, "jobs", None) or []
        if not jobs:
            return None, None, None
        j = jobs[0]
        # Job state can be a list of strings in v0.0.40
        raw_state = getattr(j, "job_state", None)
        if isinstance(raw_state, list):
            state = raw_state[0] if raw_state else None
        else:
            state = raw_state
        exit_code_obj = getattr(j, "exit_code", None)
        ec = None
        if exit_code_obj is not None:
            ec = getattr(exit_code_obj, "return_code", None)
            ec = getattr(ec, "number", ec) if ec is not None else None
        nodes = getattr(j, "nodes", None)
        return state, ec, nodes

    def _fetch_slurmdb_record(self) -> None:
        """Print a slurmdb summary for a terminated job."""
        try:
            resp = self.slurmdb_api.slurmdb_v0040_get_job(
                job_id=self.sj.job_id
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "p%03d: slurmdb lookup failed: %s",
                self.sj.partition, exc,
            )
            return
        jobs = getattr(resp, "jobs", None) or []
        if not jobs:
            log.warning(
                "p%03d: no slurmdb record for job_id=%s",
                self.sj.partition, self.sj.job_id,
            )
            return
        j = jobs[0]
        try:
            dump = j.to_dict() if hasattr(j, "to_dict") else dict(j.__dict__)
            interesting = {
                k: dump.get(k)
                for k in (
                    "state", "exit_code", "derived_exit_code", "reason",
                    "nodes", "elapsed", "time", "tres", "submit_line",
                )
                if k in dump
            }
            log.info(
                "p%03d slurmdb record: %s",
                self.sj.partition,
                json.dumps(interesting, default=str)[:4000],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("p%03d: slurmdb dump failed: %s",
                        self.sj.partition, exc)

    def _emit_lines(self, lines: List[str]) -> None:
        for ln in lines:
            log.info(ln)

    def _check_stall(self) -> None:
        """If no log file has produced output for the threshold, warn loudly."""
        if self.stall_threshold_sec <= 0:
            return
        oldest = min(f.seconds_since_progress() for f in self.followers)
        if oldest < self.stall_threshold_sec:
            self._stall_announced = False
            return
        if self._stall_announced:
            return
        self._stall_announced = True
        log.warning(
            "p%03d STALL: no new log output for %.0fs across all streams. "
            "Dumping last 50 lines of each:",
            self.sj.partition, oldest,
        )
        for f in self.followers:
            tail = f.tail_lines(50)
            log.warning(
                "p%03d %s last %d lines:\n%s",
                self.sj.partition, f.tag, len(tail), "".join(tail),
            )

    def run(self) -> None:  # noqa: D401
        """Main monitor loop."""
        log.info(
            "p%03d monitor started (job_id=%s, poll=%ds)",
            self.sj.partition, self.sj.job_id, self.poll_sec,
        )
        last_poll = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            # Poll state on cadence
            if now - last_poll >= self.poll_sec:
                state, ec, nodes = self._poll_state()
                last_poll = now
                if state and state != self.last_state:
                    log.info(
                        "p%03d STATE %s -> %s nodes=%s",
                        self.sj.partition,
                        self.last_state or "(init)", state, nodes,
                    )
                    self.last_state = state
                if state in TERMINAL_STATES:
                    self.final_state = state
                    self.exit_code = ec
                    # Drain remaining log lines before exiting.
                    for f in self.followers:
                        self._emit_lines(f.pump())
                    log.info(
                        "p%03d TERMINAL state=%s exit_code=%s",
                        self.sj.partition, state, ec,
                    )
                    if state != "COMPLETED" or (ec is not None and ec != 0):
                        self._fetch_slurmdb_record()
                    return
            # Pump tails every loop.
            for f in self.followers:
                self._emit_lines(f.pump())
            self._check_stall()
            time.sleep(1.0)
        log.info("p%03d monitor stopping (stop event set)", self.sj.partition)


# ---- Orchestration ----------------------------------------------------------
def _print_resource_plan(plan: ResourcePlan, args: argparse.Namespace) -> None:
    """Pretty-print the computed plan."""
    log.info("===== Resource plan =====")
    log.info("input_source       : %s", args.input_source)
    log.info("s3_location        : %s", args.s3_location)
    log.info("num_tiles          : %d", plan.num_tiles)
    log.info("first_tile_mb      : %.1f", plan.first_tile_mb)
    log.info("num_partitions     : %d", plan.num_partitions)
    log.info("cpus_per_task      : %d", plan.cpus_per_task)
    log.info("mem_per_cpu_mb     : %d", plan.mem_per_cpu_mb)
    log.info(
        "total_mem_per_node : %d MB",
        plan.mem_per_cpu_mb * plan.cpus_per_task,
    )
    log.info(
        "time_limit_min     : %d (~%.1f h)",
        plan.time_limit_min, plan.time_limit_min / 60,
    )
    # Sanity check: the airflow formula uses MB/hour but does not convert
    # hours to minutes, so it under-estimates wildly for big datasets.
    rough_hours = (plan.num_tiles * plan.first_tile_mb / 1024) / (
        PROCESSING_SPEED_MB_PER_HOUR
    )
    if rough_hours * 60 > plan.time_limit_min * 2:
        log.warning(
            "time_limit_min may be too low: ~%.1f h of compute estimated"
            " but only %d min requested. Consider --time-limit-min %d",
            rough_hours, plan.time_limit_min,
            int(rough_hours * 60 + TIME_BUFFER_MIN),
        )
    log.info("slurm_partition    : %s", args.slurm_partition)
    log.info("slurm_host         : %s", args.slurm_host)
    log.info("aws_profile        : %s", args.aws_profile or "(default chain)")
    log.info("=========================")


def _parse_partition_filter(spec: Optional[str], n: int) -> List[int]:
    """Parse ``--only-partitions`` like ``"7,12,19"`` or ``"0-3,7"``."""
    if not spec:
        return list(range(n))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [p for p in out if p < 0 or p >= n]
    if bad:
        raise ValueError(
            f"--only-partitions contains out-of-range ids {bad} for n={n}"
        )
    return sorted(set(out))


def submit_partitions(
    *,
    args: argparse.Namespace,
    plan: ResourcePlan,
    log_dir: Path,
    partitions: List[int],
) -> SubmissionManifest:
    """Build + submit one SLURM job per partition. Returns the manifest."""
    if args.dry_run:
        slurm_api = None
    else:
        slurm_api, _ = _make_slurm_clients(args.slurm_host)
        log.info("Pinging SLURM REST at %s ...", args.slurm_host)
        slurm_api.slurm_v0040_get_ping()
        log.info("Ping ok.")

    manifest = SubmissionManifest(
        run_id=log_dir.name,
        log_dir=str(log_dir),
        repo_dir=str(args.repo_dir),
        input_source=str(args.input_source),
        s3_location=args.s3_location,
        slurm_host=args.slurm_host,
        slurm_partition=args.slurm_partition,
        plan=asdict(plan),
        submitted_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    for p in partitions:
        js = _build_job_settings_json(
            input_source=args.input_source,
            s3_location=args.s3_location,
            num_partitions=plan.num_partitions,
            partition=p,
            output_directory=f"/tmp/hcr_partition_{p:03d}",
        )
        script = build_worker_script(
            partition=p,
            num_partitions=plan.num_partitions,
            job_settings_json=js,
            log_dir=log_dir,
            repo_dir=Path(args.repo_dir),
            python_exe=args.python_exe,
            aws_profile=args.aws_profile,
            py_spy_interval_sec=args.py_spy_interval_sec,
        )
        req = _build_submit_request(
            partition=p,
            num_partitions=plan.num_partitions,
            script=script,
            log_dir=log_dir,
            slurm_partition=args.slurm_partition,
            plan=plan,
            repo_dir=Path(args.repo_dir),
            python_exe=args.python_exe,
        )
        if args.dry_run:
            log.info("[dry-run] partition %d would submit:", p)
            log.info("[dry-run]   job_settings=%s", js)
            log.info("[dry-run]   --- script start ---\n%s", script)
            log.info("[dry-run]   --- script end ---")
            continue
        resp = slurm_api.slurm_v0040_post_job_submit(v0040_job_submit_req=req)
        job_id = str(getattr(resp, "job_id", "") or "")
        if not job_id:
            log.error("p%03d: submit returned no job_id: %r", p, resp)
            raise RuntimeError(f"submit failed for partition {p}")
        log.info("p%03d submitted job_id=%s", p, job_id)
        sj = SubmittedJob(
            partition=p,
            job_id=job_id,
            settings_json=js,
            worker_out=str(log_dir / f"worker_{p:03d}.out"),
            worker_err=str(log_dir / f"worker_{p:03d}.err"),
            pyspy_log=str(log_dir / f"worker_{p:03d}.pyspy.log"),
            slurm_out_glob=str(log_dir / f"slurm_p{p:03d}_*.out"),
            slurm_err_glob=str(log_dir / f"slurm_p{p:03d}_*.err"),
        )
        manifest.jobs.append(asdict(sj))
        _write_manifest(log_dir / "submission_manifest.json", manifest)

    return manifest


def monitor_run(
    *,
    args: argparse.Namespace,
    manifest: SubmissionManifest,
) -> int:
    """Spin up monitor threads, block until all terminate, return exit code."""
    slurm_api, slurmdb_api = _make_slurm_clients(args.slurm_host)
    stop_event = threading.Event()
    monitors: List[JobMonitor] = []
    for raw in manifest.jobs:
        sj = SubmittedJob(**raw)
        m = JobMonitor(
            sj=sj,
            slurm_api=slurm_api,
            slurmdb_api=slurmdb_api,
            poll_sec=args.poll_interval_sec,
            stall_threshold_sec=args.stall_threshold_min * 60,
            stop_event=stop_event,
        )
        monitors.append(m)
        m.start()

    def _sigint(_signum, _frame) -> None:
        log.warning("SIGINT received.")
        ans = ""
        try:
            ans = input(
                "Cancel all submitted SLURM jobs via REST? [y/N] "
            ).strip().lower()
        except EOFError:
            pass
        if ans == "y":
            for raw in manifest.jobs:
                jid = raw["job_id"]
                try:
                    slurm_api.slurm_v0040_delete_job(job_id=jid)
                    log.info("cancelled job_id=%s", jid)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "delete_job(%s) failed: %s", jid, exc
                    )
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    for m in monitors:
        m.join()

    log.info("===== Final summary =====")
    any_failed = False
    for m in monitors:
        ok = m.final_state == "COMPLETED" and (m.exit_code in (0, None))
        any_failed = any_failed or not ok
        log.info(
            "p%03d job_id=%s state=%s exit_code=%s",
            m.sj.partition, m.sj.job_id, m.final_state, m.exit_code,
        )
    log.info("=========================")
    return 1 if any_failed else 0


# ---- CLI --------------------------------------------------------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Build the argparse namespace."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    p.add_argument("--input-source", type=Path,
                   help="Path to dataset folder containing SPIM/*.czi")
    p.add_argument("--s3-location", type=str,
                   help="s3://bucket/prefix to write OME-Zarr to")
    p.add_argument("--num-partitions", type=int, default=None,
                   help=f"Defaults to min(num_tiles, {DEFAULT_MAX_PARTITIONS})")
    p.add_argument("--cpus-per-task", type=int, default=DEFAULT_CPUS_PER_TASK)
    p.add_argument("--mem-per-cpu-mb", type=int, default=None,
                   help="Override auto-computed memory per CPU (MB)")
    p.add_argument("--time-limit-min", type=int, default=None,
                   help="Override auto-computed time limit (minutes)")
    p.add_argument("--slurm-partition", type=str,
                   default=DEFAULT_SLURM_PARTITION)
    p.add_argument("--slurm-host", type=str, default=DEFAULT_SLURM_HOST)
    p.add_argument("--log-root", type=Path, default=Path(DEFAULT_LOG_ROOT))
    p.add_argument("--run-id", type=str, default=None,
                   help="Subdir under --log-root; defaults to a timestamped id")
    p.add_argument("--repo-dir", type=Path, default=Path(DEFAULT_REPO_DIR))
    p.add_argument("--python-exe", type=str, default=DEFAULT_PYTHON)
    p.add_argument("--aws-profile", type=str, default=os.environ.get(
        "AWS_PROFILE"))
    p.add_argument("--poll-interval-sec", type=int, default=15)
    p.add_argument("--stall-threshold-min", type=int, default=10,
                   help="Warn if no log progress for this many minutes (0=off)")
    p.add_argument("--py-spy-interval-sec", type=int, default=60,
                   help="Periodic stack dump interval (0 disables)")
    p.add_argument("--only-partitions", type=str, default=None,
                   help="Subset to submit, e.g. '0,3,7-9'")
    p.add_argument("--dry-run", action="store_true",
                   help="Print rendered scripts + JSON, do not submit")
    p.add_argument("--no-tail", action="store_true",
                   help="Submit but do not tail/monitor (just record manifest)")
    p.add_argument("--attach", type=str, default=None,
                   help="Attach to existing run_id (skip submit, just monitor)")
    p.add_argument("--yes", action="store_true",
                   help="Skip interactive confirmation before submitting")
    return p.parse_args(argv)


def _confirm(args: argparse.Namespace) -> bool:
    """Interactive y/N gate before live submission."""
    if args.yes or args.dry_run:
        return True
    try:
        ans = input("Submit these jobs to SLURM? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


def main(argv: Optional[List[str]] = None) -> int:
    """Entrypoint."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # ---- Attach mode: skip everything else, just monitor an existing run ---
    if args.attach:
        log_dir = args.log_root / args.attach
        _setup_logging(log_dir)
        manifest = _load_manifest(log_dir / "submission_manifest.json")
        log.info("Attaching to run_id=%s with %d jobs",
                 manifest.run_id, len(manifest.jobs))
        return monitor_run(args=args, manifest=manifest)

    # ---- Submit mode -------------------------------------------------------
    if not args.input_source or not args.s3_location:
        print(
            "ERROR: --input-source and --s3-location are required (unless"
            " --attach is used).", file=sys.stderr,
        )
        return 2

    run_id = args.run_id or (
        f"{args.input_source.name}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    log_dir = args.log_root / run_id
    _setup_logging(log_dir)

    log.info("run_id=%s", run_id)
    log.info("log_dir=%s", log_dir)

    tiles = _list_czi_tiles(args.input_source)
    plan = compute_resource_plan(
        tiles,
        num_partitions_override=args.num_partitions,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu_mb_override=args.mem_per_cpu_mb,
        time_limit_min_override=args.time_limit_min,
    )
    _print_resource_plan(plan, args)

    partitions = _parse_partition_filter(
        args.only_partitions, plan.num_partitions
    )
    if partitions != list(range(plan.num_partitions)):
        log.info("Submitting subset of partitions: %s", partitions)

    if not _confirm(args):
        log.info("Aborted by user.")
        return 1

    manifest = submit_partitions(
        args=args, plan=plan, log_dir=log_dir, partitions=partitions,
    )

    if args.dry_run:
        log.info("[dry-run] complete; no jobs submitted.")
        return 0
    if args.no_tail:
        log.info("--no-tail set; manifest at %s",
                 log_dir / "submission_manifest.json")
        log.info("Re-attach later with: --attach %s", run_id)
        return 0

    return monitor_run(args=args, manifest=manifest)


if __name__ == "__main__":
    sys.exit(main())
