"""SmolVLA training script for Viam custom training (T1).

Takes a Viam sequence dataset of SO-101 demonstrations and produces a SmolVLA
checkpoint that ``viam-labs:vla:policy`` can load.

Viam runs this with the sequence dataset's three Parquet files and an output
directory. Custom arguments come from ``--args key=value,...``.

Stages:
  prepare  lay out the export, copy images locally, fold the gripper into the
           arm's joints, measure fps, convert to LeRobot, check the dataset.
  train    prepare, then fine-tune smolvla_base and write the checkpoint.

Every log line from this script is tagged ``smolvla_so101`` and written as
plain ``key=value`` text. A heartbeat line every ``--heartbeat_s`` seconds says
which stage is running and how far it has got, so a quiet log means a stall.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("smolvla_so101")

# Fraction a sequence's measured frame rate may differ from the dataset median.
FPS_TOLERANCE = 0.02
FPS_MAX_OFF_FRACTION = 0.10  # more sequences off the rate than this means the rate itself is wrong
LEROBOT_COMMIT = "ff7cc3de1de830f5f3276918a013d04bdf9ea4be"
# Tried in order by --vcodec auto. Hardware first; then software H.264, much
# faster to encode than AV1; then LeRobot's default.
VCODEC_CANDIDATES = ("h264_nvenc", "h264", "libsvtav1")


def kv(**fields) -> str:
    """Plain key=value text. JSON in a log message breaks the CLI's log JSON."""
    return " ".join(f"{k}={v}" for k, v in fields.items())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # From Viam.
    p.add_argument("--binary_data_file", required=True)
    p.add_argument("--tabular_data_file", required=True)
    p.add_argument("--sequences_file", required=True)
    p.add_argument("--model_output_directory", required=True)
    # `viam training-script test-local` passes this for image datasets; a
    # sequence dataset has no dataset.jsonl, so it is accepted and ignored.
    p.add_argument("--dataset_file", default="", help=argparse.SUPPRESS)
    # Ours.
    p.add_argument("--stage", choices=("prepare", "train", "probe_io"), default="train",
                   help="probe_io: time image-copy methods on a sample, then exit")
    p.add_argument("--arm", default="follower-arm")
    p.add_argument("--gripper", default="follower-gripper",
                   help="gripper component; empty for no gripper channel")
    p.add_argument("--gripper_open", type=float, default=100.0)
    p.add_argument("--gripper_closed", type=float, default=0.0)
    p.add_argument("--cameras", default="overhead-camera wrist-camera",
                   help="space-separated; the first sets the frame clock")
    p.add_argument("--clamp_joint", default="",
                   help="INDEX:MAX_DEGREES, e.g. 3:95; empty for no clamp")
    p.add_argument("--tolerance_s", type=float, default=0.05)
    p.add_argument("--task_prefix", default="cmd: task:",
                   help="space-separated tag prefixes that mark the instruction, tried in order; "
                        "task: is what the capture-control web app writes")
    # Training defaults follow viam-sequence-to-lerobot's scripts/train_job.toml
    # (smolvla-box-bot): batch 64, AMP, 12 loader workers, 30 actions per call.
    p.add_argument("--steps", type=int, default=3750)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=12, help="lerobot-train data loader workers")
    p.add_argument("--use_amp", choices=("true", "false"), default="true",
                   help="mixed precision; true on a GPU, ignored on cpu and mps")
    p.add_argument("--n_action_steps", type=int, default=30,
                   help="actions executed per inference call (<= chunk_size 50); saved in config.json")
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="LR warmup in optimizer steps; 0 = steps / 30, as train_job.toml")
    p.add_argument("--decay_steps", type=int, default=0, help="LR decay in optimizer steps; 0 = steps")
    p.add_argument("--save_freq", type=int, default=0, help="checkpoint every N steps; 0 = steps / 3")
    p.add_argument("--lerobot_args", default="",
                   help="extra lerobot-train flags, separated by + (Viam --args splits on commas), "
                        "e.g. --policy.optimizer_lr=5e-5+--seed=7")
    p.add_argument("--log_freq", type=int, default=50, help="lerobot-train log line every N steps")
    p.add_argument("--device", default="cuda", help="cuda in a Viam job; mps or cpu locally")
    p.add_argument("--vcodec", default="auto",
                   help="video codec; auto tries " + ", ".join(VCODEC_CANDIDATES) +
                        " and uses the first that can actually encode here")
    p.add_argument("--copy_images", choices=("auto", "yes", "no"), default="auto",
                   help="copy images to local disk first; auto copies when any path is under /gcs")
    p.add_argument("--copy_threads", type=int, default=24)
    p.add_argument("--copy_method", choices=("fuse", "gcs_api", "gcs_tm"), default="gcs_tm",
                   help="fuse: copy through the /gcs mount; gcs_api: download the same "
                        "objects with the Cloud Storage client (threads); gcs_tm: its "
                        "transfer manager with worker processes. --copy_threads sets the count")
    p.add_argument("--probe_files", type=int, default=800, help="files per probe_io test")
    p.add_argument("--probe_tests", default="fuse:64+gcs_api:64+gcs_api:256",
                   help="probe_io tests, METHOD:COUNT separated by + (Viam --args splits on commas)")
    p.add_argument("--heartbeat_s", type=float, default=60.0)
    p.add_argument("--binary_root", default="",
                   help="directory relative image paths resolve against; "
                        "default: the binary Parquet file's parent's parent")
    p.add_argument("--work_dir", default="", help="default: a temporary directory")
    return p.parse_args(argv)


# --- Progress: stage timings and a heartbeat ---------------------------------

class Progress:
    """The current stage and its counters, read by the heartbeat thread."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.stage = "starting"
        self.stage_t0 = self.t0
        self.counters: dict[str, object] = {}
        self.timings: dict[str, float] = {}
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def run(self, name: str):
        with self._lock:
            self.stage, self.stage_t0, self.counters = name, time.monotonic(), {}
        logger.info(kv(event="stage_start", stage=name))
        try:
            yield self
        finally:
            dt = time.monotonic() - self.stage_t0
            self.timings[name] = round(dt, 1)
            logger.info(kv(event="stage_end", stage=name, seconds=f"{dt:.1f}",
                           **{k: v for k, v in self.counters.items()}))

    def set(self, **counters) -> None:
        with self._lock:
            self.counters.update(counters)

    def beat(self) -> str:
        with self._lock:
            now = time.monotonic()
            return kv(event="heartbeat", stage=self.stage,
                      stage_seconds=f"{now - self.stage_t0:.0f}",
                      total_seconds=f"{now - self.t0:.0f}", **self.counters)


def start_heartbeat(progress: Progress, every_s: float) -> threading.Event:
    stop = threading.Event()

    def loop():
        while not stop.wait(every_s):
            logger.info(progress.beat())

    threading.Thread(target=loop, daemon=True, name="heartbeat").start()
    return stop


# --- Copying images: through the /gcs mount, or with the Cloud Storage client ---

def gcs_object(path: str) -> tuple[str, str]:
    """/gcs/<bucket>/<object> -> (bucket, object)."""
    parts = path.split("/", 3)  # ['', 'gcs', bucket, object]
    if len(parts) < 4 or parts[1] != "gcs":
        raise ValueError(f"not a /gcs/ path: {path}")
    return parts[2], parts[3]


def copy_fuse(pairs, threads: int, on_done) -> None:
    def one(pair):
        shutil.copyfile(*pair)
        on_done(os.path.getsize(pair[1]))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, pairs))


def _gcs_client(pool: int):
    from google.cloud import storage
    from requests.adapters import HTTPAdapter

    try:
        client = storage.Client()
    except OSError:
        # No project in the environment. Downloading objects does not need one;
        # only listing buckets does.
        client = storage.Client(project="unused")
    # The client's HTTP session keeps about 10 connections by default, so more
    # threads than that just queue for one. Size the pool to the thread count.
    client._http.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
    return client


def copy_gcs_api(pairs, threads: int, on_done) -> None:
    """Threads in one process, sharing a connection pool sized to them."""
    client = _gcs_client(threads)
    buckets: dict[str, object] = {}
    lock = threading.Lock()

    def one(pair):
        bucket, name = gcs_object(pair[0])
        with lock:
            if bucket not in buckets:
                buckets[bucket] = client.bucket(bucket)
        buckets[bucket].blob(name).download_to_filename(pair[1])
        on_done(os.path.getsize(pair[1]))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, pairs))


def copy_gcs_tm(pairs, workers: int, on_done, batch: int = 5000) -> None:
    """The Cloud Storage transfer manager with worker processes, each with its own
    connections. Batched so progress is reported as it goes."""
    from google.cloud.storage import transfer_manager as tm

    client = _gcs_client(8)
    buckets: dict[str, object] = {}
    for k in range(0, len(pairs), batch):
        chunk = pairs[k:k + batch]
        blob_file = []
        for src, dst in chunk:
            bucket, name = gcs_object(src)
            if bucket not in buckets:
                buckets[bucket] = client.bucket(bucket)
            blob_file.append((buckets[bucket].blob(name), dst))
        results = tm.download_many(blob_file, max_workers=workers,
                                   worker_type=tm.PROCESS, raise_exception=True)
        failed = [r for r in results if r is not None]
        if failed:
            raise RuntimeError(f"{len(failed)} downloads failed, e.g. {failed[0]!r}")
        for _, dst in chunk:
            on_done(os.path.getsize(dst))


COPY_METHODS = {"fuse": copy_fuse, "gcs_api": copy_gcs_api, "gcs_tm": copy_gcs_tm}


def probe_io(args, binary: pd.DataFrame, work: Path) -> None:
    """Time each way of getting images onto local disk, on samples, then return."""
    gcs_mounts = []
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            dev, mnt, fstype, opts = line.split()[:4]
            if mnt.startswith("/gcs"):
                gcs_mounts.append(f"{mnt}:{fstype}:{opts}")
    except OSError:
        pass
    logger.info(kv(event="probe_env", gcs_mounts=len(gcs_mounts), cpus=os.cpu_count(),
                   google_application_credentials=bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")),
                   env_names=",".join(sorted(k for k in os.environ if k.startswith(("AIP_", "CLOUD_ML", "GOOGLE_"))))))
    for m in gcs_mounts:
        logger.info(kv(event="probe_mount", mount=m))
    try:
        import urllib.request
        req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/project/project-id",
                                     headers={"Metadata-Flavor": "Google"})
        urllib.request.urlopen(req, timeout=2).read()
        metadata = True
    except Exception:  # noqa: BLE001
        metadata = False
    logger.info(kv(event="probe_metadata_server", reachable=metadata))
    first = binary["path"].iloc[0]
    if first.startswith("/gcs/"):
        bucket, name = gcs_object(first)
        logger.info(kv(event="probe_source", bucket=bucket, example_object=name, images=len(binary)))

    n = args.probe_files
    tests_n = len(args.probe_tests.replace("+", ",").split(","))
    sample = binary["path"].sample(n=min(len(binary), n * tests_n), random_state=0).tolist()
    specs = args.probe_tests.replace("+", ",").split(",")
    tests = [(m, int(n)) for m, n in (t.split(":") for t in specs)]
    for i, (method, threads) in enumerate(tests):
        chunk = sample[i * n:(i + 1) * n]
        dest = work / "probe" / f"{method}-{threads}"
        dest.mkdir(parents=True, exist_ok=True)
        pairs = [(s, str(dest / f"{j:06d}{Path(s).suffix}")) for j, s in enumerate(chunk)]
        done = [0, 0]
        lock = threading.Lock()

        def on_done(nbytes):
            with lock:
                done[0] += 1
                done[1] += nbytes
        t0 = time.monotonic()
        try:
            COPY_METHODS[method](pairs, threads, on_done)
            ok, err = True, ""
        except Exception as exc:  # noqa: BLE001
            ok, err = False, f"{type(exc).__name__}:{str(exc)[:150]}".replace(" ", "_")
        dt = time.monotonic() - t0
        logger.info(kv(event="probe_result", method=method, threads=threads, ok=ok,
                       files=done[0], seconds=f"{dt:.1f}", files_per_s=f"{done[0] / dt:.1f}",
                       mb_per_s=f"{done[1] / 1e6 / dt:.2f}",
                       eta_full_min=f"{len(binary) / max(done[0] / dt, 1e-9) / 60:.1f}", error=err))


# --- GPU check -----------------------------------------------------------------

def probe_gpu() -> dict:
    """Log what GPU the job sees and whether PyTorch can use it."""
    import glob
    import warnings
    info = {"dev_nvidia": ",".join(sorted(Path(d).name for d in glob.glob("/dev/nvidia*"))) or "none"}
    for k in ("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES"):
        info[k.lower()] = os.environ.get(k, "unset")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = (out.stdout.strip() or out.stderr.strip()).replace(" ", "_")[:200]
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        info["driver_cuda"] = next((l.split("CUDA Version:")[1].split()[0]
                                    for l in head.splitlines() if "CUDA Version:" in l), "unknown")
    except (OSError, subprocess.SubprocessError) as exc:
        info["nvidia_smi"] = f"unavailable:{type(exc).__name__}"
    try:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int()
        lib.cuDriverGetVersion(ctypes.byref(v))
        info["libcuda_driver_version"] = v.value
    except OSError as exc:
        info["libcuda_driver_version"] = f"not_loadable:{str(exc)[:80]}".replace(" ", "_")
    import torch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
    info["torch_warning"] = " | ".join(str(w.message) for w in caught)[:300].replace(" ", "_") or "none"
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0).replace(" ", "_")
    else:
        try:
            torch.cuda.init()
        except Exception as exc:  # noqa: BLE001
            info["cuda_init_error"] = f"{type(exc).__name__}:{str(exc)[:200]}".replace(" ", "_")
    logger.info(kv(event="gpu", **info))
    return info


# --- Lay out the export, copying images locally ------------------------------

def task_prefixes(args) -> list[str]:
    prefixes = args.task_prefix.split()
    if not prefixes:
        raise SystemExit("--task_prefix needs at least one prefix")
    return prefixes


def instruction_tags(tags, prefixes: list[str]) -> list[str]:
    """Give a sequence the instruction under the first prefix, from the first prefix it has.

    The converter reads one prefix. A dataset can mix sequences tagged by hand
    (`cmd:`) with ones from the capture-control web app (`task:`), so each
    sequence's instruction is copied to the first prefix. Tags are only added.
    """
    tags = list(tags)
    first = prefixes[0]
    if any(t.startswith(first) and t[len(first):].strip() for t in tags):
        return tags
    for prefix in prefixes[1:]:
        values = [t[len(prefix):].strip() for t in tags if t.startswith(prefix) and t[len(prefix):].strip()]
        if values:
            return tags + [first + v for v in values]
    return tags


def write_sequences(src, dst: Path, prefixes: list[str], progress: Progress) -> None:
    sequences = pd.read_parquet(src)
    tagged = [instruction_tags(tags, prefixes) for tags in sequences["tags"]]
    progress.set(instruction_from_later_prefix=sum(len(new) > len(old) for new, old in zip(tagged, sequences["tags"])))
    sequences["tags"] = tagged
    sequences.to_parquet(dst, index=False)


def layout_export(args, export_dir: Path, work: Path, progress: Progress) -> pd.DataFrame:
    """Write sequences.parquet and binary_data.parquet with local image paths.

    In a Viam job the image paths are absolute under /gcs/, a Cloud Storage
    mount where every file check and read is a network round trip. The
    converter checks and reads every image one at a time, which took over an
    hour for 77k images. Copying them to local disk in parallel first makes
    both fast.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    binary = pd.read_parquet(args.binary_data_file)
    root = Path(args.binary_root) if args.binary_root else Path(args.binary_data_file).resolve().parent.parent

    def absolute(path: str) -> str:
        p = Path(path)
        return str(p if p.is_absolute() else (root / p).resolve())

    binary["path"] = binary["path"].map(absolute)
    on_gcs = binary["path"].str.startswith("/gcs/").any()
    copy = args.copy_images == "yes" or (args.copy_images == "auto" and on_gcs)
    progress.set(images=len(binary), on_gcs=bool(on_gcs), copying=copy)

    if copy:
        local = work / "images"
        local.mkdir(parents=True, exist_ok=True)
        sources = binary["path"].tolist()
        dests = [str(local / f"{i:07d}{Path(s).suffix}") for i, s in enumerate(sources)]
        done = [0, 0]  # copied, bytes
        lock = threading.Lock()

        def on_done(nbytes):
            with lock:
                done[0] += 1
                done[1] += nbytes
                if done[0] % 500 == 0 or done[0] == len(sources):
                    progress.set(copied=done[0], copied_mb=round(done[1] / 1e6))

        progress.set(copy_method=args.copy_method, copy_threads=args.copy_threads)
        COPY_METHODS[args.copy_method](list(zip(sources, dests)), args.copy_threads, on_done)
        binary["path"] = dests
        progress.set(copied=done[0], copied_mb=round(done[1] / 1e6))
    else:
        missing = [p for p in binary["path"].head(50) if not Path(p).exists()]
        if missing:
            raise SystemExit(f"image files not found, e.g. {missing[0]} (set --binary_root)")

    binary.to_parquet(export_dir / "binary_data.parquet", index=False)
    write_sequences(args.sequences_file, export_dir / "sequences.parquet", task_prefixes(args), progress)
    return binary


# --- Fold the gripper into the arm, clamp --------------------------------------

def normalize_gripper(position: np.ndarray, open_value: float, closed_value: float) -> np.ndarray:
    """0 = fully open, 1 = fully closed, as viam-labs:vla's do_command adapter reads it."""
    g = (open_value - position) / (open_value - closed_value)
    return np.clip(g, 0.0, 1.0)


def fold_gripper(args, export_dir: Path, progress: Progress) -> dict:
    tab = pd.read_parquet(args.tabular_data_file)
    arm = tab[(tab.component_name == args.arm) & (tab.method_name == "JointPositions")].copy()
    if arm.empty:
        raise SystemExit(f"no JointPositions rows for arm {args.arm!r}")
    arm["joints"] = arm.payload.map(lambda s: json.loads(s)["positions"]["values"])
    stats: dict = {"arm_rows": len(arm), "dropped_no_gripper": 0, "clamped": 0}

    if args.gripper:
        grip = tab[(tab.component_name == args.gripper) & (tab.method_name == "DoCommand")].copy()
        if grip.empty:
            raise SystemExit(f"no DoCommand rows for gripper {args.gripper!r}")
        grip["position"] = grip.payload.map(lambda s: json.loads(s)["docommand_output"]["position"])
        out = []
        for sid, a in arm.groupby("sequence_id"):
            g = grip[grip.sequence_id == sid].sort_values("timestamp")
            if g.empty:
                stats["dropped_no_gripper"] += len(a)
                continue
            m = pd.merge_asof(
                a.sort_values("timestamp"), g[["timestamp", "position"]],
                on="timestamp", direction="nearest",
                tolerance=pd.Timedelta(seconds=args.tolerance_s),
            )
            stats["dropped_no_gripper"] += int(m.position.isna().sum())
            out.append(m.dropna(subset=["position"]))
        arm = pd.concat(out)
        norm = normalize_gripper(arm.position.to_numpy(), args.gripper_open, args.gripper_closed)
        arm["joints"] = [list(j) + [float(v)] for j, v in zip(arm.joints, norm)]
        stats["gripper_min"] = round(float(norm.min()), 3)
        stats["gripper_max"] = round(float(norm.max()), 3)

    if args.clamp_joint:
        idx, limit = args.clamp_joint.split(":")
        idx, limit = int(idx), float(limit)

        def clamp(j):
            if abs(j[idx]) > limit:
                stats["clamped"] += 1
                j = list(j)
                j[idx] = float(np.sign(j[idx]) * limit)
            return j
        arm["joints"] = arm.joints.map(clamp)

    arm["payload"] = arm.joints.map(lambda j: json.dumps({"positions": {"values": j}}))
    stats["state_dim"] = len(arm.joints.iloc[0])
    arm[["sequence_id", "timestamp", "part_id", "component_name", "method_name", "payload"]].to_parquet(
        export_dir / "tabular_data.parquet", index=False
    )
    progress.set(**stats)
    return stats


# --- Measure the frame rate -----------------------------------------------------

def measure_fps(binary: pd.DataFrame, clock_camera: str, progress: Progress) -> tuple[int, dict]:
    """The dataset's frame rate, and the sequences too far off it to use.

    A sequence more than FPS_TOLERANCE off the median is usually one whose clock
    camera dropped frames; it is left out with a warning rather than failing the
    job. If more than FPS_MAX_OFF_FRACTION are off, the recording rate itself is
    inconsistent and the job stops.
    """
    rates = {}
    for sid, b in binary[binary.component_name == clock_camera].groupby("sequence_id"):
        t = b.timestamp.sort_values().astype("int64").to_numpy() / 1e6
        if len(t) > 1:
            rates[sid] = (len(t) - 1) / (t[-1] - t[0])
    median = statistics.median(rates.values())
    off = {s: r for s, r in rates.items() if abs(r - median) / median > FPS_TOLERANCE}
    if len(off) > FPS_MAX_OFF_FRACTION * len(rates):
        raise SystemExit(
            f"{len(off)} of {len(rates)} sequences more than {FPS_TOLERANCE:.0%} off the median "
            f"{median:.2f} fps, e.g. {next(iter(off))}: {next(iter(off.values())):.2f}; "
            "record every sequence at the same rate"
        )
    for sid, rate in off.items():
        logger.warning(kv(event="skipped", sequence=sid, reason=f"{rate:.2f}_fps_against_median_{median:.2f}"))
    kept = [r for s, r in rates.items() if s not in off]
    fps = round(median)
    progress.set(fps_median=f"{median:.2f}", fps_min=f"{min(kept):.2f}",
                 fps_max=f"{max(kept):.2f}", sequences=len(rates), fps=fps, skipped_fps=len(off))
    return fps, off


def drop_sequences(export_dir: Path, sequence_ids) -> None:
    """Remove sequences from the export so the converter never sees them."""
    if not sequence_ids:
        return
    path = export_dir / "sequences.parquet"
    sequences = pd.read_parquet(path)
    sequences[~sequences["sequence_id"].isin(set(sequence_ids))].to_parquet(path, index=False)


# --- Pick a video codec that actually works here -----------------------------------

def can_encode(codec: str) -> bool:
    """Encode a few tiny frames with PyAV, the library LeRobot encodes with."""
    try:
        import av
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="mp4") as out:
            stream = out.add_stream(codec, rate=20)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            for _ in range(3):
                frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        return buf.tell() > 0
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable here"
        logger.info(kv(event="codec_probe", codec=codec, usable=False,
                       reason=str(exc).replace(" ", "_")[:120]))
        return False


def choose_vcodec(requested: str, progress: Progress) -> str:
    if requested != "auto":
        progress.set(vcodec=requested, vcodec_source="requested")
        return requested
    for codec in VCODEC_CANDIDATES:
        if can_encode(codec):
            progress.set(vcodec=codec, vcodec_source="probed")
            return codec
    raise SystemExit("no usable video codec among " + ", ".join(VCODEC_CANDIDATES))


# --- Convert ------------------------------------------------------------------------------

def convert(args, export_dir: Path, dataset_root: Path, repo_id: str, fps: int, vcodec: str,
            progress: Progress):
    from viam_sequence_to_lerobot.convert import ConversionConfig, convert as run

    summary = run(ConversionConfig(
        export_dir=export_dir,
        output_root=dataset_root,
        repo_id=repo_id,
        task=None,
        task_prefix=task_prefixes(args)[0],
        camera_components=tuple(args.cameras.split()),
        arm_component=args.arm,
        action_space="joints",
        fps=fps,
        tolerance_s=args.tolerance_s,
        vcodec=vcodec,
    ))
    for sid, reason in summary.skipped:
        logger.warning(kv(event="skipped", sequence=sid, reason=reason.replace(" ", "_")))
    progress.set(episodes_written=summary.episodes_written, frames_written=summary.frames_written,
                 skipped=len(summary.skipped))
    return summary


# --- Check the dataset ------------------------------------------------------------------

def check_dataset(args, dataset_root: Path, repo_id: str, fps: int, n_sequences: int,
                  state_dim: int, skipped: int, progress: Progress) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from viam_sequence_to_lerobot.convert import camera_feature_key

    ds = LeRobotDataset(repo_id, root=dataset_root)
    feats = ds.meta.features
    expected_cams = {camera_feature_key(c) for c in args.cameras.split()}
    cams = {k for k in feats if k.startswith("observation.images.")}
    problems = []
    if list(feats["observation.state"]["shape"]) != [state_dim]:
        problems.append(f"state shape {feats['observation.state']['shape']} != {state_dim}")
    if list(feats["action"]["shape"]) != [state_dim]:
        problems.append(f"action shape {feats['action']['shape']} != {state_dim}")
    if cams != expected_cams:
        problems.append(f"camera keys {sorted(cams)} != {sorted(expected_cams)}")
    if ds.meta.fps != fps:
        problems.append(f"fps {ds.meta.fps} != {fps}")
    if ds.meta.total_episodes != n_sequences - skipped:
        problems.append(f"{ds.meta.total_episodes} episodes, expected {n_sequences - skipped}")
    tasks = sorted(ds.meta.tasks.index) if hasattr(ds.meta.tasks, "index") else sorted(ds.meta.tasks)
    report = {
        "episodes": ds.meta.total_episodes,
        "frames": ds.meta.total_frames,
        "fps": ds.meta.fps,
        "state_dim": state_dim,
        "camera_keys": sorted(cams),
        "tasks": tasks,
    }
    if problems:
        raise SystemExit("dataset check failed: " + "; ".join(problems))
    progress.set(episodes=report["episodes"], frames=report["frames"], tasks=len(tasks))
    for t in tasks:
        logger.info(kv(event="task", text=t.replace(" ", "_")))
    return report


# --- Train and write the model ------------------------------------------------------

# lerobot-train logs one line every --log_freq steps, e.g.
#   step:200 smpl:13K ep:24 epch:0.33 loss:0.128 grdn:0.900 lr:7.2e-06 data_s:0.011 ...
# Its tqdm bar redraws with \r and no newline, so a step line can arrive glued
# onto the end of a bar line: match anywhere in the line, not at its start.
# From step 1,000 LeRobot abbreviates the number ("step:1K", "step:2K" for both
# 1,600 and 2,100), so the label is kept but the exact step is counted instead:
# the n-th step line is step n * log_freq.
STEP_RE = re.compile(r"\bstep:([0-9.]+[KMB]?) smpl:")
FIELD_RE = re.compile(r"\b(epch|loss|grdn|lr|data_s|prep_s|updt_s|step_s|mem_gb):(-?[0-9.]+(?:e[-+]?[0-9]+)?)")


def parse_step_lines(line: str) -> list[dict]:
    """Return one {step, loss, ...} dict per lerobot-train step line in `line`."""
    starts = [m for m in STEP_RE.finditer(line)]
    rows = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(line)
        fields = {k: float(v) for k, v in FIELD_RE.findall(line[m.start():end])}
        if "loss" in fields:
            rows.append({"step_label": m.group(1), **fields})
    return rows


def summarize_curve(curve: list[dict]) -> dict:
    if not curve:
        return {"points": 0}
    losses = [r["loss"] for r in curve]
    best = min(curve, key=lambda r: r["loss"])
    step_s = [r["step_s"] for r in curve if "step_s" in r]
    return {"points": len(curve), "first_step": curve[0]["step"], "last_step": curve[-1]["step"],
            "first_loss": losses[0], "last_loss": losses[-1],
            "min_loss": best["loss"], "min_loss_step": best["step"],
            "median_step_s": round(statistics.median(step_s), 3) if step_s else None}


def train(args, dataset_root: Path, repo_id: str, work: Path, progress: Progress,
          gpu: dict) -> tuple[Path, list[dict]]:
    out = work / "train"
    exe = Path(sys.executable).with_name("lerobot-train")
    cmd = [
        str(exe) if exe.exists() else "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.input_features=null",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq or max(1, args.steps // 3)}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.scheduler_warmup_steps={args.warmup_steps or max(1, args.steps // 30)}",
        f"--policy.scheduler_decay_steps={args.decay_steps or args.steps}",
        f"--policy.use_amp={'true' if args.use_amp == 'true' and args.device.startswith('cuda') else 'false'}",
        "--policy.push_to_hub=false",
        f"--policy.device={args.device}",
        f"--output_dir={out}",
    ]
    cmd += [a for a in args.lerobot_args.split("+") if a]
    if args.device.startswith("cuda") and not gpu["cuda_available"]:
        raise SystemExit("--device cuda but PyTorch cannot use a GPU here (see event=gpu); "
                         "refusing to train on the CPU")
    progress.set(steps=args.steps, batch_size=args.batch_size, device=args.device,
                 use_amp=args.use_amp, num_workers=args.num_workers)
    logger.info("command: " + " ".join(cmd))
    # Forward every line to our stdout (the job log sees what it always did) and
    # keep the step lines: the job log can come back truncated, so the curve is
    # saved with the model instead.
    curve: list[dict] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        for row in parse_step_lines(line):
            row = {"step": (len(curve) + 1) * args.log_freq, **row}
            curve.append(row)
            progress.set(step=row["step"], loss=row["loss"])
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    ckpt = out / "checkpoints" / "last" / "pretrained_model"
    if not ckpt.is_dir():
        raise SystemExit(f"no checkpoint at {ckpt}")
    logger.info(kv(event="loss_summary", **summarize_curve(curve)))
    return ckpt, curve


def write_model(ckpt: Path, output: Path, provenance: dict, curve: list[dict],
                progress: Progress) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for f in ckpt.iterdir():
        dest = output / f.name
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, dest)
    (output / "loss_curve.json").write_text(json.dumps(curve, indent=1))
    provenance = {**provenance, "loss_summary": summarize_curve(curve)}
    (output / "viam_training.json").write_text(json.dumps(provenance, indent=2))
    files = sorted(p.name for p in output.iterdir())
    progress.set(files=len(files), model_mb=round(sum(
        p.stat().st_size for p in output.rglob("*") if p.is_file()) / 1e6))
    logger.info(kv(event="model_written", path=output, files=",".join(files)))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    progress = Progress()
    stop = start_heartbeat(progress, args.heartbeat_s)
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="smolvla-so101-"))
    export_dir, dataset_root, repo_id = work / "export", work / "lerobot", "local/so101-dataset"
    for d in (dataset_root, export_dir):
        if d.exists():
            shutil.rmtree(d)
    logger.info(kv(event="start", stage_arg=args.stage, work=work, cpus=os.cpu_count()))
    logger.info(kv(event="inputs", binary_data_file=args.binary_data_file,
                   tabular_data_file=args.tabular_data_file, sequences_file=args.sequences_file,
                   model_output_directory=args.model_output_directory))
    gpu = probe_gpu()

    try:
        if args.stage == "probe_io":
            with progress.run("probe_io"):
                binary = pd.read_parquet(args.binary_data_file)
                root = Path(args.binary_root) if args.binary_root else Path(args.binary_data_file).resolve().parent.parent
                binary["path"] = binary["path"].map(
                    lambda p: p if p.startswith("/") else str((root / p).resolve()))
                probe_io(args, binary, work)
            return 0
        with progress.run("layout"):
            binary = layout_export(args, export_dir, work, progress)
        with progress.run("fold_gripper"):
            fold = fold_gripper(args, export_dir, progress)
        with progress.run("measure_fps"):
            fps, off_rate = measure_fps(binary, args.cameras.split()[0], progress)
            drop_sequences(export_dir, off_rate)
        with progress.run("choose_vcodec"):
            vcodec = choose_vcodec(args.vcodec, progress)
        with progress.run("convert"):
            summary = convert(args, export_dir, dataset_root, repo_id, fps, vcodec, progress)
        with progress.run("check_dataset"):
            n_sequences = len(pd.read_parquet(export_dir / "sequences.parquet"))
            report = check_dataset(args, dataset_root, repo_id, fps, n_sequences,
                                   fold["state_dim"], len(summary.skipped), progress)
        provenance = {
            **report,
            "gripper": {"component": args.gripper, "open_value": args.gripper_open,
                        "closed_value": args.gripper_closed, "encoding": "0 = open, 1 = closed"},
            "clamp_joint": args.clamp_joint or None,
            "vcodec": vcodec,
            "skipped_sequences": [list(s) for s in summary.skipped]
                                 + [[sid, f"{rate:.2f} fps, off the median"] for sid, rate in off_rate.items()],
            "fold": fold,
            "lerobot_commit": LEROBOT_COMMIT,
            "training": {k: getattr(args, k) for k in (
                "steps", "batch_size", "num_workers", "use_amp", "n_action_steps",
                "warmup_steps", "decay_steps", "save_freq", "lerobot_args")},
            "stage_seconds": progress.timings,
        }
        (work / "dataset_report.json").write_text(json.dumps(provenance, indent=2))
        if args.stage == "prepare":
            logger.info(kv(event="done", stage_arg="prepare", **progress.timings))
            return 0
        with progress.run("train"):
            ckpt, curve = train(args, dataset_root, repo_id, work, progress, gpu)
        provenance["stage_seconds"] = progress.timings
        provenance["steps"], provenance["batch_size"] = args.steps, args.batch_size
        with progress.run("write_model"):
            write_model(ckpt, Path(args.model_output_directory), provenance, curve, progress)
        logger.info(kv(event="done", stage_arg="train", **progress.timings))
        return 0
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
