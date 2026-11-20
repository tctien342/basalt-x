"""Optional, independently qualified overlap settings for the next worker start."""

import hashlib
import json
import os
from pathlib import Path
import re
import time

from serve.engine.flags import flag_value

MAX_BYTES = 4096
TTL_S = 7 * 24 * 60 * 60
SCOPE = re.compile(r"[0-9a-f]{16}")


def fingerprint(engine_bin, args, cwd, env):
    """Bind hints to exact native code, launch settings and the local model files."""
    relative_to = Path(cwd or ".")
    binary = relative_to / engine_bin
    with binary.open("rb") as stream:
        executable = hashlib.file_digest(stream, "sha256").hexdigest()
    normalized = []
    args = iter(args)
    for arg in args:
        if arg in {"--conversation-cache-dir", "--expert-profile-save"}:
            next(args, None)
        elif arg == "--exp":
            experiments = (next(args, None) or "").split(",")
            normalized += [arg, ",".join(setting for setting in experiments if setting.split("=", 1)[0] not in {
                "tuner_overlap_trial_chunk", "tuner_overlap_gap_ms",
            })]
        else:
            normalized.append(arg)
    files = []
    for flag in ("--model", "--atomic-lora", "--swift-lora", "--expert-profile"):
        path = flag_value(normalized, flag)
        if path:
            info = (relative_to / path).stat()
            files.append((flag, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    system = []
    for path in ("/etc/machine-id", "/proc/driver/nvidia/version"):
        try:
            system.append(Path(path).read_text())
        except OSError:
            system.append("")
    settings = sorted((k, v) for k, v in env.items() if k.startswith("CUDA_") or k in {
        "BASALT_SHARED_KV_INCREMENTAL", "BASALT_SHARED_SOLO", "BASALT_DIRECT_PEER",
        "BASALT_GPU1_VISION_OVERLAP",
    })
    return hashlib.sha256(json.dumps([1, executable, normalized, files, system, settings]).encode()).hexdigest()


def read(path, identity, now=None):
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            return None
        row = json.loads(payload)
        if set(row) != {"version", "fingerprint", "scope", "chunk", "gap_ms", "verified_at"}:
            return None
        if type(row["version"]) is not int or row["version"] != 1 or row["fingerprint"] != identity:
            return None
        scope, chunk, gap, verified = (row[field] for field in ("scope", "chunk", "gap_ms", "verified_at"))
        now = int(time.time()) if now is None else now
        if (not SCOPE.fullmatch(scope) or not int(scope, 16) or type(chunk) is not int or
            chunk <= 0 or chunk > 32768 or chunk % 256 or type(gap) is not int or not 0 <= gap <= 60000 or
            type(verified) is not int or not 0 <= now - verified < TTL_S):
            return None
        return row
    except (OSError, TypeError, ValueError, KeyError):
        return None


def load(engine_bin, args, cwd, env):
    launch_env = os.environ if env is None else env
    if not any(name in launch_env for name in ("BASALT_OVERLAP_HINT_FILE", "BASALT_NATIVE_OVERLAP_HINT")):
        return env, None
    child_env = dict(launch_env)
    child_env.pop("BASALT_NATIVE_OVERLAP_HINT", None)
    source = child_env.pop("BASALT_OVERLAP_HINT_FILE", "")
    if not source or os.name != "posix":
        return child_env, None
    path = Path(source)
    if not path.is_absolute():
        return child_env, None
    try:
        identity = fingerprint(engine_bin, args, cwd, child_env)
    except OSError:
        return child_env, None
    row = read(path, identity)
    print(f"[server] overlap hint fingerprint {identity} {'loaded' if row else 'missing or incompatible'}", flush=True)
    if row is None:
        return child_env, None
    child_env["BASALT_NATIVE_OVERLAP_HINT"] = ",".join(str(value) for value in (
        row["scope"], row["chunk"], row["gap_ms"], row["verified_at"] + TTL_S,
    ))
    return child_env, (path, row)


def revoke(path, row):
    """Do not erase a replacement written after this worker was launched."""
    import fcntl

    try:
        with path.with_name(path.name + ".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if read(path, row["fingerprint"]) == row:
                path.unlink()
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
    except OSError as error:
        print(f"[server] overlap hint revocation failed: {error}", flush=True)
