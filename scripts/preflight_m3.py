"""Milestone 3 environment and readiness checks (ADR-003). Read-only.

Never writes, installs, downloads or opens the production database. Git is
queried with ``--no-optional-locks`` (no index refresh), no bytecode is written,
SQLite probes use an in-memory database, and the process environment is checked
to be unchanged after the checks. Exit code 0 when every check passes (warnings
allowed), 1 when any check fails.

Usage:
    python scripts/preflight_m3.py            # human-readable table
    python scripts/preflight_m3.py --json     # machine-readable result
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
_dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
import fetch_models  # noqa: E402  (no __pycache__ written for this import)

sys.dont_write_bytecode = _dont_write_bytecode

ROOT = Path(__file__).resolve().parents[1]
Status = Literal["PASS", "WARN", "FAIL", "INFO"]

REQUIRED_PINS: dict[str, str] = {"rank-bm25": "0.2.2", "networkx": "3.7", "psutil": "7.2.2"}
REQUIRED_PRESENT = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "numpy",
)
OPTIONAL = {"qdrant-client": "1.20.0"}  # declared extra; must not be needed for the core
MODEL = "bge-small-en-v1.5"
MIN_PYTHON = (3, 11)
MIN_SQLITE = (3, 35, 0)
DISK_MARGIN = 1 << 30  # 1 GiB, ADR-003
# Reference machine is 16 GiB. The approved workload (BGE-small fp32, ~128 MiB of
# weights, CPU batches, NumPy exact search) is estimated well under 1 GiB resident;
# 2 GiB available leaves headroom. Both memory conditions only WARN, never FAIL.
MIN_TOTAL_MEMORY = int(16 * 2**30 * 0.95)
MIN_AVAILABLE_MEMORY = 2 << 30
WORK_ALLOWANCE = 200 << 20  # packages + derived indexes for the benchmark corpus
SF_DATALESS = 0x40000000  # macOS: file content offloaded (iCloud "Optimize Storage")
ICLOUD_ROOTS = ("Library/Mobile Documents", "Documents", "Desktop")


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


# ------------------------------------------------------------------ checks


def check_python(version: tuple[int, int, int] = sys.version_info[:3]) -> Check:
    ok = version[:2] >= MIN_PYTHON
    return Check("python", "PASS" if ok else "FAIL", ".".join(map(str, version)))


def check_packages(versions: Callable[[str], str | None]) -> list[Check]:
    out = []
    for name, pin in REQUIRED_PINS.items():
        v = versions(name)
        ok = v == pin
        out.append(
            Check(
                f"package {name}",
                "PASS" if ok else "FAIL",
                f"installed {v or 'missing'}, pinned {pin}",
            )
        )
    for name in REQUIRED_PRESENT:
        v = versions(name)
        out.append(Check(f"package {name}", "PASS" if v else "FAIL", v or "missing"))
    for name, pin in OPTIONAL.items():
        v = versions(name)
        out.append(
            Check(
                f"optional {name}",
                "INFO",
                f"installed {v}" if v else f"not installed (optional, pin {pin})",
            )
        )
    return out


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_icloud_path(path: Path, home: Path) -> bool:
    """True for paths iCloud may sync/offload (Mobile Documents, Documents, Desktop)."""
    p = path.resolve()
    return any(p == home / r or (home / r) in p.parents for r in ICLOUD_ROOTS)


def check_locations(paths: Mapping[str, Path], home: Path) -> list[Check]:
    return [
        Check(f"location {label}", "FAIL" if is_icloud_path(p, home) else "PASS", str(p.resolve()))
        for label, p in paths.items()
    ]


def dataless_files(root: Path, flags_of: Callable[[Path], int], limit: int = 5) -> list[Path]:
    """Files whose content iCloud has offloaded (st_flags & SF_DATALESS)."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "node_modules")]
        for f in filenames:
            p = Path(dirpath) / f
            try:
                if flags_of(p) & SF_DATALESS:
                    found.append(p)
            except OSError:
                continue
            if len(found) >= limit:
                return found
    return found


def _st_flags(p: Path) -> int:
    return getattr(os.lstat(p), "st_flags", 0)


def check_disk(free_bytes: int, needed_download: int) -> Check:
    need = needed_download + WORK_ALLOWANCE + DISK_MARGIN
    status: Status = "PASS" if free_bytes >= need else "FAIL"
    return Check(
        "disk",
        status,
        f"free {free_bytes / 2**30:.2f} GiB, need {need / 2**30:.2f} GiB "
        f"(download {needed_download / 2**20:.0f} MiB + work "
        f"{WORK_ALLOWANCE >> 20} MiB + margin 1 GiB)",
    )


def check_memory(stats: tuple[int, int] | None = None) -> Check:
    """``stats`` = (total, available) bytes; read from psutil when omitted."""
    if stats is None:
        import psutil  # type: ignore[import-untyped]

        vm = psutil.virtual_memory()
        stats = (vm.total, vm.available)
    total, available = stats
    notes = []
    if total < MIN_TOTAL_MEMORY:
        notes.append("total below the 16 GiB reference machine")
    if available < MIN_AVAILABLE_MEMORY:
        notes.append(
            f"available below {MIN_AVAILABLE_MEMORY >> 30} GiB; close other applications "
            "before indexing or benchmarking"
        )
    detail = f"total {total / 2**30:.1f} GiB, available {available / 2**30:.1f} GiB"
    return Check("memory", "WARN" if notes else "PASS", "; ".join([detail, *notes]))


def check_sqlite(
    version: str = sqlite3.sqlite_version,
    connect: Callable[[str], sqlite3.Connection] = sqlite3.connect,
) -> list[Check]:
    try:
        ok = tuple(int(x) for x in version.split(".")) >= MIN_SQLITE
    except ValueError:
        ok = False
    out = [Check("sqlite version", "PASS" if ok else "FAIL", version)]
    try:
        partial_ok, trigger_ok = _sqlite_probe(connect)
    except sqlite3.Error as e:
        return [*out, Check("sqlite feature probe", "FAIL", f"probe setup failed: {e}")]
    out.append(
        Check(
            "sqlite partial unique index",
            "PASS" if partial_ok else "FAIL",
            "second ready row for one fingerprint refused" if partial_ok else "not enforced",
        )
    )
    out.append(
        Check(
            "sqlite trigger RAISE(ABORT)",
            "PASS" if trigger_ok else "FAIL",
            "BEFORE INSERT guard enforced" if trigger_ok else "not enforced",
        )
    )
    return out


def _sqlite_probe(connect: Callable[[str], sqlite3.Connection]) -> tuple[bool, bool]:
    """In-memory probe; raises ``sqlite3.Error`` if the schema can't be set up."""
    con = connect(":memory:")
    try:
        con.executescript("""
            CREATE TABLE idx(id TEXT PRIMARY KEY, fp TEXT, status TEXT);
            CREATE UNIQUE INDEX one_ready ON idx(fp) WHERE status = 'ready';
            CREATE TABLE ev(a TEXT);
            CREATE TRIGGER ev_guard BEFORE INSERT ON ev
              BEGIN SELECT CASE WHEN NEW.a IS NULL THEN RAISE(ABORT, 'guard') END; END;
            INSERT INTO idx VALUES ('i1', 'f', 'ready');
        """)
        partial_ok = trigger_ok = False
        try:
            con.execute("INSERT INTO idx VALUES ('i2', 'f', 'ready')")
        except sqlite3.IntegrityError:
            partial_ok = True
        try:
            con.execute("INSERT INTO ev VALUES (NULL)")
        except sqlite3.DatabaseError as e:
            trigger_ok = "guard" in str(e)
    finally:
        con.close()
    return partial_ok, trigger_ok


def check_torch() -> list[Check]:
    import torch

    det = hasattr(torch, "use_deterministic_algorithms")
    prec = hasattr(torch, "set_float32_matmul_precision")
    return [
        Check(
            "torch deterministic API",
            "PASS" if det and prec else "FAIL",
            f"use_deterministic_algorithms={det}, set_float32_matmul_precision={prec}",
        ),
        Check(
            "torch threads",
            "INFO",
            f"intra-op {torch.get_num_threads()}, inter-op {torch.get_num_interop_threads()} "
            "(benchmark profile fixes both)",
        ),
        Check(
            "torch MPS",
            "INFO",
            f"available={torch.backends.mps.is_available()} (benchmark uses CPU only)",
        ),
    ]


def check_blas() -> Check:
    import numpy as np

    cfg = np.show_config(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    env = {
        k: os.environ.get(k)
        for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    }
    return Check(
        "numpy BLAS", "INFO", f"{blas.get('name')} {blas.get('version')}; thread env {env}"
    )


def _problem_summary(problems: list[str], shown: int = 3) -> str:
    more = f" (+{len(problems) - shown} more)" if len(problems) > shown else ""
    return "; ".join(problems[:shown]) + more


def check_model(spec_name: str = MODEL) -> tuple[Check, int]:
    """(check, bytes to download if not verified). Offline: never contacts the network.

    The manifest is parsed and validated once and reused; any manifest or cache
    problem is a FAIL, never an exception. The download estimate is the spec's
    pinned total (a full re-fetch), not a guess."""
    spec = fetch_models.SPECS[spec_name]
    name = f"model {spec_name}"
    try:
        manifest = fetch_models.load_manifest(spec)
    except fetch_models.ManifestError as e:
        return Check(name, "FAIL", _problem_summary(e.problems)), spec.download_bytes
    problems = fetch_models.verify_offline(spec, manifest)
    if problems:
        return Check(name, "FAIL", _problem_summary(problems)), spec.download_bytes
    return Check(
        name,
        "PASS",
        f"revision {spec.revision[:12]}…, {manifest['total_bytes']} bytes, all SHA-256 verified",
    ), 0


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def check_git(root: Path, runner: Runner = subprocess.run) -> list[Check]:
    # --no-optional-locks (and GIT_OPTIONAL_LOCKS=0 for any child git) stops
    # `git status` from refreshing and rewriting .git/index.
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*args: str) -> str:
        cmd = ["git", "--no-optional-locks", "-C", str(root), *args]
        return str(runner(cmd, capture_output=True, text=True, check=True, env=env).stdout).strip()

    try:
        branch, head = git("branch", "--show-current"), git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as e:
        return [Check("git", "WARN", f"unavailable: {e}")]
    return [
        Check("git branch", "INFO", f"{branch} at {head}"),
        Check(
            "git working tree",
            "INFO",
            "clean" if not dirty else f"{len(dirty.splitlines())} changed paths",
        ),
    ]


def check_writable(path: Path) -> Check:
    target = path if path.exists() else path.parent
    ok = os.access(target, os.W_OK)
    return Check("data dir writable", "PASS" if ok else "FAIL", f"{path} (checked without writing)")


def check_ollama(url: str = "http://localhost:11434/api/ps") -> Check:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310 (fixed localhost URL)
            models = [m.get("name") for m in json.loads(r.read()).get("models", [])]
        return Check("ollama", "INFO", f"running; loaded models: {models or 'none'}")
    except OSError:
        return Check("ollama", "INFO", "not reachable (not required for retrieval)")


# ------------------------------------------------------------------ runner


def run_checks(root: Path = ROOT, home: Path | None = None) -> list[Check]:
    """Run every check; the process environment is restored and reported if any changed it."""
    before = dict(os.environ)
    try:
        checks = _run_checks(root, home)
    finally:
        after = dict(os.environ)
        if after != before:
            os.environ.clear()
            os.environ.update(before)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    checks.append(
        Check(
            "environment unchanged",
            "FAIL" if changed else "PASS",
            f"changed and restored: {changed}" if changed else "os.environ identical after checks",
        )
    )
    return checks


def _run_checks(root: Path, home: Path | None) -> list[Check]:
    home = home or Path.home()
    # The caller's environment, read before any check runs.
    offline_env = os.environ.get("HF_HUB_OFFLINE")
    hf_cache = Path(os.environ.get("HF_HUB_CACHE") or home / ".cache" / "huggingface" / "hub")
    venv = Path(sys.prefix)
    data_dir = root / "data"
    checks: list[Check] = [check_python()]
    checks += check_git(root)
    checks += check_locations(
        {"repository": root, "virtualenv": venv, "data dir": data_dir, "HF cache": hf_cache}, home
    )
    offloaded = dataless_files(root, _st_flags)
    checks.append(
        Check(
            "iCloud offloaded files",
            "FAIL" if offloaded else "PASS",
            ", ".join(map(str, offloaded)) if offloaded else "none in repository",
        )
    )
    model_check, to_download = check_model()
    checks.append(check_disk(shutil.disk_usage(root).free, to_download))
    checks.append(check_memory())
    checks += check_packages(installed_version)
    checks.append(model_check)
    checks += check_sqlite()
    checks += check_torch()
    checks.append(check_blas())
    checks.append(check_writable(data_dir))
    checks.append(Check("HF cache", "INFO", f"{hf_cache}; HF_HUB_OFFLINE={offline_env}"))
    checks.append(check_ollama())
    return checks


def summarize(checks: Iterable[Check]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    return (1 if counts.get("FAIL") else 0), counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    # Lazily imported modules (torch, numpy, psutil, huggingface_hub) must not write __pycache__.
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        checks = run_checks()
    finally:
        sys.dont_write_bytecode = saved
    code, counts = summarize(checks)
    if args.json:
        print(
            json.dumps(
                {"ok": code == 0, "counts": counts, "checks": [asdict(c) for c in checks]}, indent=2
            )
        )
    else:
        for c in checks:
            print(f"{c.status:4}  {c.name:32} {c.detail}")
        print(f"\n{'READY' if code == 0 else 'NOT READY'}: {counts}")
    return code


if __name__ == "__main__":
    sys.exit(main())
