"""Fetch and verify pinned embedding-model files (Milestone 3, ADR-003).

Downloads an explicit file list at an immutable revision into the Hugging Face
cache, verifies every file against the hub's own content identity (LFS SHA-256
for large files, git blob SHA-1 for small ones), and records a manifest with
local SHA-256 checksums under ``config/models/``. ``--verify`` re-checks the
cached files against the manifest offline.

Both paths fail closed: a manifest that is missing, unreadable, not valid JSON,
incomplete, has extra or duplicate entries, or disagrees with the pinned spec is
reported as a failure (exit code 1), never as verified.

Usage:
    python scripts/fetch_models.py bge-small-en-v1.5            # fetch + verify + write manifest
    python scripts/fetch_models.py bge-small-en-v1.5 --verify   # offline verification only
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "config" / "models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str  # immutable commit SHA
    files: tuple[str, ...]  # exact files; nothing else is downloaded
    download_bytes: int  # total size of ``files`` at ``revision`` (checked against the manifest)
    query_instruction: str | None  # from the model card; usage is a runtime setting
    passage_prefix: str | None
    card_notes: tuple[str, ...]


SPECS: dict[str, ModelSpec] = {
    "bge-small-en-v1.5": ModelSpec(
        name="bge-small-en-v1.5",
        repo_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        # Weights in safetensors only: no pytorch_model.bin, no ONNX, no README.
        files=(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "modules.json",
            "1_Pooling/config.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
        ),
        download_bytes=134_411_157,
        query_instruction="Represent this sentence for searching relevant passages: ",
        passage_prefix=None,
        card_notes=(
            "Model card: CLS pooling, L2-normalised embeddings, 384 dimensions, 512 tokens.",
            "Model card FAQ: for bge-*-v1.5 the query instruction is optional.",
            "ADR-003 R3: instruction OFF for the baseline; ON only as a separate experiment.",
            "No instruction or prefix is applied to passages.",
        ),
    ),
}


# ------------------------------------------------------------------ hashing


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Git object id of a file's content (what the hub reports as ``blob_id``)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324


def verify_file(path: Path, expected: dict[str, Any]) -> list[str]:
    """Problems with ``path`` against a manifest entry (empty list = verified)."""
    if not path.is_file():
        return [f"{expected['name']}: missing at {path}"]
    problems = []
    size = path.stat().st_size
    if size != expected["size"]:
        problems.append(f"{expected['name']}: size {size} != {expected['size']}")
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        problems.append(f"{expected['name']}: sha256 {digest} != {expected['sha256']}")
    return problems


def manifest_path(name: str) -> Path:
    return MANIFEST_DIR / f"{name}.json"


# ------------------------------------------------------------------ manifest

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
HUB_ID_FORMAT = {"lfs_sha256": _HEX64, "git_blob_sha1": _HEX40}


class ManifestError(Exception):
    """The manifest cannot be trusted; ``problems`` lists every reason found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _is_size(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _validate_entry(i: int, e: Any) -> tuple[str | None, int | None, list[str]]:
    """(name, size, problems) for one ``files`` entry."""
    if not isinstance(e, dict):
        return None, None, [f"manifest files[{i}]: not an object"]
    problems: list[str] = []
    name = e.get("name")
    if not isinstance(name, str) or not name:
        problems.append(f"manifest files[{i}]: 'name' missing or not a non-empty string")
        name = None
    label = f"manifest {name or f'files[{i}]'}"
    size = e.get("size")
    if not _is_size(size):
        problems.append(f"{label}: 'size' missing or not a non-negative integer")
        size = None
    sha = e.get("sha256")
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
        problems.append(f"{label}: 'sha256' missing or not 64 lower-case hex digits")
    method, hub_id = e.get("hub_verification"), e.get("hub_id")
    # isinstance first: an unhashable value (list, dict) would make `in` raise TypeError.
    if not isinstance(method, str) or method not in HUB_ID_FORMAT:
        problems.append(f"{label}: 'hub_verification' must be one of {sorted(HUB_ID_FORMAT)}")
    elif not isinstance(hub_id, str) or not HUB_ID_FORMAT[method].fullmatch(hub_id):
        problems.append(f"{label}: 'hub_id' missing or malformed for {method}")
    elif method == "lfs_sha256" and hub_id != sha:
        problems.append(f"{label}: local sha256 differs from the recorded hub LFS SHA-256")
    return name, size, problems


def validate_manifest(data: Any, spec: ModelSpec) -> list[str]:
    """Every reason ``data`` is not a complete, well-formed manifest for ``spec``."""
    if not isinstance(data, dict):
        return ["manifest: top level is not a JSON object"]
    problems: list[str] = []
    for key, expected in (
        ("name", spec.name),
        ("repo_id", spec.repo_id),
        ("revision", spec.revision),
    ):
        v = data.get(key)
        if not isinstance(v, str):
            problems.append(f"manifest: '{key}' missing or not a string")
        elif v != expected:
            problems.append(f"manifest: {key} {v!r} != pinned {expected!r}")
    for key in ("license", "retrieved_at"):
        if not isinstance(data.get(key), str) or not data[key]:
            problems.append(f"manifest: '{key}' missing or not a non-empty string")
    if not isinstance(data.get("model"), dict):
        problems.append("manifest: 'model' missing or not an object")

    files = data.get("files")
    if not isinstance(files, list):
        return [*problems, "manifest: 'files' missing or not a list"]
    names: list[str] = []
    sizes: list[int] = []
    for i, e in enumerate(files):
        name, size, entry_problems = _validate_entry(i, e)
        problems += entry_problems
        if name is not None:
            names.append(name)
        if size is not None:
            sizes.append(size)

    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(f"manifest: duplicate files {dupes}")
    if tuple(names) != spec.files:
        missing = [f for f in spec.files if f not in names]
        extra = [n for n in names if n not in spec.files]
        how = f"missing {missing}, unexpected {extra}" if missing or extra else "order differs"
        problems.append(f"manifest: files do not match the pinned list ({how})")

    total = data.get("total_bytes")
    if not _is_size(total):
        problems.append("manifest: 'total_bytes' missing or not a non-negative integer")
    else:
        if len(sizes) == len(files) and total != sum(sizes):
            problems.append(f"manifest: total_bytes {total} != sum of file sizes {sum(sizes)}")
        if total != spec.download_bytes:
            problems.append(f"manifest: total_bytes {total} != pinned {spec.download_bytes}")
    return problems


def load_manifest(spec: ModelSpec) -> dict[str, Any]:
    """Read, parse and validate the manifest once. Raises ``ManifestError``."""
    path = manifest_path(spec.name)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ManifestError([f"manifest missing at {path}"]) from None
    except (OSError, UnicodeDecodeError) as e:
        raise ManifestError([f"manifest unreadable at {path}: {e}"]) from None
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ManifestError([f"manifest is not valid JSON ({path}): {e}"]) from None
    problems = validate_manifest(data, spec)
    if problems:
        raise ManifestError(problems)
    return data  # type: ignore[no-any-return]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write to a temporary file in the same directory, fsync, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


# ------------------------------------------------------------------ fetching


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def model_description(files: dict[str, Path], spec: ModelSpec) -> dict[str, Any]:
    """Configuration recorded from the downloaded files themselves."""
    cfg = _read_json(files["config.json"])
    tok = _read_json(files["tokenizer_config.json"])
    pooling = _read_json(files["1_Pooling/config.json"])
    st = _read_json(files["sentence_bert_config.json"])
    modules = _read_json(files["modules.json"])
    return {
        "architecture": {
            k: cfg.get(k)
            for k in (
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "max_position_embeddings",
                "vocab_size",
                "torch_dtype",
            )
        },
        "tokenizer": {
            k: tok.get(k)
            for k in ("tokenizer_class", "do_lower_case", "model_max_length", "strip_accents")
        },
        "pooling": pooling,
        "max_seq_length": st.get("max_seq_length"),
        "do_lower_case": st.get("do_lower_case"),
        "modules": [m.get("type") for m in modules],
        "normalize": any("Normalize" in str(m.get("type")) for m in modules),
        "query_instruction": spec.query_instruction,
        "passage_prefix": spec.passage_prefix,
        "notes": list(spec.card_notes),
    }


def fetch(spec: ModelSpec) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if info.sha != spec.revision:
        raise SystemExit(f"hub returned revision {info.sha}, expected {spec.revision}")
    siblings = {s.rfilename: s for s in (info.siblings or [])}
    missing = [f for f in spec.files if f not in siblings]
    if missing:
        raise SystemExit(f"files not present at {spec.revision}: {missing}")

    entries, paths, problems = [], {}, []
    for name in spec.files:
        sib = siblings[name]
        local = Path(hf_hub_download(spec.repo_id, name, revision=spec.revision))
        paths[name] = local
        size, digest, blob = local.stat().st_size, sha256_file(local), git_blob_sha1(local)
        lfs_sha = getattr(sib.lfs, "sha256", None) if sib.lfs else None
        blob_id = sib.blob_id
        if sib.size is None:
            problems.append(f"{name}: hub reports no size")
        elif size != sib.size:
            problems.append(f"{name}: size {size} != hub {sib.size}")
        # Fail closed: every file must be checked against a hub content identity.
        if lfs_sha:
            method, hub_id = "lfs_sha256", lfs_sha
            if digest != lfs_sha:
                problems.append(f"{name}: sha256 {digest} != hub LFS {lfs_sha}")
        elif blob_id:
            method, hub_id = "git_blob_sha1", blob_id
            if blob != blob_id:
                problems.append(f"{name}: git blob {blob} != hub {blob_id}")
        else:
            method, hub_id = "none", ""
            problems.append(f"{name}: hub provides neither an LFS SHA-256 nor a blob id")
        entries.append(
            {
                "name": name,
                "size": size,
                "sha256": digest,
                "hub_verification": method,
                "hub_id": hub_id,
            }
        )
    if problems:
        raise SystemExit("verification failed:\n  " + "\n  ".join(problems))

    card = info.card_data.to_dict() if info.card_data else {}  # type: ignore[no-untyped-call]
    manifest = {
        "name": spec.name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": card.get("license"),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cache_dir_hint": "HF_HUB_CACHE (default ~/.cache/huggingface/hub)",
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
        "model": model_description(paths, spec),
    }
    problems = validate_manifest(manifest, spec)
    if problems:
        raise SystemExit("manifest failed validation:\n  " + "\n  ".join(problems))
    return manifest


def verify_offline(spec: ModelSpec, manifest: dict[str, Any] | None = None) -> list[str]:
    """Re-check cached files against the manifest without network access (empty = verified).

    ``manifest`` is an already-loaded manifest (from ``load_manifest``); it is still
    validated against ``spec``, so an incomplete manifest can never verify. Manifest
    errors are returned as problems, never raised. ``try_to_load_from_cache`` only
    reads the local cache (it makes no requests), so the process environment is left
    unchanged."""
    from huggingface_hub import try_to_load_from_cache

    if manifest is None:
        try:
            manifest = load_manifest(spec)
        except ManifestError as e:
            return e.problems
    else:
        invalid = validate_manifest(manifest, spec)
        if invalid:
            return invalid
    problems: list[str] = []
    for entry in manifest["files"]:
        cached = try_to_load_from_cache(spec.repo_id, entry["name"], revision=spec.revision)
        if not isinstance(cached, str):
            problems.append(f"{entry['name']}: not in local cache")
            continue
        try:
            problems += verify_file(Path(cached), entry)
        except OSError as e:
            problems.append(f"{entry['name']}: unreadable in cache: {e}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model", choices=sorted(SPECS))
    ap.add_argument("--verify", action="store_true", help="offline verification only")
    args = ap.parse_args(argv)
    spec = SPECS[args.model]
    if args.verify:
        problems = verify_offline(spec)
        print(json.dumps({"model": spec.name, "verified": not problems, "problems": problems}))
        return 1 if problems else 0
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    manifest = fetch(spec)
    write_manifest(manifest_path(spec.name), manifest)
    print(json.dumps({k: manifest[k] for k in ("name", "revision", "license", "total_bytes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
