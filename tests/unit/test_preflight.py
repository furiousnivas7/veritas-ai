"""P0 tooling: preflight checks and model-file verification (no network, no models)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fetch_models
import preflight_m3 as pf
import pytest

# ------------------------------------------------------------- fetch helpers


def test_sha256_and_git_blob_sha1(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_bytes(b'{"a": 1}\n')
    assert fetch_models.sha256_file(p) == hashlib.sha256(b'{"a": 1}\n').hexdigest()
    # git hash-object of the same bytes
    assert fetch_models.git_blob_sha1(p) == hashlib.sha1(b'blob 9\x00{"a": 1}\n').hexdigest()


def test_verify_file_detects_size_and_checksum_problems(tmp_path: Path) -> None:
    p = tmp_path / "w.bin"
    p.write_bytes(b"weights")
    good = {"name": "w.bin", "size": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}
    assert fetch_models.verify_file(p, good) == []
    assert any("size" in x for x in fetch_models.verify_file(p, {**good, "size": 8}))
    assert any("sha256" in x for x in fetch_models.verify_file(p, {**good, "sha256": "0" * 64}))
    assert any("missing" in x for x in fetch_models.verify_file(tmp_path / "nope", good))


def test_bge_spec_is_pinned_and_excludes_duplicate_formats() -> None:
    spec = fetch_models.SPECS["bge-small-en-v1.5"]
    assert spec.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert "model.safetensors" in spec.files
    assert not [f for f in spec.files if f.endswith((".bin", ".onnx")) or f.startswith("onnx/")]
    assert spec.passage_prefix is None


def test_committed_manifest_matches_spec() -> None:
    m = json.loads(fetch_models.manifest_path("bge-small-en-v1.5").read_text())
    spec = fetch_models.SPECS["bge-small-en-v1.5"]
    assert m["revision"] == spec.revision and m["license"] == "mit"
    assert [f["name"] for f in m["files"]] == list(spec.files)
    assert m["total_bytes"] == sum(f["size"] for f in m["files"])
    assert all(len(f["sha256"]) == 64 for f in m["files"])
    assert m["model"]["pooling"]["pooling_mode_cls_token"] is True
    assert m["model"]["normalize"] is True
    assert m["model"]["architecture"]["hidden_size"] == 384


# ----------------------------------------------------------- preflight checks


def test_python_check() -> None:
    assert pf.check_python((3, 12, 11)).status == "PASS"
    assert pf.check_python((3, 10, 9)).status == "FAIL"


def test_package_pins_are_exact() -> None:
    def versions(installed: dict[str, str]):  # type: ignore[no-untyped-def]
        return lambda name: installed.get(name)

    present = dict.fromkeys(pf.REQUIRED_PRESENT, "1.0")
    ok = {**pf.REQUIRED_PINS, **present}
    assert all(c.status != "FAIL" for c in pf.check_packages(versions(ok)))
    wrong = {**ok, "rank-bm25": "0.2.1"}
    assert [c.name for c in pf.check_packages(versions(wrong)) if c.status == "FAIL"] == [
        "package rank-bm25"
    ]
    missing = {k: v for k, v in ok.items() if k != "networkx"}
    assert "package networkx" in {
        c.name for c in pf.check_packages(versions(missing)) if c.status == "FAIL"
    }


def test_optional_qdrant_is_never_a_failure() -> None:
    checks = pf.check_packages(lambda name: None)
    (q,) = [c for c in checks if c.name == "optional qdrant-client"]
    assert q.status == "INFO" and "not installed" in q.detail


@pytest.mark.parametrize(
    ("rel", "icloud"),
    [
        ("Documents/GitHub/veritas-ai", True),
        ("Desktop/x", True),
        ("Library/Mobile Documents/com~apple~CloudDocs/x", True),
        ("dev/veritas-ai", False),
        (".cache/huggingface/hub", False),
        ("Documentsx/repo", False),
    ],
)
def test_icloud_path_detection(tmp_path: Path, rel: str, icloud: bool) -> None:
    home = tmp_path
    target = home / rel
    target.mkdir(parents=True)
    assert pf.is_icloud_path(target, home) is icloud


def test_dataless_scan_uses_flags(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    flags = {tmp_path / "b.txt": pf.SF_DATALESS}
    found = pf.dataless_files(tmp_path, lambda p: flags.get(p, 0))
    assert found == [tmp_path / "b.txt"]
    assert pf.dataless_files(tmp_path, lambda p: 0) == []


def test_disk_check_requires_margin() -> None:
    need = 135 << 20
    enough = need + pf.WORK_ALLOWANCE + pf.DISK_MARGIN
    assert pf.check_disk(enough, need).status == "PASS"
    assert pf.check_disk(enough - 1, need).status == "FAIL"


def test_sqlite_feature_probe_passes() -> None:
    assert all(c.status == "PASS" for c in pf.check_sqlite())
    assert pf.check_sqlite("3.34.1")[0].status == "FAIL"


def test_writable_check_does_not_write(tmp_path: Path) -> None:
    before = sorted(tmp_path.iterdir())
    assert pf.check_writable(tmp_path / "data").status == "PASS"
    assert sorted(tmp_path.iterdir()) == before


def test_summary_fails_only_on_fail() -> None:
    ok = [pf.Check("a", "PASS", ""), pf.Check("b", "INFO", ""), pf.Check("c", "WARN", "")]
    assert pf.summarize(ok)[0] == 0
    assert pf.summarize([*ok, pf.Check("d", "FAIL", "")])[0] == 1


def test_model_check_without_manifest_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fetch_models, "MANIFEST_DIR", tmp_path)
    check, to_download = pf.check_model()
    assert check.status == "FAIL" and to_download > 0


# ===================================================== P0 audit regressions
# Everything below uses a small fake model, a fake hub and a fake cache: no network.

FAKE_FILES: dict[str, bytes] = {
    "config.json": b'{"model_type": "bert", "hidden_size": 4}',
    "model.safetensors": b"\x00fake-weights\x01",
    "tokenizer_config.json": b'{"tokenizer_class": "BertTokenizer"}',
    "modules.json": b'[{"type": "sentence_transformers.models.Normalize"}]',
    "1_Pooling/config.json": b'{"pooling_mode_cls_token": true}',
    "sentence_bert_config.json": b'{"max_seq_length": 8}',
}
LFS_FILES = {"model.safetensors"}


@pytest.fixture
def fake_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[fetch_models.ModelSpec, Path]:
    import huggingface_hub

    cache = tmp_path / "cache"
    for name, data in FAKE_FILES.items():
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(data)
    spec = fetch_models.ModelSpec(
        name="fake-model",
        repo_id="org/fake-model",
        revision="a" * 40,
        files=tuple(FAKE_FILES),
        download_bytes=sum(len(d) for d in FAKE_FILES.values()),
        query_instruction=None,
        passage_prefix=None,
        card_notes=(),
    )
    monkeypatch.setitem(fetch_models.SPECS, spec.name, spec)
    monkeypatch.setattr(fetch_models, "MANIFEST_DIR", tmp_path / "manifests")

    def cached(repo_id: str, name: str, revision: str | None = None) -> str | None:
        assert (repo_id, revision) == (spec.repo_id, spec.revision)
        return str(cache / name) if (cache / name).is_file() else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    return spec, cache


def good_manifest(spec: fetch_models.ModelSpec, cache: Path) -> dict:  # type: ignore[type-arg]
    files = []
    for name in spec.files:
        p = cache / name
        sha = fetch_models.sha256_file(p)
        lfs = name in LFS_FILES
        files.append(
            {
                "name": name,
                "size": p.stat().st_size,
                "sha256": sha,
                "hub_verification": "lfs_sha256" if lfs else "git_blob_sha1",
                "hub_id": sha if lfs else fetch_models.git_blob_sha1(p),
            }
        )
    return {
        "name": spec.name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "license": "mit",
        "retrieved_at": "2026-11-28T00:00:00+00:00",
        "total_bytes": sum(f["size"] for f in files),
        "files": files,
        "model": {},
    }


def write_raw_manifest(spec: fetch_models.ModelSpec, text: str) -> None:
    path = fetch_models.manifest_path(spec.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_spec_download_bytes_matches_committed_manifest() -> None:
    spec = fetch_models.SPECS["bge-small-en-v1.5"]
    assert fetch_models.load_manifest(spec)["total_bytes"] == spec.download_bytes


def test_good_fake_manifest_validates(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, cache = fake_model
    assert fetch_models.validate_manifest(good_manifest(spec, cache), spec) == []


def _truncate(m: dict) -> None:  # type: ignore[type-arg]
    m["files"].pop()
    m["total_bytes"] = sum(f["size"] for f in m["files"])


def _extra(m: dict) -> None:  # type: ignore[type-arg]
    m["files"].append({**m["files"][0], "name": "pytorch_model.bin"})


def _duplicate(m: dict) -> None:  # type: ignore[type-arg]
    m["files"][-1] = dict(m["files"][0])


def _reorder(m: dict) -> None:  # type: ignore[type-arg]
    m["files"].reverse()


def _set(path: tuple[object, ...], value: object):  # type: ignore[no-untyped-def]
    def mutate(m: dict) -> None:  # type: ignore[type-arg]
        target = m
        for key in path[:-1]:
            target = target[key]  # type: ignore[index]
        target[path[-1]] = value  # type: ignore[index]

    return mutate


def _delete(key: str):  # type: ignore[no-untyped-def]
    return lambda m: m.pop(key)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_truncate, "do not match the pinned list"),
        (_extra, "unexpected ['pytorch_model.bin']"),
        (_duplicate, "duplicate files"),
        (_reorder, "order differs"),
        (_set(("revision",), "b" * 40), "revision"),
        (_set(("repo_id",), "org/other"), "repo_id"),
        (_set(("total_bytes",), 1), "total_bytes"),
        (_set(("total_bytes",), "34"), "total_bytes"),
        (_set(("files", 0, "size"), "41"), "'size'"),
        (_set(("files", 0, "size"), True), "'size'"),
        (_set(("files", 0, "size"), -1), "'size'"),
        (_set(("files", 0, "sha256"), "XYZ"), "'sha256'"),
        (_set(("files", 0, "hub_verification"), "none"), "'hub_verification'"),
        (_set(("files", 0, "hub_id"), None), "'hub_id'"),
        (_set(("files", 1, "hub_id"), "0" * 64), "hub LFS SHA-256"),
        (_set(("files", 0), "config.json"), "not an object"),
        (_set(("files",), {}), "'files'"),
        (_delete("files"), "'files'"),
        (_delete("license"), "'license'"),
        (_delete("model"), "'model'"),
    ],
)
def test_incomplete_or_malformed_manifest_never_validates(  # type: ignore[no-untyped-def]
    fake_model, mutate, expected: str
) -> None:
    spec, cache = fake_model
    m = good_manifest(spec, cache)
    mutate(m)
    problems = fetch_models.validate_manifest(m, spec)
    assert any(expected in p for p in problems), problems
    # ...and the same manifest on disk never verifies offline.
    write_raw_manifest(spec, json.dumps(m))
    assert fetch_models.verify_offline(spec)


def test_non_object_manifest_is_rejected(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, _ = fake_model
    assert fetch_models.validate_manifest([], spec) == ["manifest: top level is not a JSON object"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, "manifest missing"),
        ("{not json", "not valid JSON"),
        ("", "not valid JSON"),
        ("[]", "top level is not a JSON object"),
    ],
)
def test_unreadable_manifest_is_reported_not_raised(  # type: ignore[no-untyped-def]
    fake_model, text: str | None, expected: str
) -> None:
    spec, _ = fake_model
    if text is not None:
        write_raw_manifest(spec, text)
    with pytest.raises(fetch_models.ManifestError) as err:
        fetch_models.load_manifest(spec)
    assert expected in str(err.value)
    problems = fetch_models.verify_offline(spec)
    assert any(expected in p for p in problems)


def test_non_utf8_manifest_is_reported(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, _ = fake_model
    path = fetch_models.manifest_path(spec.name)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe{")
    with pytest.raises(fetch_models.ManifestError) as err:
        fetch_models.load_manifest(spec)
    (message,) = err.value.problems
    assert message.startswith("manifest unreadable at ") and "'utf-8' codec can't decode" in message
    assert fetch_models.verify_offline(spec) == [message]


@pytest.mark.parametrize("method", [[], {}, True, None], ids=["list", "dict", "bool", "none"])
def test_malformed_hub_verification_is_a_problem_not_a_crash(  # type: ignore[no-untyped-def]
    fake_model, method: object
) -> None:
    spec, cache = fake_model
    m = good_manifest(spec, cache)
    m["files"][0]["hub_verification"] = method
    problems = fetch_models.validate_manifest(m, spec)
    assert problems == [
        "manifest config.json: 'hub_verification' must be one of ['git_blob_sha1', 'lfs_sha256']"
    ]
    write_raw_manifest(spec, json.dumps(m))
    assert fetch_models.verify_offline(spec) == problems


def test_verify_offline_passes_on_intact_cache(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, cache = fake_model
    write_raw_manifest(spec, json.dumps(good_manifest(spec, cache)))
    assert fetch_models.verify_offline(spec) == []


def test_verify_offline_reports_missing_cached_file(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, cache = fake_model
    write_raw_manifest(spec, json.dumps(good_manifest(spec, cache)))
    (cache / "vocab.txt").unlink(missing_ok=True)
    (cache / "tokenizer_config.json").unlink()
    assert fetch_models.verify_offline(spec) == ["tokenizer_config.json: not in local cache"]


def test_verify_offline_reports_corrupted_cached_file(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, cache = fake_model
    write_raw_manifest(spec, json.dumps(good_manifest(spec, cache)))
    weights = cache / "model.safetensors"
    data = bytearray(weights.read_bytes())
    data[0] ^= 0xFF  # same size, different content
    weights.write_bytes(bytes(data))
    problems = fetch_models.verify_offline(spec)
    assert len(problems) == 1 and problems[0].startswith("model.safetensors: sha256")


def test_verify_offline_rejects_a_passed_in_invalid_manifest(fake_model) -> None:  # type: ignore[no-untyped-def]
    spec, cache = fake_model
    m = good_manifest(spec, cache)
    _truncate(m)
    assert fetch_models.verify_offline(spec, m)


def test_verify_offline_leaves_environment_unchanged(fake_model) -> None:  # type: ignore[no-untyped-def]
    import os

    spec, cache = fake_model
    write_raw_manifest(spec, json.dumps(good_manifest(spec, cache)))
    before = dict(os.environ)
    fetch_models.verify_offline(spec)
    assert dict(os.environ) == before


def test_verify_cli_exit_codes(fake_model, capsys) -> None:  # type: ignore[no-untyped-def]
    spec, cache = fake_model
    write_raw_manifest(spec, json.dumps(good_manifest(spec, cache)))
    assert fetch_models.main([spec.name, "--verify"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True

    write_raw_manifest(spec, "{broken")
    assert fetch_models.main([spec.name, "--verify"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["verified"] is False and "not valid JSON" in out["problems"][0]


# ---------------------------------------------------------------- fake hub


class FakeLfs:
    def __init__(self, sha256: str) -> None:
        self.sha256 = sha256


class FakeSibling:
    def __init__(self, path: Path, name: str) -> None:
        self.rfilename = name
        self.size: int | None = path.stat().st_size
        lfs = name in LFS_FILES
        self.lfs = FakeLfs(fetch_models.sha256_file(path)) if lfs else None
        self.blob_id: str | None = None if lfs else fetch_models.git_blob_sha1(path)


class FakeCard:
    def to_dict(self) -> dict[str, str]:
        return {"license": "mit"}


class FakeInfo:
    def __init__(self, sha: str, siblings: list[FakeSibling]) -> None:
        self.sha = sha
        self.siblings = siblings
        self.card_data = FakeCard()


@pytest.fixture
def fake_hub(fake_model, monkeypatch):  # type: ignore[no-untyped-def]
    """(spec, info): the fake hub serves ``info`` and files from the fake cache."""
    import huggingface_hub

    spec, cache = fake_model
    info = FakeInfo(spec.revision, [FakeSibling(cache / n, n) for n in spec.files])
    downloads: list[str] = []

    class FakeApi:
        def model_info(self, repo_id: str, revision: str, files_metadata: bool) -> FakeInfo:
            assert (repo_id, revision, files_metadata) == (spec.repo_id, spec.revision, True)
            return info

    def download(repo_id: str, filename: str, revision: str) -> str:
        assert (repo_id, revision) == (spec.repo_id, spec.revision)
        downloads.append(filename)
        return str(cache / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return spec, info, downloads


def test_fetch_downloads_only_the_allowlist_and_writes_a_valid_manifest(fake_hub) -> None:  # type: ignore[no-untyped-def]
    spec, _, downloads = fake_hub
    assert fetch_models.main([spec.name]) == 0
    assert downloads == list(spec.files)
    manifest = fetch_models.load_manifest(spec)
    assert [f["name"] for f in manifest["files"]] == list(spec.files)
    leftovers = [p.name for p in fetch_models.MANIFEST_DIR.iterdir()]
    assert leftovers == [f"{spec.name}.json"]  # no temporary file left behind


def _sib(info: FakeInfo, name: str) -> FakeSibling:
    (s,) = [s for s in info.siblings if s.rfilename == name]
    return s


@pytest.mark.parametrize(
    ("break_hub", "expected"),
    [
        (lambda i: setattr(i, "sha", "b" * 40), "hub returned revision"),
        (lambda i: i.siblings.pop(), "files not present"),
        (lambda i: setattr(_sib(i, "config.json"), "size", 1), "config.json: size"),
        (lambda i: setattr(_sib(i, "config.json"), "size", None), "hub reports no size"),
        (
            lambda i: setattr(_sib(i, "model.safetensors"), "lfs", FakeLfs("0" * 64)),
            "model.safetensors: sha256",
        ),
        (lambda i: setattr(_sib(i, "modules.json"), "blob_id", "0" * 40), "git blob"),
        (lambda i: setattr(_sib(i, "modules.json"), "blob_id", None), "neither an LFS"),
        (lambda i: setattr(_sib(i, "model.safetensors"), "lfs", FakeLfs("")), "neither an LFS"),
    ],
)
def test_fetch_fails_closed_on_hub_mismatch(fake_hub, break_hub, expected: str) -> None:  # type: ignore[no-untyped-def]
    spec, info, _ = fake_hub
    break_hub(info)
    with pytest.raises(SystemExit, match=expected):
        fetch_models.main([spec.name])
    assert not fetch_models.manifest_path(spec.name).exists()  # nothing recorded


def test_manifest_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    path = tmp_path / "m.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    def fail(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        fetch_models.write_manifest(path, {"new": True})
    assert path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


# ------------------------------------------------------- preflight hardening


def test_model_check_fails_on_corrupt_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fetch_models, "MANIFEST_DIR", tmp_path)
    (tmp_path / "bge-small-en-v1.5.json").write_text("{corrupt", encoding="utf-8")
    check, to_download = pf.check_model()
    spec = fetch_models.SPECS["bge-small-en-v1.5"]
    assert check.status == "FAIL" and "not valid JSON" in check.detail
    assert to_download == spec.download_bytes


def test_model_check_fails_on_truncated_real_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = json.loads(fetch_models.manifest_path("bge-small-en-v1.5").read_text("utf-8"))
    _truncate(real)
    monkeypatch.setattr(fetch_models, "MANIFEST_DIR", tmp_path)
    (tmp_path / "bge-small-en-v1.5.json").write_text(json.dumps(real), encoding="utf-8")
    check, _ = pf.check_model()
    assert check.status == "FAIL" and "do not match the pinned list" in check.detail


def test_model_check_without_manifest_uses_spec_download_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fetch_models, "MANIFEST_DIR", tmp_path)
    _, to_download = pf.check_model()
    assert to_download == fetch_models.SPECS["bge-small-en-v1.5"].download_bytes


def test_git_check_never_takes_optional_locks(tmp_path: Path) -> None:
    import subprocess

    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, kw["env"]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(cmd, 0, stdout="x\n", stderr="")

    checks = pf.check_git(tmp_path, runner=runner)
    assert [c.status for c in checks] == ["INFO", "INFO"]
    assert ("status", "--porcelain") in {tuple(cmd[4:]) for cmd, _ in calls}
    for cmd, env in calls:
        assert cmd[:2] == ["git", "--no-optional-locks"]
        assert env["GIT_OPTIONAL_LOCKS"] == "0"


def test_sqlite_setup_errors_are_reported_as_fail() -> None:
    import sqlite3

    class Broken:
        def executescript(self, script: str) -> None:
            raise sqlite3.OperationalError("no such module")

        def close(self) -> None:
            pass

    checks = pf.check_sqlite(connect=lambda _: Broken())  # type: ignore[arg-type,return-value]
    assert checks[-1].status == "FAIL" and "probe setup failed" in checks[-1].detail

    def cannot_open(_: str) -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open")

    assert pf.check_sqlite(connect=cannot_open)[-1].status == "FAIL"
    assert pf.check_sqlite("not-a-version")[0].status == "FAIL"


@pytest.mark.parametrize(
    ("total_gib", "available_gib", "status"),
    [(16, 8, "PASS"), (16, 3.9, "PASS"), (16, 1.5, "WARN"), (8, 6, "WARN"), (4, 0.5, "WARN")],
)
def test_memory_check_warns_but_never_fails(
    total_gib: float, available_gib: float, status: str
) -> None:
    check = pf.check_memory((int(total_gib * 2**30), int(available_gib * 2**30)))
    assert check.status == status


def test_run_checks_leaves_environment_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setattr(pf, "check_ollama", lambda: pf.Check("ollama", "INFO", "stubbed"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    before = dict(os.environ)
    checks = pf.run_checks()
    assert dict(os.environ) == before
    (env_check,) = [c for c in checks if c.name == "environment unchanged"]
    assert env_check.status == "PASS"
    (hf,) = [c for c in checks if c.name == "HF cache"]
    assert hf.detail.endswith("HF_HUB_OFFLINE=None")


def test_run_checks_restores_and_reports_a_mutated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    def mutating() -> pf.Check:
        os.environ["VERITAS_PREFLIGHT_CANARY"] = "1"
        return pf.Check("numpy BLAS", "INFO", "stub")

    monkeypatch.setattr(pf, "check_ollama", lambda: pf.Check("ollama", "INFO", "stubbed"))
    monkeypatch.setattr(pf, "check_blas", mutating)
    monkeypatch.delenv("VERITAS_PREFLIGHT_CANARY", raising=False)
    checks = pf.run_checks()
    assert "VERITAS_PREFLIGHT_CANARY" not in os.environ
    (env_check,) = [c for c in checks if c.name == "environment unchanged"]
    assert env_check.status == "FAIL" and "VERITAS_PREFLIGHT_CANARY" in env_check.detail


@pytest.mark.parametrize(
    ("checks", "code"),
    [
        ([pf.Check("a", "PASS", ""), pf.Check("b", "WARN", "")], 0),
        ([pf.Check("a", "PASS", ""), pf.Check("b", "FAIL", "x")], 1),
    ],
)
def test_preflight_json_exit_codes(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, capsys, checks: list[pf.Check], code: int
) -> None:
    monkeypatch.setattr(pf, "run_checks", lambda: checks)
    assert pf.main(["--json"]) == code
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is (code == 0)
    assert [c["status"] for c in out["checks"]] == [c.status for c in checks]


def test_preflight_main_restores_bytecode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(pf, "run_checks", lambda: [])
    before = sys.dont_write_bytecode
    pf.main([])
    assert sys.dont_write_bytecode is before
