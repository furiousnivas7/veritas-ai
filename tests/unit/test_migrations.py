"""Schema migration v0 -> v1 on databases created by the original Milestone 2 code."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

import veritas.core.migrations as mig
from veritas.core.db import make_engine

FIXTURE = Path(__file__).parents[1] / "fixtures" / "legacy_v0_m2.sqlite3"
TABLES = [
    "documents",
    "document_spans",
    "extraction_runs",
    "claims",
    "evidence",
    "entities",
    "entity_aliases",
    "entity_mentions",
    "entity_match_candidates",
    "rejected_extractions",
]

LEGACY_ENTITIES = """CREATE TABLE entities (
    entity_id VARCHAR(32) NOT NULL, canonical_name VARCHAR(512) NOT NULL,
    normalized_name VARCHAR(512) NOT NULL, entity_type VARCHAR(32) NOT NULL,
    PRIMARY KEY (entity_id), UNIQUE (entity_type, normalized_name))"""
LEGACY_MENTIONS = """CREATE TABLE entity_mentions (
    mention_id VARCHAR(32) NOT NULL, run_id VARCHAR(32) NOT NULL, entity_id VARCHAR(32),
    document_id VARCHAR(16) NOT NULL, span_id VARCHAR(128) NOT NULL,
    surface VARCHAR(512) NOT NULL, normalized VARCHAR(512) NOT NULL,
    entity_type VARCHAR(32) NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
    location_json JSON NOT NULL, PRIMARY KEY (mention_id),
    UNIQUE (run_id, span_id, char_start, entity_type),
    FOREIGN KEY(run_id) REFERENCES extraction_runs (run_id) ON DELETE CASCADE,
    FOREIGN KEY(entity_id) REFERENCES entities (entity_id) ON DELETE SET NULL,
    FOREIGN KEY(document_id) REFERENCES documents (document_id),
    FOREIGN KEY(span_id) REFERENCES document_spans (span_id))"""


def snapshot(path: Path) -> dict[str, list[tuple[object, ...]] | None]:
    con = sqlite3.connect(path)
    try:
        out: dict[str, list[tuple[object, ...]] | None] = {}
        for t in TABLES:
            try:
                out[t] = sorted(con.execute(f"SELECT * FROM {t}").fetchall(), key=repr)
            except sqlite3.OperationalError:
                out[t] = None
        return out
    finally:
        con.close()


def pragma(path: Path, sql: str) -> list[tuple[object, ...]]:
    con = sqlite3.connect(path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def unique_keys(path: Path, table: str) -> list[tuple[str, ...]]:
    con = sqlite3.connect(path)
    try:
        return mig.unique_constraints(con, table)
    finally:
        con.close()


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    p = tmp_path / "veritas.sqlite3"
    shutil.copy(FIXTURE, p)
    return p


def test_fixture_really_has_the_old_constraints(legacy_db: Path) -> None:
    assert pragma(legacy_db, "PRAGMA user_version") == [(0,)]
    assert ("entity_type", "normalized_name") in unique_keys(legacy_db, "entities")
    assert ("run_id", "span_id", "char_start", "entity_type") in unique_keys(
        legacy_db, "entity_mentions"
    )


def test_migration_preserves_every_row(legacy_db: Path) -> None:
    before = snapshot(legacy_db)
    assert before["claims"] and before["evidence"] and before["entity_mentions"]
    make_engine(f"sqlite:///{legacy_db}")
    after = snapshot(legacy_db)
    # v2 appends evidence.role; every pre-existing row is a model quote.
    assert {row[-1] for row in after["evidence"] or []} == {"quote"}
    after["evidence"] = sorted((row[:-1] for row in after["evidence"] or []), key=repr)
    # v3 appends claims.metadata_json; pre-existing claims had no enrichment.
    assert {row[-1] for row in after["claims"] or []} == {"[]"}
    after["claims"] = sorted((row[:-1] for row in after["claims"] or []), key=repr)
    assert after == before
    assert pragma(legacy_db, "PRAGMA user_version") == [(mig.SCHEMA_VERSION,)]
    assert pragma(legacy_db, "PRAGMA integrity_check") == [("ok",)]
    assert pragma(legacy_db, "PRAGMA foreign_key_check") == []


def test_migration_changes_the_constraints(legacy_db: Path) -> None:
    make_engine(f"sqlite:///{legacy_db}")
    assert ("entity_type", "normalized_name") not in unique_keys(legacy_db, "entities")
    assert ("run_id", "span_id", "char_start", "char_end", "entity_type") in unique_keys(
        legacy_db, "entity_mentions"
    )
    con = sqlite3.connect(legacy_db)
    try:  # both are now legal
        e = con.execute("SELECT entity_type, normalized_name FROM entities LIMIT 1").fetchone()
        con.execute("INSERT INTO entities VALUES ('ENT-dup', 'dup', ?, ?)", (e[1], e[0]))
        m = con.execute("SELECT * FROM entity_mentions LIMIT 1").fetchone()
        con.execute(
            "INSERT INTO entity_mentions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("MEN-dup", *m[1:9], m[9] + 1, m[10]),
        )
        con.commit()
    finally:
        con.close()


def test_migration_writes_a_backup_of_the_old_database(legacy_db: Path) -> None:
    before = snapshot(legacy_db)
    make_engine(f"sqlite:///{legacy_db}")
    (backup,) = legacy_db.parent.glob("veritas.sqlite3.bak-v0-*")
    assert snapshot(backup) == before
    assert pragma(backup, "PRAGMA user_version") == [(0,)]
    assert ("entity_type", "normalized_name") in unique_keys(backup, "entities")


def test_migration_is_idempotent(legacy_db: Path) -> None:
    make_engine(f"sqlite:///{legacy_db}")
    after_first = snapshot(legacy_db)
    make_engine(f"sqlite:///{legacy_db}")
    assert snapshot(legacy_db) == after_first
    assert len(list(legacy_db.parent.glob("*.bak-*"))) == 1


def test_failed_migration_rolls_back(legacy_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = snapshot(legacy_db)

    def broken(conn: sqlite3.Connection, dialect: object) -> list[str]:
        conn.execute("DELETE FROM claims")
        raise mig.MigrationError("simulated failure")

    monkeypatch.setitem(mig.MIGRATIONS, 1, broken)
    with pytest.raises(mig.MigrationError):
        make_engine(f"sqlite:///{legacy_db}")
    assert snapshot(legacy_db) == before
    assert pragma(legacy_db, "PRAGMA user_version") == [(0,)]


def test_milestone1_only_database_gains_tables_and_keeps_data(tmp_path: Path) -> None:
    """Like the user's data/veritas.sqlite3: documents + spans, no Milestone 2 tables."""
    p = tmp_path / "m1.sqlite3"
    shutil.copy(FIXTURE, p)
    con = sqlite3.connect(p)
    for t in reversed(TABLES[2:]):
        con.execute(f"DROP TABLE {t}")
    con.commit()
    con.close()
    before = snapshot(p)
    make_engine(f"sqlite:///{p}")
    after = snapshot(p)
    assert after["documents"] == before["documents"] and after["document_spans"]
    assert after["document_spans"] == before["document_spans"]
    assert all(after[t] == [] for t in TABLES[2:])


def test_synthetic_legacy_schema(tmp_path: Path) -> None:
    """Independent of the fixture: old DDL written by hand, two same-name rows blocked
    before migration and allowed after."""
    p = tmp_path / "synthetic.sqlite3"
    make_engine(f"sqlite:///{p}")  # current schema, v1
    con = sqlite3.connect(p)
    con.executescript(f"""
        PRAGMA foreign_keys=OFF;
        DROP TABLE entity_match_candidates; DROP TABLE entity_mentions; DROP TABLE entities;
        {LEGACY_ENTITIES}; {LEGACY_MENTIONS};
        PRAGMA user_version=0;
    """)
    con.execute("INSERT INTO entities VALUES ('ENT-1', 'Alpha', 'alpha', 'project')")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO entities VALUES ('ENT-2', 'Alpha', 'alpha', 'project')")
    con.commit()
    con.close()

    make_engine(f"sqlite:///{p}")
    con = sqlite3.connect(p)
    con.execute("INSERT INTO entities VALUES ('ENT-2', 'Alpha', 'alpha', 'project')")
    assert con.execute("SELECT count(*) FROM entities").fetchone() == (2,)
    assert con.execute("SELECT count(*) FROM entity_match_candidates").fetchone() == (0,)
    con.close()


def test_fresh_database_starts_at_current_version(tmp_path: Path) -> None:
    p = tmp_path / "fresh.sqlite3"
    make_engine(f"sqlite:///{p}")
    assert pragma(p, "PRAGMA user_version") == [(mig.SCHEMA_VERSION,)]
    assert not list(tmp_path.glob("*.bak-*"))


@pytest.mark.parametrize("future", [mig.SCHEMA_VERSION + 1, 99])
def test_future_schema_version_is_refused(legacy_db: Path, future: int) -> None:
    con = sqlite3.connect(legacy_db)
    con.execute(f"PRAGMA user_version = {future}")
    con.close()
    before = snapshot(legacy_db)
    with pytest.raises(mig.MigrationError, match="newer than this code supports"):
        make_engine(f"sqlite:///{legacy_db}")
    assert snapshot(legacy_db) == before
    assert pragma(legacy_db, "PRAGMA user_version") == [(future,)]
    assert not list(legacy_db.parent.glob("*.bak-*"))


def test_current_version_is_a_no_op(tmp_path: Path) -> None:
    p = tmp_path / "v.sqlite3"
    make_engine(f"sqlite:///{p}")
    assert pragma(p, "PRAGMA user_version") == [(mig.SCHEMA_VERSION,)]
    make_engine(f"sqlite:///{p}")
    assert not list(tmp_path.glob("*.bak-*"))


def test_v2_database_gains_claim_metadata_column(tmp_path: Path) -> None:
    p = tmp_path / "v2.sqlite3"
    make_engine(f"sqlite:///{p}")
    con = sqlite3.connect(p)
    con.execute("ALTER TABLE claims DROP COLUMN metadata_json")
    con.execute("PRAGMA user_version = 2")
    con.commit()
    con.close()
    make_engine(f"sqlite:///{p}")
    cols = {r[1]: r for r in pragma(p, "PRAGMA table_info('claims')")}
    assert "metadata_json" in cols and cols["metadata_json"][4] == "'[]'"  # default
    assert pragma(p, "PRAGMA user_version") == [(3,)]
    assert len(list(tmp_path.glob("v2.sqlite3.bak-v2-*"))) == 1


def test_fixture_migrates_all_the_way_to_v3(legacy_db: Path) -> None:
    make_engine(f"sqlite:///{legacy_db}")
    assert pragma(legacy_db, "PRAGMA user_version") == [(3,)]
    assert pragma(legacy_db, "SELECT count(*) FROM claims WHERE metadata_json = '[]'") == pragma(
        legacy_db, "SELECT count(*) FROM claims"
    )
