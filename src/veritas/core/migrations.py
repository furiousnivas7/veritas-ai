"""SQLite schema migrations, tracked with ``PRAGMA user_version``.

Versions
--------
* 0: unversioned. Milestone 1 tables, possibly the first Milestone 2 tables.
* 1: entities no longer UNIQUE on (entity_type, normalized_name); entity_mentions
  unique on (run_id, span_id, char_start, char_end, entity_type).
* 2: evidence.role (quote | scope | header), NOT NULL DEFAULT 'quote'. Every row
  written before v2 came from a model quote, so the default is accurate.
* 3: claims.metadata_json (currency/period MetadataDecision list), NOT NULL DEFAULT
  '[]'. Claims written before v3 had no enrichment, so "no decisions" is accurate.

Safety
------
* Before any migration changes a non-empty file database, a full copy is written
  with SQLite's online backup API to ``<db>.bak-v<from>-<timestamp>``.
* Constraint changes use SQLite's documented table rebuild ("making other kinds of
  table schema changes"): foreign keys off, create the new table, copy every
  column, drop the old one, rename, recreate indexes, ``PRAGMA foreign_key_check``.
  Row counts are compared before commit; any mismatch or FK violation rolls back.
* Existing ids (documents, spans, claims, evidence, entities) are never rewritten.
* Tables that do not exist yet are left to ``create_all``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Table
from sqlalchemy.engine import Dialect, Engine
from sqlalchemy.schema import CreateIndex, CreateTable

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3


class MigrationError(RuntimeError):
    pass


Conn = sqlite3.Connection


def _user_version(conn: Conn) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _tables(conn: Conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def unique_constraints(conn: Conn, table: str) -> list[tuple[str, ...]]:
    """Column tuples of every UNIQUE constraint/index on ``table`` (not the PK)."""
    out = []
    for row in conn.execute(f"PRAGMA index_list('{table}')").fetchall():
        name, unique, origin = row[1], row[2], row[3]
        if unique and origin in ("u", "c"):
            cols = tuple(r[2] for r in conn.execute(f"PRAGMA index_info('{name}')"))
            out.append(cols)
    return out


def rebuild_table(conn: Conn, table: Table, dialect: Dialect) -> int:
    """Recreate ``table`` with its current ORM definition, keeping every row.
    Must run with foreign keys OFF inside a transaction. Returns rows copied."""
    name = table.name
    tmp = f"_new_{name}"
    before = conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    old_cols = {r[1] for r in conn.execute(f"PRAGMA table_info('{name}')")}
    cols = [c.name for c in table.columns if c.name in old_cols]
    missing = [
        c.name
        for c in table.columns
        if c.name not in old_cols and not c.nullable and c.server_default is None
    ]
    if missing:
        raise MigrationError(f"{name}: cannot fill new NOT NULL columns {missing}")

    ddl = str(CreateTable(table).compile(dialect=dialect)).strip()
    if not ddl.startswith(f"CREATE TABLE {name} "):
        raise MigrationError(f"unexpected DDL for {name}: {ddl[:60]}")
    conn.execute(ddl.replace(f"CREATE TABLE {name} ", f'CREATE TABLE "{tmp}" ', 1))
    col_list = ", ".join(f'"{c}"' for c in cols)
    conn.execute(f'INSERT INTO "{tmp}" ({col_list}) SELECT {col_list} FROM "{name}"')
    conn.execute(f'DROP TABLE "{name}"')
    conn.execute(f'ALTER TABLE "{tmp}" RENAME TO "{name}"')
    for index in table.indexes:
        conn.execute(str(CreateIndex(index).compile(dialect=dialect)))

    after = conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    if before != after:
        raise MigrationError(f"{name}: row count changed {before} -> {after}")
    return int(after)


def _to_v1(conn: Conn, dialect: Dialect) -> list[str]:
    from veritas.core.db import EntityMentionRow, EntityRow

    done = []
    tables = _tables(conn)
    if "entities" in tables and ("entity_type", "normalized_name") in unique_constraints(
        conn, "entities"
    ):
        n = rebuild_table(conn, EntityRow.__table__, dialect)  # type: ignore[arg-type]
        done.append(f"entities: dropped UNIQUE(entity_type, normalized_name) ({n} rows kept)")
    want = ("run_id", "span_id", "char_start", "char_end", "entity_type")
    if "entity_mentions" in tables and want not in unique_constraints(conn, "entity_mentions"):
        n = rebuild_table(conn, EntityMentionRow.__table__, dialect)  # type: ignore[arg-type]
        done.append(f"entity_mentions: unique key now includes char_end ({n} rows kept)")
    return done


def _to_v2(conn: Conn, dialect: Dialect) -> list[str]:
    if "evidence" not in _tables(conn):
        return []
    cols = {r[1] for r in conn.execute("PRAGMA table_info('evidence')")}
    if "role" in cols:
        return []
    before = conn.execute("SELECT count(*) FROM evidence").fetchone()[0]
    conn.execute("ALTER TABLE evidence ADD COLUMN role VARCHAR(16) NOT NULL DEFAULT 'quote'")
    after = conn.execute("SELECT count(*) FROM evidence WHERE role = 'quote'").fetchone()[0]
    if before != after:
        raise MigrationError(f"evidence: {before} rows but {after} with role 'quote'")
    return [f"evidence: added role column ({after} existing rows = 'quote')"]


def _to_v3(conn: Conn, dialect: Dialect) -> list[str]:
    if "claims" not in _tables(conn):
        return []
    cols = {r[1] for r in conn.execute("PRAGMA table_info('claims')")}
    if "metadata_json" in cols:
        return []
    before = conn.execute("SELECT count(*) FROM claims").fetchone()[0]
    conn.execute("ALTER TABLE claims ADD COLUMN metadata_json JSON NOT NULL DEFAULT '[]'")
    after = conn.execute("SELECT count(*) FROM claims WHERE metadata_json = '[]'").fetchone()[0]
    if before != after:
        raise MigrationError(f"claims: {before} rows but {after} with metadata_json '[]'")
    return [f"claims: added metadata_json ({after} existing rows = [])"]


MIGRATIONS: dict[int, Callable[[Conn, Dialect], list[str]]] = {1: _to_v1, 2: _to_v2, 3: _to_v3}


def _backup(conn: Conn, db_path: Path, from_version: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = db_path.with_name(f"{db_path.name}.bak-v{from_version}-{stamp}")
    dst = sqlite3.connect(dest)
    try:
        conn.backup(dst)
    finally:
        dst.close()
    return dest


def migrate(engine: Engine, db_path: Path | None) -> list[str]:
    """Bring the database to SCHEMA_VERSION. Returns a log of applied changes.
    ``db_path`` is the SQLite file (None for in-memory databases: no backup)."""
    raw = engine.raw_connection()
    try:
        conn = raw.driver_connection
        if not isinstance(conn, sqlite3.Connection):
            raise MigrationError("migrations support SQLite only")
        conn.isolation_level = None  # explicit BEGIN/COMMIT below
        version = _user_version(conn)
        if version > SCHEMA_VERSION:
            raise MigrationError(
                f"database schema version {version} is newer than this code supports "
                f"({SCHEMA_VERSION}); refusing to open it. Use a newer VERITAS version."
            )
        if version == SCHEMA_VERSION:
            return []
        if not _tables(conn):  # fresh database: create_all builds the current schema
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return []

        applied: list[str] = []
        if db_path is not None:
            backup = _backup(conn, db_path, version)
            applied.append(f"backup written to {backup}")

        conn.execute("PRAGMA foreign_keys=OFF")  # not changeable inside a transaction
        try:
            for target in range(version + 1, SCHEMA_VERSION + 1):
                conn.execute("BEGIN IMMEDIATE")
                try:
                    changes = MIGRATIONS[target](conn, engine.dialect)
                    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
                    if violations:
                        raise MigrationError(f"foreign key violations: {violations[:5]}")
                    conn.execute(f"PRAGMA user_version = {target}")
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                applied.extend(f"v{target}: {c}" for c in changes or ["no table changes needed"])
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()
    for line in applied:
        log.info("schema migration", extra={"change": line})
    return applied


__all__ = ["SCHEMA_VERSION", "MigrationError", "migrate", "rebuild_table", "unique_constraints"]
