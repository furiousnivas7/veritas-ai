# database/

The ORM models live in the package at [`src/veritas/core/db.py`](../src/veritas/core/db.py) so they can be imported. This folder is reserved for:

* `migrations/`: Alembic migrations, introduced the first time the schema must change under existing data (see ADR-001).
* `models/`: entity-relationship diagrams and schema notes.

The runtime database file is `data/veritas.sqlite3` (git-ignored). The location is set with `VERITAS_DATA_DIR` or `VERITAS_DATABASE_URL`.
