# database/

The ORM models live in the package at [`src/veritas/core/db.py`](../src/veritas/core/db.py) so they can be imported. This folder is reserved for:

* `migrations/`: reserved. Schema migrations currently live in [`src/veritas/core/migrations.py`](../src/veritas/core/migrations.py) (versioned with `PRAGMA user_version`, backup before changes; see docs/data_model.md).
* `models/`: entity-relationship diagrams and schema notes.

The runtime database file is `data/veritas.sqlite3` (git-ignored). The location is set with `VERITAS_DATA_DIR` or `VERITAS_DATABASE_URL`.
