import json
import logging

import pytest

from veritas.core.logging import configure_logging


def test_json_logging_includes_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    logging.getLogger("veritas.test").info("registered", extra={"document_id": "DOC-1"})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["msg"] == "registered"
    assert rec["document_id"] == "DOC-1"
    assert rec["level"] == "INFO"
