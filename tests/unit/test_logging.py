"""Tests for structured logging configuration."""

from structlog.testing import capture_logs

from rag_loader.observability.logging import configure_logging, get_logger


def test_logging_emits_structured_event():
    configure_logging(level="INFO", json_logs=True)
    with capture_logs() as logs:
        get_logger("test").info("indexing_started", documents=3)
    assert any(
        entry["event"] == "indexing_started" and entry.get("documents") == 3 for entry in logs
    )
