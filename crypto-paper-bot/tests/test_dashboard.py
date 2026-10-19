"""Smoke test: the dashboard renders end to end on offline synthetic data."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


def test_dashboard_renders_backtest_and_paper(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERBOT_SOURCE", "synthetic")
    import paperbot.config as config
    monkeypatch.setattr(config, "DEFAULT_DB_PATH", tmp_path / "dash.sqlite")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert {"Total return", "Buy & hold", "Max drawdown", "Win rate"} <= set(labels)

    create = next(b for b in at.button if b.label == "Create account")
    create.click().run()
    assert not at.exception, at.exception
    assert any("Created" in s.value for s in at.success)

    assert (tmp_path / "dash.sqlite").exists()
    process = next(b for b in at.button if b.label == "Process new candles now")
    process.click().run()
    assert not at.exception, at.exception
    assert "Equity" in [m.label for m in at.metric]
