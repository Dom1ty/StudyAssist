from __future__ import annotations

from pathlib import Path

from study_assistant.config import load_env_file


def test_load_env_file_reads_missing_variables(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "DASHSCOPE_API_KEY=test-key",
                "QUOTED_VALUE=\"hello world\"",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)

    load_env_file(env_path)

    assert Path(env_path).exists()
    assert __import__("os").environ["DASHSCOPE_API_KEY"] == "test-key"
    assert __import__("os").environ["QUOTED_VALUE"] == "hello world"
