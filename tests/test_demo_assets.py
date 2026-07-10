import os

from demos.config import SessionDemoConfig, SummaryAgentConfig
from demos.tools import calculator, weather


def test_demo_tools_keep_their_public_behavior():
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == "14.0"
    assert "Taipei: sunny, 26 C" in weather.invoke({"city": "Taipei"})


def test_demo_only_configs_read_environment(monkeypatch):
    monkeypatch.setenv("MAIN_MODEL", "openai:test-main")
    monkeypatch.setenv("SUMMARY_MODEL", "openai:test-summary")
    monkeypatch.setenv("THREAD_ID", "test-thread")
    monkeypatch.setenv("MAX_WINDOW_TOKENS", "123")
    monkeypatch.setenv("MEMORY_MODEL", "openai:test-memory")

    summary = SummaryAgentConfig.from_env()
    session = SessionDemoConfig.from_env()

    assert summary.main_model == "openai:test-main"
    assert summary.summary_model == "openai:test-summary"
    assert summary.thread_id == "test-thread"
    assert session.main_model == "openai:test-main"
    assert session.memory_model == "openai:test-memory"
    assert session.max_window_tokens == 123

    assert os.environ["THREAD_ID"] == "test-thread"
