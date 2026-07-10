from types import SimpleNamespace

from memory_agent.clients.llm import OpenAIClient, TokenLedger


class FakeChatModel:
    def invoke(self, messages):
        return SimpleNamespace(content="hello", usage_metadata={"input_tokens": 7, "output_tokens": 2})


def test_openai_client_records_token_usage_by_role():
    ledger = TokenLedger()
    client = OpenAIClient("model", chat_model_factory=lambda model: FakeChatModel(), role="agent", token_ledger=ledger)

    assert client.complete("system", [{"role": "user", "content": "hi"}]) == "hello"

    assert ledger.to_dict() == {
        "agent": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9, "calls": 1}
    }


def test_token_ledger_can_predeclare_required_roles():
    ledger = TokenLedger()
    ledger.ensure_roles("updater", "compactor", "agent", "judge")
    ledger.record_text("agent", "1234", "hello")

    summary = ledger.to_dict()

    assert set(summary) == {"updater", "compactor", "agent", "judge"}
    assert summary["agent"]["calls"] == 1
    assert summary["judge"]["total_tokens"] == 0
