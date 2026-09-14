import json
import subprocess

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.agents.utils.structured import invoke_structured_or_freetext
from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.subscription_client import (
    SubscriptionCLIChatModel,
    SubscriptionCLIError,
    SubscriptionUsageLimitError,
    _extract_json,
)

USAGE_LIMIT_NOTICE = "You've hit your session limit · resets 4am (Europe/Bratislava)"


class Rating(BaseModel):
    rating: str
    confidence: int


@tool
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Retrieve stock price data for a ticker."""
    return f"CSV for {symbol} {start_date}..{end_date}: close=101.5"


def _claude(tmp_path, **kwargs):
    return SubscriptionCLIChatModel(
        provider="claude-code", model="sonnet", command="claude", workdir=str(tmp_path), **kwargs
    )


def _claude_json(result, is_error=False, status=None):
    """What ``claude --print --output-format json`` writes to stdout."""
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "api_error_status": status,
        "result": result,
    })


def _fake_cli(monkeypatch, replies):
    """Patch subprocess.run to answer with ``replies`` in order; returns the recorded calls.

    A reply is the model's text, or a ready CompletedProcess for failures.
    """
    calls = []
    replies = list(replies)

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "input": kwargs["input"]})
        reply = replies.pop(0)
        if isinstance(reply, subprocess.CompletedProcess):
            return reply
        return subprocess.CompletedProcess(cmd, 0, stdout=_claude_json(reply), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _system_prompt(cmd):
    return cmd[cmd.index("--system-prompt") + 1]


def test_subscription_providers_do_not_require_api_keys():
    assert get_api_key_env("codex-cli") is None
    assert get_api_key_env("claude-code") is None


def test_subscription_providers_are_in_model_catalog():
    assert get_model_options("codex-cli", "quick")
    assert get_model_options("claude-code", "deep")


def test_factory_creates_subscription_clients(monkeypatch):
    monkeypatch.setenv("CODEX_CLI_COMMAND", "/bin/echo")
    client = create_llm_client("codex-cli", "default")
    llm = client.get_llm()

    assert isinstance(llm, SubscriptionCLIChatModel)
    assert llm.provider == "codex-cli"
    assert llm.command == "/bin/echo"


def test_codex_cli_uses_output_last_message(monkeypatch, tmp_path):
    def fake_run(cmd, input, text, capture_output, timeout, check, cwd):
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w") as handle:
            handle.write("subscription response")
        return subprocess.CompletedProcess(cmd, 0, stdout="logs", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    llm = SubscriptionCLIChatModel(
        provider="codex-cli",
        model="default",
        command="codex",
        workdir=str(tmp_path),
    )

    result = llm.invoke([HumanMessage(content="hello")])

    assert result.content == "subscription response"


def test_claude_cli_reads_the_json_result(monkeypatch, tmp_path):
    def fake_run(cmd, input, text, capture_output, timeout, check, cwd):
        assert cmd[:4] == ["claude", "--print", "--output-format", "json"]
        return subprocess.CompletedProcess(cmd, 0, stdout=_claude_json("claude response\n") + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _claude(tmp_path).invoke("hello")

    assert result.content == "claude response"


def test_claude_runs_without_its_own_tools_and_with_our_system_prompt(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, ["ok"])

    _claude(tmp_path).invoke([SystemMessage(content="You are the Market Analyst."), HumanMessage(content="AAPL")])

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert _system_prompt(cmd) == "You are the Market Analyst."
    assert "Market Analyst" not in calls[0]["input"]
    assert "[user]\nAAPL" in calls[0]["input"]


def test_claude_always_replaces_the_coding_agent_system_prompt(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, ["ok"])

    _claude(tmp_path).invoke("hello")

    assert "TradingAgents" in _system_prompt(calls[0]["cmd"])


def test_oversized_system_prompt_moves_to_stdin(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, ["ok"])
    huge = "x" * 200_000

    _claude(tmp_path).invoke([SystemMessage(content=huge), HumanMessage(content="go")])

    assert huge not in calls[0]["cmd"]
    assert calls[0]["input"].startswith(f"[system]\n{huge}")


@pytest.mark.parametrize(
    "process",
    [
        # Current CLI: an error result on stdout, non-zero exit.
        subprocess.CompletedProcess([], 1, stdout=_claude_json(USAGE_LIMIT_NOTICE, is_error=True, status=429), stderr=""),
        # Flagged as an error but exiting 0.
        subprocess.CompletedProcess([], 0, stdout=_claude_json(USAGE_LIMIT_NOTICE, is_error=True), stderr=""),
        # Plain-text notice instead of JSON (older CLIs).
        subprocess.CompletedProcess([], 1, stdout="", stderr="Claude AI usage limit reached|1757901600\n"),
        # Notice returned as if it were the answer.
        subprocess.CompletedProcess([], 0, stdout=_claude_json(USAGE_LIMIT_NOTICE), stderr=""),
    ],
)
def test_used_up_subscription_raises_usage_limit_error(monkeypatch, tmp_path, process):
    _fake_cli(monkeypatch, [process])

    with pytest.raises(SubscriptionUsageLimitError):
        _claude(tmp_path).invoke("analyze AAPL")


def test_usage_limit_error_keeps_the_reset_time(monkeypatch, tmp_path):
    _fake_cli(monkeypatch, [subprocess.CompletedProcess([], 1, stdout=_claude_json(USAGE_LIMIT_NOTICE, is_error=True), stderr="")])

    with pytest.raises(SubscriptionUsageLimitError, match="resets 4am"):
        _claude(tmp_path).invoke("analyze AAPL")


@pytest.mark.parametrize(
    "process",
    [
        subprocess.CompletedProcess([], 1, stdout=_claude_json("API Error: 500 Internal server error", is_error=True, status=500), stderr=""),
        subprocess.CompletedProcess([], 1, stdout=_claude_json("Server is temporarily limiting requests (not your usage limit)", is_error=True, status=429), stderr=""),
        subprocess.CompletedProcess([], 1, stdout="", stderr="Not logged in"),
        subprocess.CompletedProcess([], 0, stdout=_claude_json(""), stderr=""),
    ],
)
def test_other_failures_are_ordinary_cli_errors(monkeypatch, tmp_path, process):
    _fake_cli(monkeypatch, [process])

    with pytest.raises(SubscriptionCLIError) as raised:
        _claude(tmp_path).invoke("analyze AAPL")

    assert not isinstance(raised.value, SubscriptionUsageLimitError)


def test_long_answer_mentioning_a_limit_is_not_a_usage_limit(monkeypatch, tmp_path):
    report = "You've reached your target allocation in AAPL.\n" + "Details. " * 200
    _fake_cli(monkeypatch, [report])

    assert _claude(tmp_path).invoke("analyze AAPL").content == report.strip()


def test_structured_agent_does_not_retry_after_usage_limit(monkeypatch, tmp_path):
    limit = subprocess.CompletedProcess([], 1, stdout=_claude_json(USAGE_LIMIT_NOTICE, is_error=True), stderr="")
    calls = _fake_cli(monkeypatch, [limit, "free-text answer"])
    llm = _claude(tmp_path)

    with pytest.raises(SubscriptionUsageLimitError):
        invoke_structured_or_freetext(llm.with_structured_output(Rating), llm, "rate NVDA", str, "Portfolio Manager")

    assert len(calls) == 1


def test_bound_tools_are_described_and_json_reply_becomes_tool_call(monkeypatch, tmp_path):
    reply = '{"tool_calls": [{"name": "get_stock_data", "args": {"symbol": "AAPL", "start_date": "2026-09-01", "end_date": "2026-09-12"}}]}'
    calls = _fake_cli(monkeypatch, [reply])

    result = _claude(tmp_path).bind_tools([get_stock_data]).invoke("analyze AAPL")

    system = _system_prompt(calls[0]["cmd"])
    assert "get_stock_data" in system and '"tool_calls"' in system
    assert result.content == ""
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "get_stock_data"
    assert call["args"] == {"symbol": "AAPL", "start_date": "2026-09-01", "end_date": "2026-09-12"}
    assert call["id"].startswith("call_")


def test_tool_call_in_code_fence_or_after_prose_is_accepted(monkeypatch, tmp_path):
    _fake_cli(monkeypatch, [
        '```json\n{"tool_calls": [{"name": "get_stock_data", "args": {}}]}\n```',
        'Fetching prices first.\n{"tool_calls": [{"name": "get_stock_data", "args": {}}]}',
    ])
    llm = _claude(tmp_path).bind_tools([get_stock_data])

    assert llm.invoke("a").tool_calls[0]["name"] == "get_stock_data"
    assert llm.invoke("b").tool_calls[0]["name"] == "get_stock_data"


def test_plain_text_or_unknown_tool_stays_final_answer(monkeypatch, tmp_path):
    report = 'Final report. Example payload: {"tool_calls": [{"name": "rm_rf", "args": {}}]}'
    _fake_cli(monkeypatch, ["## Market report\nAAPL is range-bound.", report])
    llm = _claude(tmp_path).bind_tools([get_stock_data])

    first = llm.invoke("a")
    second = llm.invoke("b")

    assert first.tool_calls == [] and first.content.startswith("## Market report")
    assert second.tool_calls == [] and second.content == report


def test_analyst_tool_loop_feeds_real_tool_results_back(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, [
        '{"tool_calls": [{"name": "get_stock_data", "args": {"symbol": "AAPL", "start_date": "2026-09-01", "end_date": "2026-09-12"}}]}',
        "AAPL closed at 101.5.",
    ])
    llm = _claude(tmp_path).bind_tools([get_stock_data])
    messages = [SystemMessage(content="You are the Market Analyst."), HumanMessage(content="AAPL")]

    first = llm.invoke(messages)
    messages.append(first)
    messages.extend(get_stock_data.invoke(call) for call in first.tool_calls)
    final = llm.invoke(messages)

    second_input = calls[1]["input"]
    call_id = first.tool_calls[0]["id"]
    assert f'"id": "{call_id}"' in second_input
    assert f"[tool result get_stock_data for call {call_id}]" in second_input
    assert "close=101.5" in second_input
    assert final.tool_calls == [] and final.content == "AAPL closed at 101.5."


def test_subscription_structured_output_parses_pydantic(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, ['{"rating":"buy","confidence":87}'])

    result = _claude(tmp_path).with_structured_output(Rating).invoke("rate NVDA")

    assert "JSON Schema" in calls[0]["input"]
    assert result == Rating(rating="buy", confidence=87)


def test_structured_output_sends_system_messages_as_system_prompt(monkeypatch, tmp_path):
    calls = _fake_cli(monkeypatch, ['{"rating":"hold","confidence":50}'])

    _claude(tmp_path).with_structured_output(Rating).invoke(
        [SystemMessage(content="You are the Portfolio Manager."), HumanMessage(content="Decide on AAPL.")]
    )

    assert _system_prompt(calls[0]["cmd"]) == "You are the Portfolio Manager."
    assert "Decide on AAPL." in calls[0]["input"]


def test_extract_json_ignores_trailing_braces():
    text = 'Here is the result: {"rating":"buy","confidence":87} and a note {not json}'

    assert _extract_json(text) == {"rating": "buy", "confidence": 87}


def test_extract_json_handles_nested_strings():
    text = '```json\n{"rating":"buy","confidence":87,"note":"uses } inside a string"}\n```'

    assert _extract_json(text) == {
        "rating": "buy",
        "confidence": 87,
        "note": "uses } inside a string",
    }


@pytest.mark.parametrize("provider", ["claude-code", "codex-cli"])
def test_bind_tools_keeps_original_model_unbound(provider, tmp_path):
    llm = SubscriptionCLIChatModel(provider=provider, model="default", command="x", workdir=str(tmp_path))

    bound = llm.bind_tools([get_stock_data])

    assert bound.bound_tools == (get_stock_data,)
    assert llm.bound_tools == ()


def test_each_call_logs_its_graph_step_and_reply(monkeypatch, tmp_path, caplog):
    import logging
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    _fake_cli(monkeypatch, ['{"tool_calls": [{"name": "get_stock_data", "args": {}}]}', "Bayer looks cheap."])
    llm = _claude(tmp_path)

    class State(TypedDict):
        answer: str

    def analyst(state):
        llm.bind_tools([get_stock_data]).invoke("go again")
        return {"answer": llm.invoke("go").content}

    graph = StateGraph(State)
    graph.add_node("News Analyst", analyst)
    graph.add_edge(START, "News Analyst")
    graph.add_edge("News Analyst", END)

    with caplog.at_level(logging.INFO, logger="tradingagents.llm_clients.subscription_client"):
        graph.compile().invoke({"answer": ""})

    records = [r for r in caplog.records if r.name == "tradingagents.llm_clients.subscription_client"]
    assert [r.step for r in records] == ["News Analyst", "News Analyst"]
    assert records[0].reply == "requested get_stock_data"
    assert records[1].reply == "Bayer looks cheap."
