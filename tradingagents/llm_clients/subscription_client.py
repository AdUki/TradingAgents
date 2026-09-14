"""LangChain chat wrappers for subscription-backed local AI CLIs.

These providers let users who are authenticated in Codex CLI or Claude Code
run TradingAgents without separate OpenAI/Anthropic API keys. Each LangChain
invocation spawns the local CLI, so they are slower than API providers.

The CLIs have no native LangChain tool calling, so it is emulated: bound tools
are described in the system prompt, the model replies with a JSON
``tool_calls`` object when it needs data, and that reply becomes a LangChain
tool call the graph's ToolNode executes. Without this, analysts never fetch
market data and write reports from nothing. Structured agents use
prompt-constrained JSON parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict

from .base_client import BaseLLMClient
from .validators import validate_model

# Linux rejects a single argv string over 128 KiB; larger system prompts go
# through stdin instead of --system-prompt.
_MAX_SYSTEM_PROMPT_ARG_BYTES = 100_000

# Replaces Claude Code's coding-agent system prompt when the caller sends none,
# so the model never answers as a coding assistant reviewing "a prompt template".
_DEFAULT_SYSTEM_PROMPT = "You are an assistant working inside the TradingAgents trading-research application."


class SubscriptionCLIError(RuntimeError):
    """Raised when a subscription-backed CLI cannot produce a response."""


class SubscriptionUsageLimitError(SubscriptionCLIError):
    """Raised when the subscription's usage limit is used up.

    Every call fails the same way until the limit resets, so callers should
    stop or wait instead of moving on to the next analysis.
    """


# How Claude Code words a used-up subscription, e.g. "You've hit your session
# limit · resets 4am"; older versions printed "Claude AI usage limit reached|<epoch>".
_USAGE_LIMIT_PREFIXES = (
    "you've hit your",
    "you've reached your",
    "you're out of usage credits",
    "your org is out of usage",
    "claude ai usage limit reached",
)


def _is_usage_limit(message: str) -> bool:
    lines = message.replace("’", "'").lower().splitlines()
    return any(line.strip().startswith(_USAGE_LIMIT_PREFIXES) for line in lines)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _to_messages(input_: Any) -> list[BaseMessage]:
    if isinstance(input_, str):
        return [HumanMessage(content=input_)]
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    if isinstance(input_, list):
        return convert_to_messages(input_)
    return [HumanMessage(content=str(input_))]


def _render_transcript(messages: list[BaseMessage]) -> str:
    """Flatten a conversation, including tool calls and their results, to text."""
    blocks: list[str] = []
    for message in messages:
        content = _stringify_content(message.content)
        if isinstance(message, ToolMessage):
            name = f" {message.name}" if message.name else ""
            blocks.append(f"[tool result{name} for call {message.tool_call_id}]\n{content}")
        elif isinstance(message, AIMessage):
            calls = [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in message.tool_calls]
            if calls:
                envelope = json.dumps({"tool_calls": calls}, ensure_ascii=False)
                content = f"{content}\n{envelope}" if content else envelope
            blocks.append(f"[assistant]\n{content}")
        else:
            role = {"human": "user", "system": "system"}.get(message.type, message.type)
            blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


def _tool_spec(tool: Any) -> dict[str, Any]:
    return convert_to_openai_tool(tool)["function"]


def _tool_protocol(tools: tuple[Any, ...]) -> str:
    specs = [_tool_spec(tool) for tool in tools]
    return (
        "# Tools\n"
        "You can use the tools below, but you cannot run them yourself. To call tools, reply "
        "with ONLY this JSON object and nothing else (no prose, no code fences):\n"
        '{"tool_calls": [{"name": "<tool name>", "args": {<arguments>}}]}\n'
        "You may request several tools in one reply. Their results come back to you as "
        "[tool result ...] messages. Never say a tool is unavailable: request it. Once you have "
        "the data you need, reply with your final answer as plain text, not JSON.\n"
        f"Available tools (JSON Schema):\n{json.dumps(specs, ensure_ascii=False, indent=1)}"
    )


def _schema_json(schema: Any) -> str:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return json.dumps(schema.model_json_schema(), indent=2)
    if isinstance(schema, dict):
        return json.dumps(schema, indent=2)
    if hasattr(schema, "schema"):
        return json.dumps(schema.schema(), indent=2)
    return json.dumps(schema, indent=2, default=str)


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _extract_json(text: str) -> Any:
    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for candidate in _json_candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("response did not contain JSON")


def _json_candidates(text: str) -> list[str]:
    """Return balanced JSON-looking objects/arrays from free-form text."""
    candidates: list[str] = []
    pairs = {"{": "}", "[": "]"}
    for start, char in enumerate(text):
        if char not in pairs:
            continue
        stack = [pairs[char]]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in pairs:
                stack.append(pairs[current])
            elif stack and current == stack[-1]:
                stack.pop()
                if not stack:
                    candidates.append(text[start : index + 1])
                    break
    return candidates


def _parse_tool_calls(text: str, tool_names: set[str]) -> list[dict[str, Any]] | None:
    """Tool calls requested by a reply, or None when it is a final answer.

    Only a ``tool_calls`` object whose every call names a bound tool counts, so
    a report that merely quotes some JSON stays a final answer.
    """
    cleaned = _strip_code_fence(text)
    for candidate in [cleaned, *_json_candidates(cleaned)]:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("tool_calls"), list):
            continue
        calls = [call for call in payload["tool_calls"] if isinstance(call, dict)]
        if calls and all(call.get("name") in tool_names for call in calls):
            return [
                {
                    "name": call["name"],
                    "args": call["args"] if isinstance(call.get("args"), dict) else {},
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                }
                for call in calls
            ]
    return None


# An answer this short that reads like a usage-limit notice is the notice, even
# if the CLI didn't flag it as an error; real analyst replies are far longer.
_MAX_NOTICE_CHARS = 300


def _claude_answer(result: subprocess.CompletedProcess[str]) -> str:
    """The model's answer from ``claude --print --output-format json``.

    Raises SubscriptionUsageLimitError when the subscription is used up and
    SubscriptionCLIError for any other failure.
    """
    stdout = (result.stdout or "").strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None

    if not isinstance(payload, dict):
        details = (result.stderr or "").strip() or stdout or f"exit code {result.returncode}"
        _raise_cli_error(details)
    answer = str(payload.get("result") or "").strip()
    if payload.get("is_error") or result.returncode != 0:
        _raise_cli_error(answer or str(payload.get("subtype") or f"exit code {result.returncode}"))
    if not answer:
        raise SubscriptionCLIError("Claude Code completed but produced no output")
    if len(answer) <= _MAX_NOTICE_CHARS and _is_usage_limit(answer):
        _raise_cli_error(answer)
    return answer


def _raise_cli_error(details: str) -> None:
    if _is_usage_limit(details):
        raise SubscriptionUsageLimitError(f"Claude usage limit reached: {details}")
    raise SubscriptionCLIError(f"claude-code failed: {details}")


class SubscriptionCLIChatModel(BaseChatModel):
    """Chat model that shells out to Codex CLI or Claude Code."""

    provider: str
    model: str
    command: str
    timeout: int = 600
    workdir: str | None = None
    bound_tools: tuple[Any, ...] = ()
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return f"subscription-{self.provider}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "command": self.command,
        }

    def bind_tools(self, tools: Any, **kwargs: Any) -> SubscriptionCLIChatModel:
        return self.model_copy(update={"bound_tools": tuple(tools)})

    def with_structured_output(self, schema: Any, **kwargs: Any):
        schema_text = _schema_json(schema)

        def invoke(input_: Any):
            messages = [
                *_to_messages(input_),
                HumanMessage(
                    content=(
                        "Return only valid JSON that conforms to this JSON Schema. "
                        "Do not include markdown fences or explanatory prose.\n"
                        f"JSON Schema:\n{schema_text}"
                    )
                ),
            ]
            message = self.invoke(messages)
            parsed = _extract_json(message.content)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(parsed)
            return parsed

        return RunnableLambda(invoke)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n\n".join(
            _stringify_content(m.content) for m in messages if isinstance(m, SystemMessage)
        )
        if self.bound_tools:
            system = f"{system}\n\n{_tool_protocol(self.bound_tools)}".strip()
        prompt = _render_transcript([m for m in messages if not isinstance(m, SystemMessage)])
        if stop:
            prompt += "\n\nStop sequences to respect: " + ", ".join(stop)

        started = time.monotonic()
        content = self._run_cli(system, prompt)
        elapsed = time.monotonic() - started

        tool_calls = None
        if self.bound_tools:
            tool_calls = _parse_tool_calls(content, {_tool_spec(t)["name"] for t in self.bound_tools})

        # One line per call, since a ticker's analysis runs silently for minutes.
        # LangGraph passes the running node's name in the run metadata.
        step = (getattr(run_manager, "metadata", None) or {}).get("langgraph_node") or "model call"
        if tool_calls:
            outcome = reply = "requested " + ", ".join(call["name"] for call in tool_calls)
        else:
            outcome = f"answered ({len(content):,} chars)"
            reply = " ".join(content.split())[:200]
        logging.getLogger(__name__).info(
            "%s: %s in %.0fs", step, outcome, elapsed, extra={"step": step, "reply": reply}
        )

        message = AIMessage(content="", tool_calls=tool_calls) if tool_calls else AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _run_cli(self, system: str, prompt: str) -> str:
        if self.provider == "codex-cli":
            return self._run_codex(f"[system]\n{system}\n\n{prompt}" if system else prompt)
        if self.provider == "claude-code":
            return self._run_claude(system, prompt)
        raise SubscriptionCLIError(f"Unsupported subscription CLI provider: {self.provider}")

    def _run_codex(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            pass
        try:
            cmd = [
                self.command,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
            ]
            if self.model and self.model != "default":
                cmd.extend(["-m", self.model])
            cmd.extend(["-o", temp_file.name, "-"])
            self._run_subprocess(cmd, prompt)
            with open(temp_file.name, encoding="utf-8") as output:
                text = output.read().strip()
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
        if not text:
            raise SubscriptionCLIError("Codex CLI completed but produced no final message")
        return text

    def _run_claude(self, system: str, prompt: str) -> str:
        cmd = [
            self.command,
            "--print",
            # JSON flags a failed call (is_error) so an error notice such as a
            # used-up usage limit is never mistaken for the model's answer.
            "--output-format",
            "json",
            "--no-session-persistence",
            # The model plays a TradingAgents role, not a coding agent: no Claude
            # Code tools or MCP servers (data tools are emulated above).
            "--tools",
            "",
            "--strict-mcp-config",
        ]
        if self.model and self.model != "default":
            cmd.extend(["--model", self.model])
        system = system or _DEFAULT_SYSTEM_PROMPT
        if len(system.encode()) <= _MAX_SYSTEM_PROMPT_ARG_BYTES:
            cmd.extend(["--system-prompt", system])
        else:
            cmd.extend(["--system-prompt", _DEFAULT_SYSTEM_PROMPT])
            prompt = f"[system]\n{system}\n\n{prompt}"
        return _claude_answer(self._run_subprocess(cmd, prompt, check=False))

    def _run_subprocess(
        self, cmd: list[str], prompt: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=check,
                cwd=self.workdir or os.getcwd(),
            )
        except FileNotFoundError as exc:
            raise SubscriptionCLIError(
                f"Could not find {cmd[0]!r}. Install/login to the CLI or set the matching command env var."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SubscriptionCLIError(
                f"{self.provider} timed out after {self.timeout}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or f"exit code {exc.returncode}"
            raise SubscriptionCLIError(f"{self.provider} failed: {details}") from exc


class SubscriptionCLIClient(BaseLLMClient):
    """Client for subscription-backed local CLIs such as Codex and Claude Code."""

    def __init__(self, model: str, base_url: str | None = None, provider: str = "codex-cli", **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        command_env = {
            "codex-cli": "CODEX_CLI_COMMAND",
            "claude-code": "CLAUDE_CODE_COMMAND",
        }.get(self.provider)
        default_command = {
            "codex-cli": "codex",
            "claude-code": "claude",
        }.get(self.provider)
        if not command_env or not default_command:
            raise ValueError(f"Unsupported subscription provider: {self.provider}")

        command = os.environ.get(command_env) or shutil.which(default_command) or default_command
        timeout = int(os.environ.get("TRADINGAGENTS_SUBSCRIPTION_CLI_TIMEOUT", "600"))
        return SubscriptionCLIChatModel(
            provider=self.provider,
            model=self.model,
            command=command,
            timeout=timeout,
            workdir=self.kwargs.get("workdir"),
        )

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
