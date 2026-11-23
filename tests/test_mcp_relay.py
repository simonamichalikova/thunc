"""The MCP relay (thunc/mcp_relay.py), run in this process instead of as the claude CLI's subprocess.

Its socket, stdin, stdout, thread and os._exit are replaced with test doubles: a fake connection whose
files are StringIO, a stub thread that keeps its target instead of starting it, and a mock os._exit,
so a test can drive the relay line by line and read what it sent each way.
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from thunc import mcp_relay

TOOLS = [{"name": "read", "description": "Read a file.", "inputSchema": {"type": "object"}}]


class FakeConnection:
    """Stands in for the TCP connection to the agent run: replies come from `replies`, calls go to `sent`."""

    def __init__(self, replies):
        self.run_in = io.StringIO("".join(json.dumps(r) + "\n" for r in replies))
        self.sent = io.StringIO()

    def makefile(self, mode, encoding):
        return self.run_in if mode == "r" else self.sent

    def messages(self):
        return [json.loads(line) for line in self.sent.getvalue().splitlines()]


class StubThread:
    """Keeps the answers() target for the test to call, instead of running it in the background."""

    started = []

    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        StubThread.started.append(self.target)


class Exit(Exception):
    pass


@pytest.fixture
def relay(monkeypatch, tmp_path):
    """Runs the relay on the given stdin lines; returns what it wrote to stdout and the doubles it used."""

    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(TOOLS), encoding="utf-8")
    monkeypatch.setenv("THUNC_RELAY", "4321 secret")
    monkeypatch.setenv("THUNC_RELAY_TOOLS", str(tools))
    StubThread.started = []
    monkeypatch.setattr(mcp_relay.threading, "Thread", StubThread)
    exit_ = MagicMock(side_effect=Exit)  # the real os._exit would end pytest itself
    monkeypatch.setattr(mcp_relay.os, "_exit", exit_)

    def run(*stdin, replies=()):
        connection = FakeConnection(replies)
        connect = MagicMock(return_value=connection)
        monkeypatch.setattr(mcp_relay.socket, "create_connection", connect)
        stdout = io.StringIO()
        monkeypatch.setattr(mcp_relay.sys, "stdin", io.StringIO("".join(line + "\n" for line in stdin)))
        monkeypatch.setattr(mcp_relay.sys, "stdout", stdout)
        with pytest.raises(Exit):
            mcp_relay.main()
        return SimpleNamespace(stdout=stdout, connection=connection, connect=connect, exit=exit_)

    return run


def lines(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def request(ident, method, **params):
    return json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})


def test_send_writes_one_json_line_and_flushes():
    stream = MagicMock()
    mcp_relay._send(stream, {"text": "čaj ☕"})
    stream.write.assert_called_once_with('{"text": "čaj ☕"}\n')  # not escaped to \u...
    stream.flush.assert_called_once_with()


def test_connects_to_the_run_and_sends_the_token_first(relay):
    run = relay()
    run.connect.assert_called_once_with(("127.0.0.1", 4321))
    assert run.connection.messages() == [{"token": "secret"}]
    assert lines(run.stdout) == []
    run.exit.assert_called_once_with(0)  # stdin closed, so the relay stops


def test_initialize_answers_with_the_clients_protocol_version(relay):
    run = relay(request(1, "initialize", protocolVersion="2024-11-05"))
    assert lines(run.stdout) == [{"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
        "serverInfo": {"name": "thunc", "version": "1"},
    }}]  # fmt: skip


def test_initialize_without_a_version_uses_the_default(relay):
    run = relay(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
    assert lines(run.stdout)[0]["result"]["protocolVersion"] == "2025-06-18"


def test_tools_list_returns_the_tools_from_the_file(relay):
    run = relay(request(2, "tools/list"))
    assert lines(run.stdout) == [{"jsonrpc": "2.0", "id": 2, "result": {"tools": TOOLS}}]


def test_a_tool_call_is_forwarded_to_the_run_not_answered(relay):
    run = relay(request(3, "tools/call", name="read", arguments={"path": "a.py"}))
    assert lines(run.stdout) == []  # the answer comes later, from the run
    assert run.connection.messages()[1:] == [{"id": 3, "name": "read", "arguments": {"path": "a.py"}}]


def test_a_tool_call_without_params_is_forwarded_with_empty_ones(relay):
    run = relay(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call"}))
    assert run.connection.messages()[1:] == [{"id": 4, "name": "", "arguments": {}}]


def test_ping_is_answered_and_an_unknown_method_is_an_error(relay):
    run = relay(request(5, "ping"), request(6, "resources/list"))
    assert lines(run.stdout) == [
        {"jsonrpc": "2.0", "id": 5, "result": {}},
        {"jsonrpc": "2.0", "id": 6, "error": {"code": -32601, "message": "method not found: resources/list"}},
    ]


def test_notifications_and_lines_that_are_not_json_are_skipped(relay):
    notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    run = relay("not json {", notification, request(7, "ping"))
    assert lines(run.stdout) == [{"jsonrpc": "2.0", "id": 7, "result": {}}]


def test_the_runs_answers_become_mcp_results(relay):
    replies = [{"id": 3, "text": "print(1)", "error": False}, {"id": 4, "text": "no such file", "error": True}]
    run = relay(replies=replies)
    run.exit.reset_mock()
    answers = StubThread.started[0]
    with pytest.raises(Exit):
        answers()
    assert lines(run.stdout) == [
        {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "print(1)"}], "isError": False}},
        {"jsonrpc": "2.0", "id": 4, "result": {"content": [{"type": "text", "text": "no such file"}], "isError": True}},
    ]
    run.exit.assert_called_once_with(0)  # the run closed the connection
