"""The agent's perceive-decide-act loop, with a scripted stand-in for the model.

No model service is called: a fake client returns pre-written turns, so these tests check the
loop itself: tool results flow back to the model, reprocessing waits for a person, declines and
errors are handled, and every step lands in the trace."""
import json
import threading
import time
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from worldline import agent


@pytest.fixture
def clip_dir(tmp_path, monkeypatch):
    """A tiny processed clip: 10 frames, one tube moving left to right."""
    d = tmp_path / "data" / "demo"
    d.mkdir(parents=True)
    w, h, T = 64, 36, 10
    atlas = np.zeros((h, w * T, 3), np.uint8)
    for k in range(T):
        atlas[10:26, k * w + 5 + 4 * k:k * w + 15 + 4 * k] = 200
    cv2.imwrite(str(d / "atlas.jpg"), atlas)
    n = 8
    meta = {"duration_s": 5.0, "frames": T, "times": [i * 0.5 for i in range(T)], "frame_w": w, "frame_h": h,
            "atlas": {"file": "atlas.jpg", "cols": T, "rows": 1}, "cuts": [],
            "tracks": [{"id": 1, "start": 1, "end": 8, "enter_s": 0.5, "exit_s": 4.0, "dwell_s": 3.5,
                        "path_len": 0.4, "found_by": "detector", "motion": 0.3, "src": "d" * n,
                        "cx": list(np.linspace(0.2, 0.7, n)), "cy": [0.5] * n, "r": [[0.1] * 8] * n}]}
    (d / "worldline.json").write_text(json.dumps(meta))
    monkeypatch.setattr(agent, "DATA", tmp_path / "data")
    return d


def text(t):
    return NS(type="text", text=t)


def call(tid, name, **inp):
    return NS(type="tool_use", id=tid, name=name, input=inp)


def turn(*blocks, stop="tool_use"):
    usage = NS(input_tokens=100, output_tokens=20, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    return NS(content=list(blocks), stop_reason=stop, usage=usage)


class FakeClient:
    """Returns scripted turns and records every request it was sent."""
    def __init__(self, turns):
        self.turns, self.requests = list(turns), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw["messages"], default=lambda o: o.__dict__)))
        if isinstance(self.turns[0], Exception):
            raise self.turns.pop(0)
        return self.turns.pop(0)


def run_with(monkeypatch, turns, decision=None):
    fake = FakeClient(turns)
    monkeypatch.setattr(agent, "client", lambda: fake)
    run = agent.Run("When did the animal appear?")
    th = threading.Thread(target=agent.agent_loop, args=(run,))
    th.start()
    if decision is not None:
        for _ in range(200):
            if run.status == "awaiting_approval":
                break
            time.sleep(0.01)
        assert run.status == "awaiting_approval"
        run.decide(*decision)
    th.join(5)
    return run, fake


def test_tools_measure_and_show_the_tube(clip_dir):
    assert agent.list_clips()[0]["clip"] == "demo"
    blocks = agent.inspect_tube("demo", 1)
    facts = json.loads(blocks[0]["text"])
    assert facts["enter_s"] == 0.5 and facts["dwell_s"] == 3.5 and facts["evidence"]["detector_frames"] == 8
    assert sum(b["type"] == "image" for b in blocks) == 3
    frame = agent.view_frame("demo", 2.0)
    assert json.loads(frame[0]["text"])["tubes_covering_this_frame"] == [1]
    with pytest.raises(agent.ToolError):
        agent.inspect_tube("demo", 7)
    with pytest.raises(agent.ToolError):
        agent.load_meta("../etc")


def test_observations_reach_the_model_and_shape_the_answer(clip_dir, monkeypatch):
    run, fake = run_with(monkeypatch, [
        turn(text("Let me look."), call("t1", "inspect_tube", clip="demo", tube=1)),
        turn(text("demo#1 arrives at 0.5 s and stays 3.5 s."), stop="end_turn"),
    ])
    assert run.status == "done" and "demo#1" in run.answer
    second = fake.requests[1]
    result = second[-1]["content"][0]
    assert result["type"] == "tool_result" and not result["is_error"]
    assert '"enter_s": 0.5' in result["content"][0]["text"]       # the measurement fed back to the model
    kinds = [e["kind"] for e in run.events]
    assert kinds == ["model", "tool_result", "model", "final"]


def test_reprocessing_waits_for_approval(clip_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "reprocess_clip", lambda *a: calls.append(a) or {"new_clip": "_agent/x/demo__1", "tubes": []})
    monkeypatch.setattr(agent, "CLIPS", clip_dir.parent)   # a "source video" must exist for the check
    (clip_dir.parent / "demo.webm").write_bytes(b"x")
    run, fake = run_with(monkeypatch, [
        turn(call("t1", "reprocess_clip", clip="demo", reason="the animal may be cut short", sensitivity="high",
                  start_s=None, end_s=None)),
        turn(text("Done."), stop="end_turn"),
    ], decision=(True, ""))
    assert calls and run.status == "done"
    kinds = [e["kind"] for e in run.events]
    assert "approval_request" in kinds and {"kind": "approval"} .items() <= next(e for e in run.events if e["kind"] == "approval").items()
    assert next(e for e in run.events if e["kind"] == "approval")["approved"] is True


def test_a_declined_request_is_reported_back(clip_dir, monkeypatch):
    monkeypatch.setattr(agent, "reprocess_clip", lambda *a: pytest.fail("must not run when declined"))
    run, fake = run_with(monkeypatch, [
        turn(call("t1", "reprocess_clip", clip="demo", reason="r", sensitivity="high", start_s=1.0, end_s=3.0)),
        turn(text("Without reprocessing, demo#1 is my best answer."), stop="end_turn"),
    ], decision=(False, "not now"))
    result = fake.requests[1][-1]["content"][0]
    assert result["is_error"] and "declined" in result["content"]
    assert run.status == "done"


def test_tool_errors_go_back_to_the_model(clip_dir, monkeypatch):
    run, fake = run_with(monkeypatch, [
        turn(call("t1", "inspect_tube", clip="nope", tube=1)),
        turn(text("That clip doesn't exist."), stop="end_turn"),
    ])
    result = fake.requests[1][-1]["content"][0]
    assert result["is_error"] and "no clip called" in result["content"]


def test_service_errors_and_step_limit_end_the_run(clip_dir, monkeypatch):
    import anthropic
    import httpx
    err = anthropic.APIStatusError("denied", response=httpx.Response(403, request=httpx.Request("POST", "https://x")),
                                   body={"error": {"message": "not available for this account"}})
    run, _ = run_with(monkeypatch, [err])
    assert run.status == "error" and "not available" in run.answer
    monkeypatch.setattr(agent, "MAX_TURNS", 2)
    run, _ = run_with(monkeypatch, [turn(call("a", "list_clips")), turn(call("b", "list_clips"))])
    assert run.status == "stopped"


def test_one_question_at_a_time_and_daily_cap(monkeypatch):
    monkeypatch.setattr(agent, "agent_loop", lambda run: None)
    agents = agent.Agents()
    first = agents.start("q1")
    first.status = "running"
    with pytest.raises(RuntimeError):
        agents.start("q2")
    first.status = "done"
    monkeypatch.setattr(agent, "MAX_RUNS_PER_DAY", 1)
    with pytest.raises(RuntimeError):
        agents.start("q3")
