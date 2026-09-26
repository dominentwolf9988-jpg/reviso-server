import asyncio
import json
import os

import firebase_admin
import httpx
import pytest
from fastapi import HTTPException

os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
os.environ["APP_INTERNAL_AUTH_KEY"] = "test-admin-key"
os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = "{}"

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

import main

DIM_NAMES = [name for name, _ in main.JEV_SCORE_DIMS]


def _decisions(levels, confidences=None, genuine=0.99,
               model="typesafe/jev-1.13-20260917"):
    answers = {}
    for name in DIM_NAMES:
        level = levels.get(name, 3)
        conf = (confidences or {}).get(name, 0.9)
        answers[name] = {
            "type": "score",
            "score": level,
            "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1},
            "confidence": conf,
        }
    answers["genuine"] = {"type": "noul", "noul": genuine}
    return {"model": model, "answers": answers,
            "usage": {"input_tokens": 400, "output_tokens": 35}}


def _completion(content):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def _decisions_response(levels, confidences=None, genuine=0.99,
                        model="typesafe/jev-1.13-20260917"):
    return httpx.Response(
        200, json=_decisions(levels, confidences, genuine, model)
    )


class _FakeHttpClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def post(self, path, *, json, timeout=None):
        self.calls.append({"path": path, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


@pytest.fixture
def score_route(monkeypatch):
    refunds = []
    consumes = []

    async def get_uid(authorization):
        assert authorization == "Bearer token"
        return "firebase-user"

    async def consume_credits(uid, cost):
        assert uid == "firebase-user"
        consumes.append(cost)
        return {"remaining": 40, "reset_at": 123456}

    async def refund(uid, amount, reason):
        refunds.append((uid, amount, reason))

    monkeypatch.setattr(main, "get_uid", get_uid)
    monkeypatch.setattr(main, "consume_credits", consume_credits)
    monkeypatch.setattr(main, "_refund", refund)
    return {"refunds": refunds, "consumes": consumes}


def _request():
    return main.ScoreRequest(
        question="What is photosynthesis?",
        actual_answer="Plants convert sunlight to glucose.",
        user_answer="Plants make food from sunlight.",
    )


def test_weighted_math_and_accuracy_gate():
    assert main._jev_weighted_score({n: 3 for n, _ in main.JEV_SCORE_DIMS}) == 100
    assert main._jev_weighted_score({n: 0 for n, _ in main.JEV_SCORE_DIMS}) == 0
    # Fluent-but-wrong: high style dims, zero accuracy -> capped at 25.
    levels = {n: 3 for n, _ in main.JEV_SCORE_DIMS}
    levels["accuracy"] = 0
    assert main._jev_weighted_score(levels) <= 25


def test_jev_served_shape_has_no_note(monkeypatch, score_route):
    levels = {n: 3 for n in DIM_NAMES}
    client = _FakeHttpClient([_decisions_response(levels)])
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.score_answer(_request(), authorization="Bearer token")
    )

    content = json.loads(response["choices"][0]["message"]["content"])
    assert set(content) == {*DIM_NAMES, "score"}
    assert all(isinstance(content[name], int) for name in DIM_NAMES)
    assert content["score"] == 100
    assert "note" not in content
    assert [call["path"] for call in client.calls] == [
        "https://openrouter.ai/api/alpha/decisions"
    ]
    assert score_route["consumes"] == [main.ENDPOINT_COSTS["score"]]
    assert score_route["refunds"] == []


def test_low_confidence_falls_back_to_llm(monkeypatch, score_route):
    levels = {n: 2 for n in DIM_NAMES}
    low = {n: 0.1 for n in DIM_NAMES}
    llm_content = json.dumps({"score": 70, "note": "y"})
    client = _FakeHttpClient(
        [_decisions_response(levels, low), _completion(llm_content)]
    )
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.score_answer(_request(), authorization="Bearer token")
    )

    assert json.loads(response["choices"][0]["message"]["content"])["score"] == 70
    assert [call["path"] for call in client.calls] == [
        "https://openrouter.ai/api/alpha/decisions",
        "/chat/completions",
    ]
    # Single base charge; fallback never double-charges.
    assert score_route["consumes"] == [main.ENDPOINT_COSTS["score"]]
    assert score_route["refunds"] == []


def test_jev_transport_error_falls_back(monkeypatch, score_route):
    llm_content = json.dumps({"score": 60, "note": "z"})
    client = _FakeHttpClient(
        [httpx.ConnectError("down"), _completion(llm_content)]
    )
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.score_answer(_request(), authorization="Bearer token")
    )

    assert json.loads(response["choices"][0]["message"]["content"])["score"] == 60
    assert score_route["refunds"] == []


def test_llm_failure_still_refunds_base(monkeypatch, score_route):
    async def low_conf(question, reference_answer, student_answer):
        return None

    async def boom(payload, primary_model, fallback_model):
        raise RuntimeError("down")

    monkeypatch.setattr(main, "score_with_jev", low_conf)
    monkeypatch.setattr(main, "call_with_fallback", boom)

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.score_answer(_request(), authorization="Bearer token"))

    assert error.value.status_code == 500
    assert score_route["refunds"] == [
        ("firebase-user", main.ENDPOINT_COSTS["score"], "server_error")
    ]
