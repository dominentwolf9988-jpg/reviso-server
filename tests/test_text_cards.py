import asyncio
import json
import os

import firebase_admin
import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
os.environ["APP_INTERNAL_AUTH_KEY"] = "test-admin-key"
os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = "{}"

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

import main


def _cards(count):
    return [
        {
            "front": f"Concept {index + 1}",
            "back": f"Explanation of concept {index + 1} in plain words.",
        }
        for index in range(count)
    ]


def _valid_output(count=5):
    return {
        "cards": _cards(count),
        "confidence": 91,
        "note": "Generated from the supplied notes.",
    }


def _completion(content):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


class _FakeTextClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def post(self, path, *, json, timeout):
        self.calls.append({"path": path, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


@pytest.fixture
def text_route(monkeypatch):
    refunds = []
    consumes = []
    extra_consumes = []

    async def get_uid(authorization):
        assert authorization == "Bearer token"
        return "firebase-user"

    async def consume_credits(uid, cost):
        assert uid == "firebase-user"
        assert cost == main.ENDPOINT_COSTS["text"]
        consumes.append(cost)
        return {"remaining": 40, "reset_at": 123456}

    async def try_consume_credits(uid, cost):
        assert uid == "firebase-user"
        extra_consumes.append(cost)
        return {"remaining": 40 - cost, "reset_at": 123456, "ok": True}

    async def refund(uid, amount, reason):
        refunds.append((uid, amount, reason))

    monkeypatch.setattr(main, "get_uid", get_uid)
    monkeypatch.setattr(main, "consume_credits", consume_credits)
    monkeypatch.setattr(main, "try_consume_credits", try_consume_credits)
    monkeypatch.setattr(main, "_refund", refund)
    return {
        "refunds": refunds,
        "consumes": consumes,
        "extra_consumes": extra_consumes,
    }


def _request(text=None):
    body = text if text is not None else ("Study notes. " * 40).strip()
    return main.TextCardsRequest(text=body)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 5),
        (1, 5),
        (19, 5),
        (20, 5),
        (21, 6),
        (25, 6),
        (26, 7),
        (30, 7),
        (50, 11),
        (100, 21),
        (300, 61),
    ],
)
def test_text_cards_cost_table(count, expected):
    assert main.text_cards_cost(count) == expected


def test_valid_output_within_base_tier_charges_no_extra(
    monkeypatch, text_route
):
    expected = _valid_output(count=20)
    client = _FakeTextClient([_completion(json.dumps(expected, ensure_ascii=False))])
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.generate_cards_from_text(_request(), authorization="Bearer token")
    )

    content = response["choices"][0]["message"]["content"]
    assert content == main.TextCardsOutput.model_validate(expected).model_dump_json()
    assert response["credits"]["remaining"] == 40
    assert text_route["consumes"] == [main.ENDPOINT_COSTS["text"]]
    assert text_route["extra_consumes"] == []
    assert text_route["refunds"] == []
    assert len(client.calls) == 1

    payload = client.calls[0]["json"]
    assert payload["model"] == main.TEXT_MODEL
    assert payload["reasoning"] == {"enabled": False}
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == main.TEXT_CARDS_MAX_TOKENS
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "text_flashcards"
    assert response_format["json_schema"]["strict"] is True
    cards_schema = response_format["json_schema"]["schema"]["properties"]["cards"]
    assert {branch.get("maxItems") for branch in cards_schema["anyOf"]} == {
        0,
        main.TEXT_CARDS_MAX_COUNT,
    }


def test_extra_tier_charges_one_credit_per_five_cards(monkeypatch, text_route):
    expected = _valid_output(count=26)
    client = _FakeTextClient([_completion(json.dumps(expected, ensure_ascii=False))])
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.generate_cards_from_text(_request(), authorization="Bearer token")
    )

    assert json.loads(response["choices"][0]["message"]["content"]) == expected
    # 26 cards -> total 7, base 5 already consumed, extra 2 charged after.
    assert text_route["extra_consumes"] == [2]
    assert response["credits"]["remaining"] == 40 - 2
    assert text_route["refunds"] == []


def test_empty_cards_allowed_without_extra_charge(monkeypatch, text_route):
    expected = {"cards": [], "confidence": 0, "note": "No usable content"}
    client = _FakeTextClient([_completion(json.dumps(expected))])
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.generate_cards_from_text(_request(), authorization="Bearer token")
    )

    assert json.loads(response["choices"][0]["message"]["content"]) == expected
    assert text_route["extra_consumes"] == []
    assert text_route["refunds"] == []


def test_over_cap_output_returns_502_and_refunds_base(monkeypatch, text_route):
    oversized = {
        "cards": _cards(main.TEXT_CARDS_MAX_COUNT + 1),
        "confidence": 90,
        "note": "Too many cards.",
    }
    client = _FakeTextClient(
        [_completion(json.dumps(oversized)), _completion(json.dumps(oversized))]
    )
    monkeypatch.setattr(main, "_http_client", client)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.generate_cards_from_text(_request(), authorization="Bearer token")
        )

    assert error.value.status_code == 502
    assert len(client.calls) == 2
    assert text_route["extra_consumes"] == []
    assert text_route["refunds"] == [
        ("firebase-user", main.ENDPOINT_COSTS["text"], "text_cards_502")
    ]


def test_extra_charge_shortfall_still_returns_cards(monkeypatch, text_route, caplog):
    expected = _valid_output(count=25)

    async def shortfall(uid, cost):
        assert cost == 1
        return {"remaining": 0, "reset_at": 123456, "ok": False}

    monkeypatch.setattr(main, "try_consume_credits", shortfall)
    client = _FakeTextClient([_completion(json.dumps(expected, ensure_ascii=False))])
    monkeypatch.setattr(main, "_http_client", client)

    with caplog.at_level("WARNING", logger="proxy"):
        response = asyncio.run(
            main.generate_cards_from_text(_request(), authorization="Bearer token")
        )

    assert json.loads(response["choices"][0]["message"]["content"]) == expected
    assert "text_cards_undercharge" in caplog.text
    assert text_route["refunds"] == []


def test_request_rejects_short_and_oversized_text():
    with pytest.raises(ValidationError):
        main.TextCardsRequest(text="Too short. " * 5)
    with pytest.raises(ValidationError):
        main.TextCardsRequest(text="x" * 20_001)
    request = main.TextCardsRequest(text=("Study notes. " * 40).strip())
    assert len(request.text) >= 200
