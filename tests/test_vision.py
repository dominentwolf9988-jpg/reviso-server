import asyncio
import base64
import json
import logging
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


def _cards(count=7):
    return [
        {
            "front": f"Formula {index + 1}",
            "back": f"F = m × a; x² + y² = r² ({index + 1})",
        }
        for index in range(count)
    ]


def _valid_output():
    return {
        "cards": _cards(),
        "confidence": 94,
        "note": "Formulas use plain Unicode symbols.",
    }


def _completion(content):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


class _FakeVisionClient:
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
def vision_route(monkeypatch):
    refunds = []

    async def get_uid(authorization):
        assert authorization == "Bearer token"
        return "firebase-user"

    async def consume_credits(uid, cost):
        assert uid == "firebase-user"
        assert cost == main.ENDPOINT_COSTS["vision"]
        return {"remaining": 40, "reset_at": 123456}

    async def refund(uid, amount, reason):
        refunds.append((uid, amount, reason))

    monkeypatch.setattr(main, "get_uid", get_uid)
    monkeypatch.setattr(main, "consume_credits", consume_credits)
    monkeypatch.setattr(main, "_refund", refund)
    return refunds


def _request():
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode("ascii")
    return main.VisionRequest(image_data=encoded, mime_type="image/png")


def test_valid_output_is_canonical_and_requests_strict_schema(
    monkeypatch, vision_route
):
    expected = _valid_output()
    client = _FakeVisionClient(
        [_completion(json.dumps(expected, ensure_ascii=False, indent=2))]
    )
    monkeypatch.setattr(main, "_http_client", client)

    response = asyncio.run(
        main.generate_cards_from_image(_request(), authorization="Bearer token")
    )

    content = response["choices"][0]["message"]["content"]
    assert content == main.VisionOutput.model_validate(expected).model_dump_json()
    assert "×" in content
    assert "²" in content
    assert vision_route == []
    assert len(client.calls) == 1

    payload = client.calls[0]["json"]
    assert payload["model"] == main.VISION_MODEL
    assert payload["provider"] == {"require_parameters": True}
    assert payload["reasoning"] == {"enabled": False}
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == main.VISION_MAX_TOKENS
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    cards_schema = response_format["json_schema"]["schema"]["properties"]["cards"]
    assert {branch.get("maxItems") for branch in cards_schema["anyOf"]} == {0, 15}


def test_malformed_primary_uses_one_valid_fallback(monkeypatch, vision_route, caplog):
    fallback = "```json\n" + json.dumps(_valid_output(), ensure_ascii=False) + "\n```"
    client = _FakeVisionClient([_completion("not JSON"), _completion(fallback)])
    monkeypatch.setattr(main, "_http_client", client)
    caplog.set_level(logging.WARNING, logger="proxy")

    response = asyncio.run(
        main.generate_cards_from_image(_request(), authorization="Bearer token")
    )

    assert json.loads(response["choices"][0]["message"]["content"]) == _valid_output()
    assert [call["json"]["model"] for call in client.calls] == [
        main.VISION_MODEL,
        main.VISION_FALLBACK_MODEL,
    ]
    assert vision_route == []
    assert (
        f"Vision model {main.VISION_MODEL} output rejected: "
        "invalid JSON at line 1, column 1"
    ) in caplog.text
    assert "not JSON" not in caplog.text


def test_schema_failure_logs_field_without_card_content(
    monkeypatch, vision_route, caplog
):
    invalid = _valid_output()
    invalid["cards"][0]["back"] = r"PRIVATE-CARD-TEXT\omega"
    client = _FakeVisionClient(
        [
            _completion(json.dumps(invalid)),
            _completion(json.dumps(_valid_output(), ensure_ascii=False)),
        ]
    )
    monkeypatch.setattr(main, "_http_client", client)
    caplog.set_level(logging.WARNING, logger="proxy")

    response = asyncio.run(
        main.generate_cards_from_image(_request(), authorization="Bearer token")
    )

    assert response["choices"][0]["message"]["content"]
    assert "Card text must not contain LaTeX or Markdown" in caplog.text
    assert "PRIVATE-CARD-TEXT" not in caplog.text


def test_both_invalid_returns_502_and_refunds_once(monkeypatch, vision_route):
    invalid_count = {**_valid_output(), "cards": _cards(6)}
    client = _FakeVisionClient(
        [_completion("{"), _completion(json.dumps(invalid_count))]
    )
    monkeypatch.setattr(main, "_http_client", client)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.generate_cards_from_image(_request(), authorization="Bearer token")
        )

    assert error.value.status_code == 502
    assert len(client.calls) == 2
    assert vision_route == [
        ("firebase-user", main.ENDPOINT_COSTS["vision"], "vision_502")
    ]


def test_total_timeout_returns_504_and_refunds_once(monkeypatch, vision_route):
    async def never_returns():
        await asyncio.sleep(1)

    client = _FakeVisionClient([httpx.ReadTimeout("primary timed out"), never_returns])
    monkeypatch.setattr(main, "_http_client", client)
    monkeypatch.setattr(main, "VISION_TOTAL_DEADLINE_SECONDS", 0.03)
    monkeypatch.setattr(main, "VISION_PRIMARY_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.generate_cards_from_image(_request(), authorization="Bearer token")
        )

    assert error.value.status_code == 504
    assert len(client.calls) == 2
    assert vision_route == [
        ("firebase-user", main.ENDPOINT_COSTS["vision"], "vision_timeout")
    ]


def test_cancelled_request_refunds_once(monkeypatch, vision_route):
    async def cancelled(payload):
        raise asyncio.CancelledError()

    monkeypatch.setattr(main, "call_validated_vision_with_fallback", cancelled)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            main.generate_cards_from_image(_request(), authorization="Bearer token")
        )

    assert vision_route == [
        ("firebase-user", main.ENDPOINT_COSTS["vision"], "vision_cancelled")
    ]


def test_cancellation_during_credit_consumption_refunds(monkeypatch, vision_route):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_consume(uid, cost):
        started.set()
        await release.wait()
        return {"remaining": 40, "reset_at": 123456}

    async def run_cancelled_request():
        request_task = asyncio.create_task(
            main.generate_cards_from_image(_request(), authorization="Bearer token")
        )
        await started.wait()
        request_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    monkeypatch.setattr(main, "consume_credits", slow_consume)
    asyncio.run(run_cancelled_request())

    assert vision_route == [
        ("firebase-user", main.ENDPOINT_COSTS["vision"], "vision_cancelled")
    ]


def test_whitelisted_failure_does_not_refund(monkeypatch, vision_route):
    async def consume_whitelisted(uid, cost):
        return {"remaining": 9999, "reset_at": 0, "whitelisted": True}

    client = _FakeVisionClient([_completion("{"), _completion("{")])
    monkeypatch.setattr(main, "consume_credits", consume_whitelisted)
    monkeypatch.setattr(main, "_http_client", client)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.generate_cards_from_image(_request(), authorization="Bearer token")
        )

    assert error.value.status_code == 502
    assert vision_route == []


def test_output_card_contract_is_strict():
    assert (
        main.VisionOutput.model_validate(
            {"cards": [], "confidence": 0, "note": "No usable content"}
        ).cards
        == []
    )
    assert len(main.VisionOutput.model_validate(_valid_output()).cards) == 7

    for count in (1, 6, 16):
        with pytest.raises(ValidationError):
            main.VisionOutput.model_validate(
                {**_valid_output(), "cards": _cards(count)}
            )

    invalid_outputs = []
    empty_front = _valid_output()
    empty_front["cards"][0]["front"] = "   "
    invalid_outputs.append(empty_front)
    invalid_outputs.extend(
        [
            {**_valid_output(), "confidence": -1},
            {**_valid_output(), "confidence": 101},
            {**_valid_output(), "confidence": "94"},
            {**_valid_output(), "note": "n" * 501},
            {**_valid_output(), "unexpected": True},
        ]
    )
    duplicate_cards = _valid_output()
    duplicate_cards["cards"][1] = duplicate_cards["cards"][0]
    invalid_outputs.append(duplicate_cards)
    latex_card = _valid_output()
    latex_card["cards"][0]["back"] = r"F = m\omega^2r"
    invalid_outputs.append(latex_card)

    for output in invalid_outputs:
        with pytest.raises(ValidationError):
            main.VisionOutput.model_validate(output)

    long_card = _valid_output()
    long_card["cards"][0]["front"] = "f" * 500
    long_card["cards"][0]["back"] = "b" * 1_000
    validated = main.VisionOutput.model_validate(long_card)
    assert len(validated.cards[0].front) == 500
    assert len(validated.cards[0].back) == 1_000

    card_schema = main.VisionOutput.model_json_schema()["$defs"]["VisionCard"]
    assert "maxLength" not in card_schema["properties"]["front"]
    assert "maxLength" not in card_schema["properties"]["back"]
    assert (
        "front is a non-empty string of at most 100 characters"
        in main.VISION_SYSTEM_PROMPT
    )
    assert (
        "back is a non-empty string of at most 250 characters"
        in main.VISION_SYSTEM_PROMPT
    )


def test_vision_request_validates_base64_mime_and_data_uri(monkeypatch):
    gif_encoded = base64.b64encode(b"GIF89aimage").decode("ascii")
    raw = main.VisionRequest(image_data=gif_encoded, mime_type="image/gif")
    assert raw.image_data == gif_encoded
    assert raw.mime_type == "image/gif"

    webp_encoded = base64.b64encode(b"RIFFsizeWEBPimage").decode("ascii")
    data_uri = main.VisionRequest(image_data=f"data:image/webp;base64,{webp_encoded}")
    assert data_uri.image_data == webp_encoded
    assert data_uri.mime_type == "image/webp"

    invalid_requests = [
        {"image_data": gif_encoded, "mime_type": "image/heic"},
        {"image_data": "not-base64", "mime_type": "image/png"},
        {"image_data": "", "mime_type": "image/png"},
        {
            "image_data": f"data:image/webp;base64,{webp_encoded}",
            "mime_type": "image/png",
        },
        {"image_data": gif_encoded, "mime_type": "image/jpeg"},
    ]
    for request in invalid_requests:
        with pytest.raises(ValidationError):
            main.VisionRequest.model_validate(request)

    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 3)
    too_large = base64.b64encode(b"\xff\xd8\xffx").decode("ascii")
    with pytest.raises(ValidationError):
        main.VisionRequest(image_data=too_large, mime_type="image/jpeg")
