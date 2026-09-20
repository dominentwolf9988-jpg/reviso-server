import asyncio
import base64
from urllib.parse import quote_from_bytes, urlencode

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from admob_ssv import (
    InvalidSsvCallback,
    validate_reward_parameters,
    verify_admob_query,
)


def _signed_query(parameters: list[tuple[str, str]]):
    private_key = ec.generate_private_key(ec.SECP256R1())
    signed_content = urlencode(parameters).encode("ascii")
    signature = private_key.sign(signed_content, ec.ECDSA(hashes.SHA256()))
    encoded_signature = quote_from_bytes(
        base64.urlsafe_b64encode(signature).rstrip(b"="), safe=""
    ).encode("ascii")
    raw_query = signed_content + b"&signature=" + encoded_signature + b"&key_id=123"

    async def key_lookup(key_id: int):
        assert key_id == 123
        return private_key.public_key()

    return raw_query, key_lookup


def test_valid_signed_callback_preserves_parameters():
    raw_query, key_lookup = _signed_query(
        [
            ("ad_unit", "4774906413"),
            ("custom_data", "intent_with-special_data"),
            ("reward_amount", "5"),
            ("reward_item", "credits"),
            ("timestamp", "1780000000000"),
            ("transaction_id", "abcdef0123456789"),
            ("user_id", "firebase-user"),
        ]
    )

    result = asyncio.run(verify_admob_query(raw_query, key_lookup=key_lookup))

    assert result["custom_data"] == "intent_with-special_data"
    assert result["reward_amount"] == "5"
    assert result["user_id"] == "firebase-user"


def test_tampered_callback_is_rejected():
    raw_query, key_lookup = _signed_query(
        [("reward_amount", "5"), ("transaction_id", "abcdef0123456789")]
    )
    tampered = raw_query.replace(b"reward_amount=5", b"reward_amount=500")

    with pytest.raises(InvalidSsvCallback):
        asyncio.run(verify_admob_query(tampered, key_lookup=key_lookup))


def test_unsigned_parameters_after_key_id_are_rejected():
    raw_query, key_lookup = _signed_query(
        [("reward_amount", "5"), ("transaction_id", "abcdef0123456789")]
    )

    with pytest.raises(InvalidSsvCallback):
        asyncio.run(
            verify_admob_query(
                raw_query + b"&reward_amount=500",
                key_lookup=key_lookup,
            )
        )


def _valid_reward_parameters(timestamp: int = 1_780_000_000_000):
    return {
        "ad_unit": "4774906413",
        "custom_data": "intent_abcdef123456789",
        "reward_amount": "5",
        "reward_item": "credits",
        "timestamp": str(timestamp),
        "transaction_id": "abcdef0123456789",
        "user_id": "firebase-user",
    }


def _validate(parameters: dict[str, str]):
    return validate_reward_parameters(
        parameters,
        expected_ad_unit="ca-app-pub-123/4774906413",
        expected_reward_item="credits",
        expected_reward_amount=5,
        max_age_seconds=600,
        future_tolerance_seconds=120,
        now_ms=1_780_000_000_000,
    )


def test_reward_parameters_are_bound_to_expected_configuration():
    verified = _validate(_valid_reward_parameters())

    assert verified["intent_id"] == "intent_abcdef123456789"
    assert verified["callback_timestamp_ms"] == 1_780_000_000_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ad_unit", "wrong-unit"),
        ("reward_amount", "500"),
        ("reward_item", "coins"),
        ("transaction_id", "bad/id"),
        ("user_id", "bad/user"),
    ],
)
def test_unexpected_reward_parameters_are_rejected(field: str, value: str):
    parameters = _valid_reward_parameters()
    parameters[field] = value

    with pytest.raises(InvalidSsvCallback):
        _validate(parameters)


def test_old_callback_is_rejected():
    parameters = _valid_reward_parameters(timestamp=1_779_999_000_000)

    with pytest.raises(InvalidSsvCallback):
        _validate(parameters)


def test_microsecond_timestamp_is_normalized():
    parameters = _valid_reward_parameters(timestamp=1_780_000_000_000_000)

    verified = _validate(parameters)

    assert verified["callback_timestamp_ms"] == 1_780_000_000_000
