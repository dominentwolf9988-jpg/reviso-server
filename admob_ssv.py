"""AdMob rewarded-ad server-side verification helpers."""

import asyncio
import base64
import hmac
import re
import time
from typing import Awaitable, Callable
from urllib.parse import parse_qs, unquote_to_bytes

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


ADMOB_KEY_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
KEY_CACHE_SECONDS = 12 * 60 * 60


class InvalidSsvCallback(ValueError):
    """The callback is malformed or its signature is invalid."""


class SsvKeyUnavailable(RuntimeError):
    """Google's verification keys could not be loaded."""


class AdMobPublicKeyStore:
    def __init__(self) -> None:
        self._keys: dict[int, ec.EllipticCurvePublicKey] = {}
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def get(self, key_id: int) -> ec.EllipticCurvePublicKey:
        if time.monotonic() >= self._expires_at or key_id not in self._keys:
            await self._refresh()
        key = self._keys.get(key_id)
        if key is None:
            # Refresh once more for a key rotation that raced the first fetch.
            await self._refresh(force=True)
            key = self._keys.get(key_id)
        if key is None:
            raise InvalidSsvCallback(f"Unknown AdMob key_id: {key_id}")
        return key

    async def _refresh(self, *, force: bool = False) -> None:
        async with self._lock:
            if not force and self._keys and time.monotonic() < self._expires_at:
                return
            try:
                # Keep this client separate from the OpenRouter client so no
                # provider authorization header can leak to Google's key host.
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(ADMOB_KEY_URL)
                    response.raise_for_status()
                    payload = response.json()
                keys: dict[int, ec.EllipticCurvePublicKey] = {}
                for item in payload.get("keys", []):
                    key_id = int(item["keyId"])
                    public_key = serialization.load_pem_public_key(
                        item["pem"].encode("ascii")
                    )
                    if isinstance(public_key, ec.EllipticCurvePublicKey):
                        keys[key_id] = public_key
                if not keys:
                    raise ValueError("Google returned no ECDSA keys")
            except Exception as exc:
                raise SsvKeyUnavailable("Could not load AdMob verification keys") from exc
            self._keys = keys
            self._expires_at = time.monotonic() + KEY_CACHE_SECONDS


_public_keys = AdMobPublicKeyStore()
_safe_identifier = re.compile(r"^[A-Za-z0-9_-]{8,160}$")


async def warm_admob_keys() -> None:
    """Populate Google's key cache before the first callback arrives."""
    await _public_keys._refresh()


async def verify_admob_query(
    raw_query: bytes,
    *,
    key_lookup: Callable[[int], Awaitable[ec.EllipticCurvePublicKey]] | None = None,
) -> dict[str, str]:
    """Verify an untouched AdMob query and return its signed parameters."""
    signature_marker = b"&signature="
    key_marker = b"&key_id="
    signature_index = raw_query.find(signature_marker)
    if signature_index <= 0:
        raise InvalidSsvCallback("Missing signature")

    signed_content = raw_query[:signature_index]
    signature_and_key = raw_query[signature_index + len(signature_marker) :]
    key_index = signature_and_key.find(key_marker)
    if key_index <= 0:
        raise InvalidSsvCallback("Missing key_id")

    encoded_signature = signature_and_key[:key_index]
    encoded_key_id = signature_and_key[key_index + len(key_marker) :]
    if b"&" in encoded_key_id:
        raise InvalidSsvCallback("Unexpected unsigned parameters")

    try:
        key_id = int(encoded_key_id.decode("ascii"))
        signature_value = unquote_to_bytes(encoded_signature.decode("ascii"))
        signature_value += b"=" * (-len(signature_value) % 4)
        signature = base64.urlsafe_b64decode(signature_value)
    except (ValueError, UnicodeError) as exc:
        raise InvalidSsvCallback("Invalid signature encoding") from exc

    lookup = key_lookup or _public_keys.get
    public_key = await lookup(key_id)
    try:
        public_key.verify(signature, signed_content, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise InvalidSsvCallback("Invalid AdMob signature") from exc

    try:
        parsed = parse_qs(
            signed_content.decode("ascii"),
            keep_blank_values=True,
            strict_parsing=True,
        )
    except (UnicodeError, ValueError) as exc:
        raise InvalidSsvCallback("Invalid signed parameters") from exc
    if any(len(values) != 1 for values in parsed.values()):
        raise InvalidSsvCallback("Duplicate signed parameters")
    return {key: values[0] for key, values in parsed.items()}


def validate_reward_parameters(
    parameters: dict[str, str],
    *,
    expected_ad_unit: str,
    expected_reward_item: str,
    expected_reward_amount: int,
    max_age_seconds: int,
    future_tolerance_seconds: int,
    now_ms: int | None = None,
) -> dict[str, str | int]:
    required = {
        "ad_unit",
        "custom_data",
        "reward_amount",
        "reward_item",
        "timestamp",
        "transaction_id",
        "user_id",
    }
    if not required.issubset(parameters):
        raise InvalidSsvCallback("Missing required signed parameters")

    ad_unit = parameters["ad_unit"]
    accepted_ad_units = {expected_ad_unit, expected_ad_unit.rsplit("/", 1)[-1]}
    if not any(hmac.compare_digest(ad_unit, value) for value in accepted_ad_units):
        raise InvalidSsvCallback("Unexpected ad unit")
    if not hmac.compare_digest(parameters["reward_item"], expected_reward_item):
        raise InvalidSsvCallback("Unexpected reward item")

    try:
        reward_amount = int(parameters["reward_amount"])
        callback_timestamp_ms = int(parameters["timestamp"])
    except ValueError as exc:
        raise InvalidSsvCallback("Invalid numeric parameters") from exc
    if reward_amount != expected_reward_amount:
        raise InvalidSsvCallback("Unexpected reward amount")

    current_time_ms = now_ms or int(time.time() * 1000)
    # Google's documentation describes milliseconds, while older callback
    # examples contain microseconds. Normalize either representation.
    if callback_timestamp_ms > current_time_ms * 100:
        callback_timestamp_ms //= 1000
    if callback_timestamp_ms < current_time_ms - max_age_seconds * 1000:
        raise InvalidSsvCallback("Callback timestamp is too old")
    if callback_timestamp_ms > current_time_ms + future_tolerance_seconds * 1000:
        raise InvalidSsvCallback("Callback timestamp is in the future")

    intent_id = parameters["custom_data"]
    transaction_id = parameters["transaction_id"]
    uid = parameters["user_id"]
    if not _safe_identifier.fullmatch(intent_id):
        raise InvalidSsvCallback("Invalid reward intent")
    if not _safe_identifier.fullmatch(transaction_id):
        raise InvalidSsvCallback("Invalid transaction ID")
    if not uid or len(uid) > 128 or "/" in uid:
        raise InvalidSsvCallback("Invalid user ID")

    return {
        "uid": uid,
        "intent_id": intent_id,
        "transaction_id": transaction_id,
        "ad_unit": ad_unit,
        "callback_timestamp_ms": callback_timestamp_ms,
    }
