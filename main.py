"""
OpenRouter Secure Proxy — Production
Auth   : Firebase ID tokens (Google login) per user endpoint
Credits: Stored in Firestore, per Firebase UID, resets every 24 h
Admin  : x-api-key (APP_INTERNAL_AUTH_KEY) for credit top-up only
"""

import os
import base64
import hmac
import json
import logging
import asyncio
import re
import secrets
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from typing import Annotated, Optional, Union, List

import httpx
import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore as fb_fs
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

try:
    from .admob_ssv import (
        InvalidSsvCallback,
        SsvKeyUnavailable,
        validate_reward_parameters,
        verify_admob_query,
        warm_admob_keys,
    )
except ImportError:
    from admob_ssv import (
        InvalidSsvCallback,
        SsvKeyUnavailable,
        validate_reward_parameters,
        verify_admob_query,
        warm_admob_keys,
    )

try:
    from .notifications.models import (
        DeviceRegistrationRequest,
        DeviceUnregistrationRequest,
        NotificationEventBatchRequest,
        NotificationSettingsRequest,
        NotificationStateRequest,
    )
    from .notifications.repository import (
        AccountDeletionPending,
        InstallationOwnershipError,
        NotificationConflict,
        NotificationPolicyError,
    )
    from .notifications.service import NotificationService
except ImportError:
    from notifications.models import (
        DeviceRegistrationRequest,
        DeviceUnregistrationRequest,
        NotificationEventBatchRequest,
        NotificationSettingsRequest,
        NotificationStateRequest,
    )
    from notifications.repository import (
        AccountDeletionPending,
        InstallationOwnershipError,
        NotificationConflict,
        NotificationPolicyError,
    )
    from notifications.service import NotificationService

load_dotenv()

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("proxy")

# ---------------------------------------------------------------------------
# Config — hard-fail at startup
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY     = os.getenv("OPENROUTER_API_KEY", "")
ADMIN_API_KEY          = os.getenv("APP_INTERNAL_AUTH_KEY", "")   # admin endpoints only
FIREBASE_SA_JSON       = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
ALLOWED_ORIGINS        = os.getenv("ALLOWED_ORIGINS", "*").split(",")
REQUEST_TIMEOUT        = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_IMAGE_BYTES        = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))  # 15 MB
VISION_TOTAL_DEADLINE_SECONDS = 72.0
VISION_PRIMARY_TIMEOUT_SECONDS = 35.0
VISION_MAX_TOKENS = 3_000
# Text-to-cards generation: larger inputs can produce many cards, so the
# token budget and deadline are deliberately higher than vision's.
TEXT_CARDS_TOTAL_DEADLINE_SECONDS = 120.0
TEXT_CARDS_PRIMARY_TIMEOUT_SECONDS = 60.0
TEXT_CARDS_MAX_TOKENS = 20_000
# Technical safety cap (token budget / response size / worst-case cost).
# Business pricing stays per-card below.
TEXT_CARDS_MAX_COUNT = 300
TEXT_CARDS_BASE_FREE_COUNT = 20
TEXT_CARDS_SIZE_STEP = 5
TEXT_MODEL               = "google/gemini-2.5-flash-lite"
VISION_MODEL             = "google/gemini-2.5-flash-lite"
TEXT_FALLBACK_MODEL      = "qwen/qwen3.5-flash-02-23"
VISION_FALLBACK_MODEL    = "qwen/qwen3.5-flash-02-23" #"google/gemini-2.5-flash-lite"
BASE_URL = "https://openrouter.ai/api/v1"
# BASE_URL = "https://integrate.api.nvidia.com/v1"

# Credit rules
CREDITS_PER_PERIOD     = 50
CREDIT_REFRESH_SECONDS = 86_400   # 24 hours

REWARD_AMOUNT             = 5       # credits earned per rewarded ad watch
AD_REWARD_COOLDOWN_SECONDS = 15      # min seconds between reward claims
REWARD_INTENT_EXPIRY_SECONDS = 10 * 60
REWARD_INTENT_COOLDOWN_SECONDS = 5
RECENT_LOGIN_SECONDS = 5 * 60
AI_REPORT_COOLDOWN_SECONDS = 5
AI_REPORT_MAX_OUTPUT_LENGTH = 12_000
SSV_MAX_AGE_SECONDS = 10 * 60
SSV_FUTURE_TOLERANCE_SECONDS = 2 * 60
ADMOB_REWARDED_AD_UNIT_ID = os.getenv(
    "ADMOB_REWARDED_AD_UNIT_ID", "4774906413"
)
ADMOB_REWARD_ITEM = os.getenv("ADMOB_REWARD_ITEM", "credits")

ENDPOINT_COSTS = {
    "score":  5,
    "vision": 10,
    "chat":   5,
    "text":   5,   # base only; text cards add per-card charges post-generation
}

for name, val in [
    ("OPENROUTER_API_KEY",            OPENROUTER_API_KEY),
    ("APP_INTERNAL_AUTH_KEY",         ADMIN_API_KEY),
    ("FIREBASE_SERVICE_ACCOUNT_JSON", FIREBASE_SA_JSON),
]:
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")

# ---------------------------------------------------------------------------
# Firebase Admin SDK — one-time init
# ---------------------------------------------------------------------------
try:
    _fb_app = firebase_admin.get_app()          # already initialised (e.g. hot-reload)
    logger.info("Firebase Admin reused existing app")
except ValueError:
    _fb_app = firebase_admin.initialize_app(    # first run
        credentials.Certificate(json.loads(FIREBASE_SA_JSON))
    )
    logger.info("Firebase Admin initialised")

# Thread pool for Firestore (synchronous SDK) so we never block the event loop
_executor = ThreadPoolExecutor(max_workers=20)

# ---------------------------------------------------------------------------
# Shared async HTTP client for OpenRouter (created/closed in lifespan)
# ---------------------------------------------------------------------------
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    logger.info("Startup — creating shared HTTP client")
    _http_client = httpx.AsyncClient(
        # base_url="https://openrouter.ai/api/v1",
        base_url=BASE_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(connect=10.0, read=REQUEST_TIMEOUT, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    try:
        await warm_admob_keys()
    except SsvKeyUnavailable:
        logger.warning("AdMob verification keys will be fetched on first callback")
    yield
    logger.info("Shutdown — closing HTTP client")
    await _http_client.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="OpenRouter Secure Proxy", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization", "x-api-key"],
)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
class AccountDeletionInProgress(Exception):
    pass


def _deletion_requested_sync(uid: str) -> bool:
    return (
        fb_fs.client()
        .collection("account_deletions")
        .document(uid)
        .get()
        .exists
    )


async def _verify_token(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header. Expected: Bearer <firebase_id_token>",
        )
    token = authorization[7:]
    try:
        loop = asyncio.get_event_loop()
        decoded = await loop.run_in_executor(
            _executor, lambda: fb_auth.verify_id_token(token, check_revoked=True)
        )
        return decoded
    except fb_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Firebase token has expired")
    except fb_auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except Exception:
        logger.exception("Token verification failed")
        raise HTTPException(status_code=401, detail="Authentication failed")


async def get_uid(authorization: Optional[str]) -> str:
    """Verify an active Firebase user who is not being deleted."""
    decoded = await _verify_token(authorization)
    uid = decoded["uid"]
    loop = asyncio.get_event_loop()
    deleting = await loop.run_in_executor(_executor, _deletion_requested_sync, uid)
    if deleting:
        raise HTTPException(status_code=403, detail="Account deletion in progress")
    return uid


async def get_recent_uid(authorization: Optional[str]) -> str:
    """Verify a recent login for the idempotent deletion endpoint."""
    decoded = await _verify_token(authorization)
    now = int(datetime.now(timezone.utc).timestamp())
    auth_time = int(decoded.get("auth_time", 0))
    if auth_time <= 0 or now - auth_time > RECENT_LOGIN_SECONDS:
        raise HTTPException(status_code=401, detail="Recent login required")
    return decoded["uid"]


def verify_admin_key(provided: Optional[str]) -> None:
    """Constant-time check for internal admin key (used by credit top-up only)."""
    if not provided:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    if not hmac.compare_digest(ADMIN_API_KEY.encode(), provided.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Credit system — all Firestore calls run in _executor
# ---------------------------------------------------------------------------
#
# Firestore document path: user_credits/{uid}
# Fields:
#   remaining  int   credits left in the current 24 h window
#   reset_at   int   Unix timestamp (UTC) when the window resets
#   created_at int   first-seen timestamp
#
def _consume_sync(uid: str, cost: int) -> dict:
    """
    Atomically deduct `cost` credits from the user's balance.
    Returns {"remaining": int, "reset_at": int, "ok": bool}
    """
    db  = fb_fs.client()
    ref = db.collection("user_credits").document(uid)
    deletion_ref = db.collection("account_deletions").document(uid)
    now = int(datetime.now(timezone.utc).timestamp())

    @fb_fs.transactional
    def _txn(transaction, ref, deletion_ref):
        if deletion_ref.get(transaction=transaction).exists:
            return {"ok": False, "deleted": True}
        snap = ref.get(transaction=transaction)

        if not snap.exists:
            # First-time user — provision with full credits minus this call
            reset_at      = now + CREDIT_REFRESH_SECONDS
            new_remaining = CREDITS_PER_PERIOD - cost
            transaction.set(ref, {
                "remaining":  new_remaining,
                "reset_at":   reset_at,
                "created_at": now,
            })
            return {"remaining": new_remaining, "reset_at": reset_at, "ok": True}

        d         = snap.to_dict()
        remaining = d.get("remaining", CREDITS_PER_PERIOD)
        reset_at  = d.get("reset_at",  now + CREDIT_REFRESH_SECONDS)

        # Auto-refresh if the 24 h window has passed
        if now >= reset_at:
            if remaining < CREDITS_PER_PERIOD:
                remaining = CREDITS_PER_PERIOD
            reset_at  = now + CREDIT_REFRESH_SECONDS

        if remaining < cost:
            return {"remaining": remaining, "reset_at": reset_at, "ok": False}

        new_remaining = remaining - cost
        transaction.update(ref, {"remaining": new_remaining, "reset_at": reset_at})
        return {"remaining": new_remaining, "reset_at": reset_at, "ok": True}

    return _txn(db.transaction(), ref, deletion_ref)


def _read_sync(uid: str) -> dict:
    """Read current balance without consuming any credits."""
    db   = fb_fs.client()
    ref  = db.collection("user_credits").document(uid)
    snap = ref.get()
    now  = int(datetime.now(timezone.utc).timestamp())

    if not snap.exists:
        return {"remaining": CREDITS_PER_PERIOD, "reset_at": now + CREDIT_REFRESH_SECONDS}

    d         = snap.to_dict()
    remaining = d.get("remaining", CREDITS_PER_PERIOD)
    reset_at  = d.get("reset_at",  now + CREDIT_REFRESH_SECONDS)

    if now >= reset_at:
        final = remaining if remaining >= CREDITS_PER_PERIOD else CREDITS_PER_PERIOD
        return {"remaining": final, "reset_at": now + CREDIT_REFRESH_SECONDS}

    return {"remaining": remaining, "reset_at": reset_at}


def _add_sync(uid: str, amount: int) -> dict:
    """
    Add `amount` credits to uid's balance.
    Called after a verified ad-reward event. Can exceed the daily cap intentionally.
    """
    db = fb_fs.client()
    ref = db.collection("user_credits").document(uid)
    deletion_ref = db.collection("account_deletions").document(uid)
    now = int(datetime.now(timezone.utc).timestamp())

    @fb_fs.transactional
    def _txn(transaction, ref, deletion_ref):
        if deletion_ref.get(transaction=transaction).exists:
            raise AccountDeletionInProgress()
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            reset_at = now + CREDIT_REFRESH_SECONDS
            new_remaining = CREDITS_PER_PERIOD + amount
            transaction.set(
                ref,
                {
                    "remaining": new_remaining,
                    "reset_at": reset_at,
                    "created_at": now,
                },
            )
            return {"remaining": new_remaining, "reset_at": reset_at}

        data = snap.to_dict() or {}
        remaining = data.get("remaining", CREDITS_PER_PERIOD)
        reset_at = data.get("reset_at", now + CREDIT_REFRESH_SECONDS)
        if now >= reset_at:
            remaining = max(remaining, CREDITS_PER_PERIOD)
            reset_at = now + CREDIT_REFRESH_SECONDS

        new_remaining = remaining + amount
        transaction.update(
            ref,
            {"remaining": new_remaining, "reset_at": reset_at},
        )
        return {"remaining": new_remaining, "reset_at": reset_at}

    return _txn(db.transaction(), ref, deletion_ref)


def _is_whitelisted_sync(uid: str) -> bool:
    """
    Check if uid exists in the whitelisted_users Firestore collection.
    Whitelisted users bypass all credit checks — unlimited usage.
    Document presence = whitelisted. No fields needed.
    """
    try:
        db  = fb_fs.client()
        ref = db.collection("whitelisted_users").document(uid)
        return ref.get().exists
    except Exception:
        # On any Firestore error, fail safe — treat as NOT whitelisted.
        # Never grant unlimited access on an uncertain read.
        logger.exception("Whitelist check failed uid=%s — defaulting to not whitelisted", uid)
        return False


async def is_whitelisted(uid: str) -> bool:
    """Async wrapper for whitelist check."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _is_whitelisted_sync, uid)


def _claim_reward_sync(uid: str, reward_amount: int = 0) -> dict:
    """
    Atomically check cooldown and add reward credits.
    Uses REWARD_AMOUNT unless reward_amount is specified and positive.

    Returns {"remaining": int, "reset_at": int, "ok": bool}
    When ok=False, also returns {"retry_after": int} (seconds until allowed).
    """
    grant = reward_amount if reward_amount > 0 else REWARD_AMOUNT

    db  = fb_fs.client()
    ref = db.collection("user_credits").document(uid)
    now = int(datetime.now(timezone.utc).timestamp())

    @fb_fs.transactional
    def _txn(transaction, ref):
        snap = ref.get(transaction=transaction)

        if not snap.exists:
            reset_at      = now + CREDIT_REFRESH_SECONDS
            new_remaining = CREDITS_PER_PERIOD + grant
            transaction.set(ref, {
                "remaining":      new_remaining,
                "reset_at":       reset_at,
                "created_at":     now,
                "last_reward_at": now,
            })
            return {"remaining": new_remaining, "reset_at": reset_at, "ok": True}

        d            = snap.to_dict()
        remaining    = d.get("remaining", CREDITS_PER_PERIOD)
        reset_at     = d.get("reset_at",  now + CREDIT_REFRESH_SECONDS)
        last_reward  = d.get("last_reward_at", 0)

        # Cooldown check
        elapsed = now - last_reward
        if elapsed < AD_REWARD_COOLDOWN_SECONDS:
            return {
                "ok":          False,
                "remaining":   remaining,
                "reset_at":    reset_at,
                "retry_after": AD_REWARD_COOLDOWN_SECONDS - elapsed,
            }

        # Auto-refresh if the 24 h window has passed
        if now >= reset_at:
            if remaining < CREDITS_PER_PERIOD:
                remaining = CREDITS_PER_PERIOD
            reset_at  = now + CREDIT_REFRESH_SECONDS

        new_remaining = remaining + grant
        transaction.update(ref, {
            "remaining":      new_remaining,
            "reset_at":       reset_at,
            "last_reward_at": now,
        })
        return {"remaining": new_remaining, "reset_at": reset_at, "ok": True}

    return _txn(db.transaction(), ref)


# Async wrappers (offload sync Firestore to thread pool)
async def consume_credits(uid: str, cost: int) -> dict:
    # Whitelisted users (developers/testers) bypass all credit checks.
    # Check whitelist first before hitting the credits collection.
    if await is_whitelisted(uid):
        logger.info("Whitelist bypass uid=%s cost=%d", uid, cost)
        return {"remaining": 9999, "reset_at": 0, "whitelisted": True}

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _consume_sync, uid, cost)

    if result.get("deleted"):
        raise HTTPException(status_code=403, detail="Account deletion in progress")
    if not result["ok"]:
        secs_left = max(0, result["reset_at"] - int(datetime.now(timezone.utc).timestamp()))
        raise HTTPException(
            status_code=402,
            detail={
                "error":     "insufficient_credits",
                "remaining": result["remaining"],
                "cost":      cost,
                "reset_at":  result["reset_at"],
                "resets_in": f"{secs_left // 3600}h {(secs_left % 3600) // 60}m",
                "message": (
                    f"This action costs {cost} credits but you only have "
                    f"{result['remaining']}. Credits reset in "
                    f"{secs_left // 3600}h {(secs_left % 3600) // 60}m, "
                    "or watch an ad to top up."
                ),
            },
        )
    return result


async def try_consume_credits(uid: str, cost: int) -> dict:
    """Best-effort consume for post-generation charges.

    Unlike consume_credits, never raises on insufficient balance: returns
    {"ok": True, "remaining": ..., "reset_at": ...} on success and
    {"ok": False, "remaining": ..., "reset_at": ...} when the balance is
    too low. Callers decide whether to deliver anyway and log the gap.
    """
    if await is_whitelisted(uid):
        logger.info("Whitelist bypass uid=%s cost=%d", uid, cost)
        return {"remaining": 9999, "reset_at": 0, "whitelisted": True, "ok": True}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _consume_sync, uid, cost)
    if result.get("deleted"):
        raise HTTPException(status_code=403, detail="Account deletion in progress")
    return {
        "remaining": result["remaining"],
        "reset_at": result["reset_at"],
        "ok": bool(result["ok"]),
    }


async def read_credits(uid: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _read_sync, uid)


async def add_credits(uid: str, amount: int) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _add_sync, uid, amount)


async def claim_reward(uid: str, reward_amount: int = 0) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _claim_reward_sync, uid, reward_amount)


def _create_reward_intent_sync(uid: str) -> dict:
    db = fb_fs.client()
    now = int(datetime.now(timezone.utc).timestamp())
    intent_id = secrets.token_urlsafe(32)
    intent_ref = db.collection("ad_reward_intents").document(intent_id)
    state_ref = db.collection("ad_reward_users").document(uid)
    deletion_ref = db.collection("account_deletions").document(uid)

    @fb_fs.transactional
    def _txn(transaction, state_ref, intent_ref, deletion_ref):
        if deletion_ref.get(transaction=transaction).exists:
            return {"ok": False, "deleted": True}
        state = state_ref.get(transaction=transaction)
        last_intent_at = (state.to_dict() or {}).get("last_intent_at", 0)
        elapsed = now - last_intent_at
        if elapsed < REWARD_INTENT_COOLDOWN_SECONDS:
            return {
                "ok": False,
                "retry_after": REWARD_INTENT_COOLDOWN_SECONDS - elapsed,
            }
        transaction.set(state_ref, {"last_intent_at": now}, merge=True)
        transaction.set(
            intent_ref,
            {
                "uid": uid,
                "status": "pending",
                "created_at": now,
                "expires_at": now + REWARD_INTENT_EXPIRY_SECONDS,
            },
        )
        return {
            "ok": True,
            "intent_id": intent_id,
            "expires_at": now + REWARD_INTENT_EXPIRY_SECONDS,
        }

    return _txn(db.transaction(), state_ref, intent_ref, deletion_ref)


def _read_reward_intent_sync(uid: str, intent_id: str) -> Optional[dict]:
    db = fb_fs.client()
    snapshot = db.collection("ad_reward_intents").document(intent_id).get()
    if not snapshot.exists:
        return None
    data = snapshot.to_dict() or {}
    if data.get("uid") != uid:
        return None
    if (
        data.get("status") == "pending"
        and int(data.get("expires_at", 0))
        < int(datetime.now(timezone.utc).timestamp())
    ):
        return {"status": "expired"}
    return data


def _grant_verified_reward_sync(
    uid: str,
    intent_id: str,
    transaction_id: str,
    ad_unit: str,
    callback_timestamp_ms: int,
) -> dict:
    db = fb_fs.client()
    now = int(datetime.now(timezone.utc).timestamp())
    intent_ref = db.collection("ad_reward_intents").document(intent_id)
    reward_ref = db.collection("admob_ssv_transactions").document(transaction_id)
    credit_ref = db.collection("user_credits").document(uid)
    deletion_ref = db.collection("account_deletions").document(uid)

    @fb_fs.transactional
    def _txn(transaction, intent_ref, reward_ref, credit_ref, deletion_ref):
        if deletion_ref.get(transaction=transaction).exists:
            return {"status": "ignored", "reason": "account_deletion"}
        reward_snapshot = reward_ref.get(transaction=transaction)
        if reward_snapshot.exists:
            return {"status": "duplicate"}

        intent_snapshot = intent_ref.get(transaction=transaction)
        if not intent_snapshot.exists:
            return {"status": "ignored", "reason": "unknown_intent"}
        intent = intent_snapshot.to_dict() or {}
        if intent.get("uid") != uid:
            return {"status": "ignored", "reason": "uid_mismatch"}
        if intent.get("status") != "pending":
            return {"status": "ignored", "reason": "intent_not_pending"}
        if int(intent.get("expires_at", 0)) < now:
            return {"status": "ignored", "reason": "intent_expired"}

        credit_snapshot = credit_ref.get(transaction=transaction)
        if credit_snapshot.exists:
            credit = credit_snapshot.to_dict() or {}
            remaining = int(credit.get("remaining", CREDITS_PER_PERIOD))
            reset_at = int(
                credit.get("reset_at", now + CREDIT_REFRESH_SECONDS)
            )
            if now >= reset_at:
                remaining = max(remaining, CREDITS_PER_PERIOD)
                reset_at = now + CREDIT_REFRESH_SECONDS
            new_remaining = remaining + REWARD_AMOUNT
            transaction.update(
                credit_ref,
                {
                    "remaining": new_remaining,
                    "reset_at": reset_at,
                    "last_reward_at": now,
                },
            )
        else:
            reset_at = now + CREDIT_REFRESH_SECONDS
            new_remaining = CREDITS_PER_PERIOD + REWARD_AMOUNT
            transaction.set(
                credit_ref,
                {
                    "remaining": new_remaining,
                    "reset_at": reset_at,
                    "created_at": now,
                    "last_reward_at": now,
                },
            )

        transaction.set(
            reward_ref,
            {
                "uid": uid,
                "intent_id": intent_id,
                "reward_amount": REWARD_AMOUNT,
                "ad_unit": ad_unit,
                "callback_timestamp_ms": callback_timestamp_ms,
                "processed_at": now,
            },
        )
        transaction.update(
            intent_ref,
            {
                "status": "credited",
                "transaction_id": transaction_id,
                "credited_at": now,
                "remaining": new_remaining,
                "reset_at": reset_at,
            },
        )
        return {
            "status": "credited",
            "remaining": new_remaining,
            "reset_at": reset_at,
        }

    return _txn(
        db.transaction(),
        intent_ref,
        reward_ref,
        credit_ref,
        deletion_ref,
    )


async def create_reward_intent(uid: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _create_reward_intent_sync, uid)


async def read_reward_intent(uid: str, intent_id: str) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _read_reward_intent_sync, uid, intent_id
    )


async def grant_verified_reward(
    uid: str,
    intent_id: str,
    transaction_id: str,
    ad_unit: str,
    callback_timestamp_ms: int,
) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _grant_verified_reward_sync,
        uid,
        intent_id,
        transaction_id,
        ad_unit,
        callback_timestamp_ms,
    )


_safe_identifier = re.compile(r"^[A-Za-z0-9_-]{8,160}$")


def validate_ssv_parameters(parameters: dict[str, str]) -> dict[str, Union[str, int]]:
    return validate_reward_parameters(
        parameters,
        expected_ad_unit=ADMOB_REWARDED_AD_UNIT_ID,
        expected_reward_item=ADMOB_REWARD_ITEM,
        expected_reward_amount=REWARD_AMOUNT,
        max_age_seconds=SSV_MAX_AGE_SECONDS,
        future_tolerance_seconds=SSV_FUTURE_TOLERANCE_SECONDS,
    )


async def _refund(uid: str, amount: int, reason: str) -> None:
    """Best-effort credit refund when our server or OpenRouter fails. Never raises."""
    try:
        await add_credits(uid, amount)
        logger.info("Refunded %d credits uid=%s reason=%s", amount, uid, reason)
    except AccountDeletionInProgress:
        logger.info("Refund skipped during account deletion uid=%s", uid)
    except Exception:
        logger.exception("Refund failed uid=%s amount=%d", uid, amount)


def _create_ai_report_sync(uid: str, feature: str, reason: str, output: str) -> dict:
    db = fb_fs.client()
    now = int(datetime.now(timezone.utc).timestamp())
    report_id = secrets.token_urlsafe(24)
    report_ref = db.collection("ai_reports").document(report_id)
    state_ref = db.collection("ai_report_users").document(uid)
    deletion_ref = db.collection("account_deletions").document(uid)

    @fb_fs.transactional
    def _txn(transaction, report_ref, state_ref, deletion_ref):
        if deletion_ref.get(transaction=transaction).exists:
            return {"ok": False, "deleted": True}
        state = state_ref.get(transaction=transaction)
        last_report_at = int((state.to_dict() or {}).get("last_report_at", 0))
        elapsed = now - last_report_at
        if elapsed < AI_REPORT_COOLDOWN_SECONDS:
            return {
                "ok": False,
                "retry_after": AI_REPORT_COOLDOWN_SECONDS - elapsed,
            }
        transaction.set(state_ref, {"last_report_at": now}, merge=True)
        transaction.set(
            report_ref,
            {
                "uid": uid,
                "feature": feature,
                "reason": reason,
                "output": output,
                "created_at": now,
                "status": "open",
            },
        )
        return {"ok": True, "report_id": report_id}

    return _txn(db.transaction(), report_ref, state_ref, deletion_ref)


async def create_ai_report(uid: str, feature: str, reason: str, output: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _create_ai_report_sync,
        uid,
        feature,
        reason,
        output,
    )


def _delete_query_sync(db, collection_name: str, uid: str) -> int:
    deleted = 0
    while True:
        documents = list(
            db.collection(collection_name)
            .where("uid", "==", uid)
            .limit(400)
            .stream()
        )
        if not documents:
            return deleted
        batch = db.batch()
        for document in documents:
            batch.delete(document.reference)
        batch.commit()
        deleted += len(documents)


def _begin_account_deletion_sync(uid: str) -> dict:
    db = fb_fs.client()
    tombstone_ref = db.collection("account_deletions").document(uid)
    snapshot = tombstone_ref.get()
    if snapshot.exists:
        return {"status": (snapshot.to_dict() or {}).get("status", "drive_pending")}

    now = int(datetime.now(timezone.utc).timestamp())
    tombstone_ref.set(
        {
            "status": "drive_pending",
            "requested_at": now,
            "updated_at": now,
        }
    )
    return {"status": "drive_pending"}


async def begin_account_deletion(uid: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _begin_account_deletion_sync, uid)


def _delete_account_data_sync(uid: str) -> dict:
    """Idempotently remove backend data, retaining a permanent suppression marker."""
    db = fb_fs.client()
    tombstone_ref = db.collection("account_deletions").document(uid)
    now = int(datetime.now(timezone.utc).timestamp())
    snapshot = tombstone_ref.get()
    existing = snapshot.to_dict() or {}
    already_deleted = existing.get("status") == "backend_deleted"

    processing = {"status": "processing", "updated_at": now}
    if not snapshot.exists:
        processing["requested_at"] = now
    tombstone_ref.set(processing, merge=True)

    for collection_name in (
        "users",
        "user_credits",
        "whitelisted_users",
        "ad_reward_users",
        "ai_report_users",
        "notification_states",
    ):
        db.collection(collection_name).document(uid).delete()

    deleted_documents = 0
    for collection_name in (
        "ad_reward_intents",
        "admob_ssv_transactions",
        "ai_reports",
        "notification_devices",
        "notification_events",
        "notification_jobs",
        "notification_sends",
    ):
        deleted_documents += _delete_query_sync(db, collection_name, uid)

    completed_at = int(datetime.now(timezone.utc).timestamp())
    tombstone_ref.set(
        {
            "status": "backend_deleted",
            "completed_at": completed_at,
            "updated_at": completed_at,
        },
        merge=True,
    )
    logger.info(
        "Backend account data deleted uid=%s query_documents=%d",
        uid,
        deleted_documents,
    )
    return {"status": "backend_deleted", "already_deleted": already_deleted}


async def delete_account_data(uid: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _delete_account_data_sync, uid)


# ---------------------------------------------------------------------------
# OpenRouter call with exponential-backoff retry on 429 / 5xx
# ---------------------------------------------------------------------------
class _RetryableError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(_RetryableError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def call_openrouter(payload: dict) -> str:
    print(payload["model"])
    resp = await _http_client.post("/chat/completions", json=payload)
    if resp.status_code == 429 or resp.status_code >= 500:
        logger.warning("OpenRouter %s — will retry", resp.status_code)
        raise _RetryableError(f"OpenRouter {resp.status_code}")
    if resp.status_code >= 400:
        logger.error("OpenRouter 4xx %s: %s", resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail="Upstream API error")
    return resp.json()["choices"][0]["message"]["content"]


async def call_with_fallback(payload: dict, primary_model: str, fallback_model: str) -> str:
    payload["model"] = primary_model
    try:
        return await call_openrouter(payload)
    except Exception:
        logger.warning(
            "Primary %s failed — falling back to %s", primary_model, fallback_model
        )
    payload["model"] = fallback_model
    return await call_openrouter(payload)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SCORE_SYSTEM_PROMPT = """You are a strict, objective scoring agent. Follow these instructions exactly.

You will receive:

* REFERENCE_ANSWER: the authoritative/correct answer.
* STUDENT_ANSWER: the user's answer to be scored.

TASK:

1. Compare STUDENT_ANSWER to REFERENCE_ANSWER and assign a score (0-100 integer) for each of the following **nine** criteria:

  * **Accuracy / Meaning** (weight 30): Correctness of facts and whether the answer conveys the intended meaning.
  * **Completeness / Relevance** (weight 15): Inclusion of essential elements from the reference and relevance to the question.
  * **Clarity & Conciseness** (weight 12): Clear, direct phrasing without unnecessary words.
  * **Grammar & Spelling** (weight 8): Correct grammar, punctuation, and spelling.
  * **Style & Tone** (weight 8): Appropriate register and readability for the expected audience.
  * **Originality / Paraphrase Quality** (weight 7): Uses own phrasing where appropriate; not a verbatim copy unless required.
  * **Structure & Organization** (weight 8): Logical flow, sentence ordering, and use of paragraphs or lists when helpful.
  * **Terminology & Precision** (weight 7): Correct use of domain-specific terms and precise wording.
  * **Evidence / Examples (when applicable)** (weight 5): Use of relevant examples or brief evidence if they strengthen the answer.

2. Compute the final weighted percentage:
  final_score = round(
  0.30*Accuracy + 0.15*Completeness + 0.12*Clarity + 0.08*Grammar +
  0.08*Style + 0.07*Originality + 0.08*Structure + 0.07*Terminology + 0.05*Evidence
  )

3. if the user answer and the reference answer is same give a 100% score

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

Output ONLY a single JSON object. No other text, code fences, markdown, or explanation.

Schema:
{"accuracy":<int>,"completeness":<int>,"clarity":<int>,"grammar":<int>,"style":<int>,"originality":<int>,"structure":<int>,"terminology":<int>,"evidence":<int>,"note":"<str>","score":<int>}

Rules:
- All score fields are integers 0-100 except "note" (string, <=140 chars).
- The "note" field MUST be a helpful study tip about the question's topic
  (e.g., mnemonic, key insight, memory aid, common misconception, related concept).
  Do NOT comment on the student's answer quality.
- Do not use the note to generate hateful, sexually exploitative, self-harm,
  violent wrongdoing, illegal activity, or other dangerous instructions.
- The "score" field MUST be the last field.
- It is the final weighted percentage (same formula as before).

If you cannot comply: {"accuracy":0,"completeness":0,"clarity":0,"grammar":0,"style":0,"originality":0,"structure":0,"terminology":0,"evidence":0,"note":"unable to score","score":0}

END OF SYSTEM INSTRUCTIONS."""

VISION_SYSTEM_PROMPT = """You create educational flashcards from one supplied image.

GROUNDING RULES:
- Use only information visible in the image or directly required to clarify that information.
- Correct obvious OCR mistakes when the intended text is clear.
- Do not invent missing facts, answers, examples, or unrelated content.
- If there is too little usable image content for 7 distinct grounded cards, return the fallback with zero cards instead of padding or guessing.
- Do not generate hateful, sexually exploitative, self-harm, violent wrongdoing, illegal activity, or other dangerous instructions. Return the unsafe fallback when the image cannot be transformed safely.

TEXT RULES:
- Card text must be plain text and Unicode only.
- Write formulas with ordinary text and Unicode symbols such as ×, ÷, √, ≤, ≥, ², and →.
- Do not use LaTeX, backslash commands, Markdown, headings, bullets, tables, links, or code formatting in card text.
- Correct spelling and punctuation without changing the image's meaning.

OUTPUT REQUIREMENTS:
- Return exactly one JSON object and nothing else. Do not use code fences.
- Use exactly these top-level keys in this order: cards, confidence, note.
- cards must contain either zero cards or 7 to 15 cards. Never return 1 to 6 cards.
- Every card must contain exactly front and back. front is a non-empty string of at most 100 characters. back is a non-empty string of at most 250 characters.
- confidence is an integer from 0 to 100 indicating confidence that the cards accurately reflect the image.
- note is a non-empty plain-text string of at most 500 characters.
- A successful result must contain 7 to 15 distinct, useful cards.

Schema: {"cards":[{"front":"string","back":"string"}],"confidence":0,"note":"string"}
Fallback: {"cards":[],"confidence":0,"note":"No usable content"}
Unsafe fallback: {"cards":[],"confidence":0,"note":"Content could not be processed safely"}"""

TEXT_CARDS_SYSTEM_PROMPT = """You create educational flashcards from one block of supplied text.

GROUNDING RULES:
- Use only information present in the supplied text or directly required to clarify it.
- Do not invent missing facts, answers, examples, or unrelated content.
- Split the material into as many distinct, useful cards as the text genuinely supports. Cover the whole text instead of repeating one idea.
- If the text has too little usable content, return the fallback with zero cards instead of padding or guessing.
- Do not generate hateful, sexually exploitative, self-harm, violent wrongdoing, illegal activity, or other dangerous instructions. Return the unsafe fallback when the text cannot be transformed safely.

TEXT RULES:
- Card text must be plain text and Unicode only.
- Write formulas with ordinary text and Unicode symbols such as ×, ÷, √, ≤, ≥, ², and →.
- Do not use LaTeX, backslash commands, Markdown, headings, bullets, tables, links, or code formatting in card text.
- Keep facts faithful to the source. Correct obvious typos without changing meaning.

OUTPUT REQUIREMENTS:
- Return exactly one JSON object and nothing else. Do not use code fences.
- Use exactly these top-level keys in this order: cards, confidence, note.
- cards must contain either zero cards or 1 to 300 cards. Never return more than 300 cards.
- Every card must contain exactly front and back. front is a non-empty string of at most 100 characters. back is a non-empty string of at most 250 characters.
- confidence is an integer from 0 to 100 indicating confidence that the cards accurately reflect the supplied text.
- note is a non-empty plain-text string of at most 500 characters.
- Cards must be distinct. Do not produce two cards with the same question and answer.

Schema: {"cards":[{"front":"string","back":"string"}],"confidence":0,"note":"string"}
Fallback: {"cards":[],"confidence":0,"note":"No usable content"}
Unsafe fallback: {"cards":[],"confidence":0,"note":"Content could not be processed safely"}"""


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    question: str
    actual_answer: str
    user_answer: str

    @field_validator("question", "actual_answer", "user_answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        if len(v) > 20_000:
            raise ValueError("Field exceeds 20 000 character limit")
        return v.strip()


class MessageContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    mime_type: Optional[str] = None


VISION_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    }
)
_VISION_DATA_URI = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)


def _matches_vision_mime(data: bytes, mime_type: str) -> bool:
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/webp":
        return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if mime_type == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    return False


class VisionRequest(BaseModel):
    image_data: str
    mime_type: str = "image/jpeg"

    @field_validator("mime_type")
    @classmethod
    def valid_mime_type(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in VISION_MIME_TYPES:
            raise ValueError("Unsupported vision image MIME type")
        return value

    @model_validator(mode="after")
    def valid_image_data(self):
        encoded = self.image_data.strip()
        if encoded.startswith("data:"):
            match = _VISION_DATA_URI.fullmatch(encoded)
            if match is None:
                raise ValueError("image_data must be a base64 data URI")
            data_uri_mime, encoded = match.groups()
            data_uri_mime = data_uri_mime.lower()
            if data_uri_mime not in VISION_MIME_TYPES:
                raise ValueError("Unsupported vision image MIME type")
            if (
                "mime_type" in self.model_fields_set
                and self.mime_type != data_uri_mime
            ):
                raise ValueError("mime_type does not match image_data data URI")
            self.mime_type = data_uri_mime

        if not encoded:
            raise ValueError("image_data must not be empty")
        max_encoded_length = ((MAX_IMAGE_BYTES + 2) // 3) * 4
        if len(encoded) > max_encoded_length:
            raise ValueError(f"Decoded image exceeds {MAX_IMAGE_BYTES} byte limit")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image_data must be valid base64") from exc
        if not decoded:
            raise ValueError("image_data must decode to non-empty data")
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError(f"Decoded image exceeds {MAX_IMAGE_BYTES} byte limit")
        if not _matches_vision_mime(decoded, self.mime_type):
            raise ValueError("mime_type does not match the decoded image")

        # Store one representation so the upstream payload cannot disagree with MIME validation.
        self.image_data = encoded
        return self


class TextCardsRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        stripped = value.strip()
        if len(stripped) < 200:
            raise ValueError("text must be at least 200 characters")
        if len(stripped) > 20_000:
            raise ValueError("text exceeds 20 000 character limit")
        return stripped


_VisionCardText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
_VisionNote = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=500),
]


class VisionCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    front: _VisionCardText
    back: _VisionCardText

    @field_validator("front", "back")
    @classmethod
    def plain_text_only(cls, value: str) -> str:
        if "\\" in value or "`" in value:
            raise ValueError("Card text must not contain LaTeX or Markdown")
        return value


_VisionCards = Union[
    Annotated[List[VisionCard], Field(max_length=0)],
    Annotated[List[VisionCard], Field(min_length=7, max_length=15)],
]


class VisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cards: _VisionCards
    confidence: Annotated[int, Field(strict=True, ge=0, le=100)]
    note: _VisionNote

    @field_validator("cards")
    @classmethod
    def distinct_cards(cls, cards: List[VisionCard]) -> List[VisionCard]:
        normalized = {
            (
                " ".join(card.front.casefold().split()),
                " ".join(card.back.casefold().split()),
            )
            for card in cards
        }
        if len(normalized) != len(cards):
            raise ValueError("Cards must be distinct")
        return cards


_TextCards = Union[
    Annotated[List[VisionCard], Field(max_length=0)],
    Annotated[
        List[VisionCard],
        Field(min_length=1, max_length=TEXT_CARDS_MAX_COUNT),
    ],
]


class TextCardsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cards: _TextCards
    confidence: Annotated[int, Field(strict=True, ge=0, le=100)]
    note: _VisionNote

    @field_validator("cards")
    @classmethod
    def distinct_cards(cls, cards: List[VisionCard]) -> List[VisionCard]:
        normalized = {
            (
                " ".join(card.front.casefold().split()),
                " ".join(card.back.casefold().split()),
            )
            for card in cards
        }
        if len(normalized) != len(cards):
            raise ValueError("Cards must be distinct")
        return cards


def text_cards_cost(card_count: int) -> int:
    """Base cost for up to 20 cards, then 1 credit per 5 extra cards."""
    if card_count <= TEXT_CARDS_BASE_FREE_COUNT:
        return ENDPOINT_COSTS["text"]
    extra_cards = card_count - TEXT_CARDS_BASE_FREE_COUNT
    return ENDPOINT_COSTS["text"] + (
        extra_cards + TEXT_CARDS_SIZE_STEP - 1
    ) // TEXT_CARDS_SIZE_STEP


class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContentItem]]

class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def check_messages(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        if len(v) > 50:
            raise ValueError("Too many messages (max 50)")
        return v


class AddCreditsRequest(BaseModel):
    uid: str
    amount: int

    @field_validator("uid")
    @classmethod
    def uid_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("uid must not be empty")
        return v.strip()

    @field_validator("amount")
    @classmethod
    def valid_amount(cls, v: int) -> int:
        if v <= 0 or v > 500:
            raise ValueError("amount must be between 1 and 500")
        return v


class AiReportRequest(BaseModel):
    feature: str
    reason: str
    output: str

    @field_validator("feature")
    @classmethod
    def valid_feature(cls, value: str) -> str:
        if value not in {"image_cards", "text_cards", "answer_scoring"}:
            raise ValueError("Unsupported AI feature")
        return value

    @field_validator("reason")
    @classmethod
    def valid_reason(cls, value: str) -> str:
        if value not in {"unsafe", "incorrect", "irrelevant", "other"}:
            raise ValueError("Unsupported report reason")
        return value

    @field_validator("output")
    @classmethod
    def valid_output(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Reported output must not be empty")
        if len(value) > AI_REPORT_MAX_OUTPUT_LENGTH:
            raise ValueError("Reported output is too large")
        return value


# ---------------------------------------------------------------------------
# Strict, bounded vision generation
# ---------------------------------------------------------------------------
class _InvalidVisionContent(Exception):
    pass


class _VisionModelError(Exception):
    pass


class _VisionRequestBudgetExceeded(Exception):
    pass


class _VisionDeadlineExceeded(Exception):
    pass


class _VisionGenerationFailed(Exception):
    pass


_VISION_CODE_FENCE = re.compile(
    r"\A```(?:json)?\r?\n(.*)\r?\n```\Z",
    re.DOTALL,
)


def _reject_json_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


def _vision_validation_summary(error: ValidationError) -> str:
    issues = []
    validation_errors = error.errors(include_url=False)
    for item in validation_errors[:5]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "response"
        message = str(item.get("msg", "validation failed")).replace("\n", " ")
        issues.append(f"{location}: {message}")
    remaining = len(validation_errors) - len(issues)
    if remaining > 0:
        issues.append(f"and {remaining} more validation error(s)")
    return "; ".join(issues)


def _parse_strict_json_output(content: str, model_cls, label: str):
    if not isinstance(content, str):
        raise _InvalidVisionContent(f"{label} content is not a string")

    text = content.strip()
    fenced = _VISION_CODE_FENCE.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1)

    try:
        parsed = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise _InvalidVisionContent(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise _InvalidVisionContent(f"invalid JSON: {exc}") from exc

    try:
        return model_cls.model_validate(parsed)
    except ValidationError as exc:
        raise _InvalidVisionContent(
            f"schema validation failed: {_vision_validation_summary(exc)}"
        ) from exc


def parse_vision_output(content: str) -> VisionOutput:
    return _parse_strict_json_output(content, VisionOutput, "Vision")


def parse_text_output(content: str) -> TextCardsOutput:
    return _parse_strict_json_output(content, TextCardsOutput, "Text cards")


async def _call_vision_model(payload: dict, model: str, timeout: float) -> str:
    if _http_client is None:
        raise _VisionModelError("OpenRouter client is not initialized")

    request_payload = {**payload, "model": model}
    http_timeout = httpx.Timeout(
        timeout + 1.0,
        connect=min(10.0, timeout),
        write=min(10.0, timeout),
        pool=min(5.0, timeout),
    )
    try:
        response = await asyncio.wait_for(
            _http_client.post(
                "/chat/completions",
                json=request_payload,
                timeout=http_timeout,
            ),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        raise _VisionRequestBudgetExceeded() from exc
    except httpx.HTTPError as exc:
        raise _VisionModelError(
            f"transport failed ({type(exc).__name__})"
        ) from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise _VisionModelError(f"upstream returned HTTP {response.status_code}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise _VisionModelError("Malformed vision upstream response") from exc
    if not isinstance(content, str):
        raise _VisionModelError("Vision upstream content is not a string")
    return content


async def call_validated_vision_with_fallback(payload: dict) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + VISION_TOTAL_DEADLINE_SECONDS
    last_error = None

    for attempt, model in enumerate((VISION_MODEL, VISION_FALLBACK_MODEL)):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _VisionDeadlineExceeded() from last_error
        request_budget = (
            min(VISION_PRIMARY_TIMEOUT_SECONDS, remaining)
            if attempt == 0
            else remaining
        )
        try:
            raw_content = await _call_vision_model(payload, model, request_budget)
            output = parse_vision_output(raw_content)
            return output.model_dump_json()
        except asyncio.CancelledError:
            raise
        except _InvalidVisionContent as exc:
            last_error = exc
            logger.warning("Vision model %s output rejected: %s", model, exc)
        except _VisionRequestBudgetExceeded as exc:
            logger.warning(
                "Vision model %s timed out after %.1f seconds",
                model,
                request_budget,
            )
            if attempt == 1:
                raise _VisionDeadlineExceeded() from exc
            last_error = exc
        except _VisionModelError as exc:
            last_error = exc
            logger.warning("Vision model %s request failed: %s", model, exc)
        except Exception as exc:
            last_error = exc
            logger.exception("Unexpected vision model failure model=%s", model)

        if attempt == 0:
            logger.warning(
                "Trying fallback vision model %s after %s failed",
                VISION_FALLBACK_MODEL,
                VISION_MODEL,
            )

    raise _VisionGenerationFailed() from last_error


async def call_validated_text_with_fallback(payload: dict) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TEXT_CARDS_TOTAL_DEADLINE_SECONDS
    last_error = None

    for attempt, model in enumerate((TEXT_MODEL, TEXT_FALLBACK_MODEL)):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _VisionDeadlineExceeded() from last_error
        request_budget = (
            min(TEXT_CARDS_PRIMARY_TIMEOUT_SECONDS, remaining)
            if attempt == 0
            else remaining
        )
        try:
            raw_content = await _call_vision_model(payload, model, request_budget)
            output = parse_text_output(raw_content)
            return output.model_dump_json()
        except asyncio.CancelledError:
            raise
        except _InvalidVisionContent as exc:
            last_error = exc
            logger.warning("Text cards model %s output rejected: %s", model, exc)
        except _VisionRequestBudgetExceeded as exc:
            logger.warning(
                "Text cards model %s timed out after %.1f seconds",
                model,
                request_budget,
            )
            if attempt == 1:
                raise _VisionDeadlineExceeded() from exc
            last_error = exc
        except _VisionModelError as exc:
            last_error = exc
            logger.warning("Text cards model %s request failed: %s", model, exc)
        except Exception as exc:
            last_error = exc
            logger.exception("Unexpected text cards model failure model=%s", model)

        if attempt == 0:
            logger.warning(
                "Trying fallback text cards model %s after %s failed",
                TEXT_FALLBACK_MODEL,
                TEXT_MODEL,
            )

    raise _VisionGenerationFailed() from last_error


async def consume_vision_credits(uid: str, cost: int) -> dict:
    consume_task = asyncio.create_task(consume_credits(uid, cost))
    try:
        return await asyncio.shield(consume_task)
    except asyncio.CancelledError as cancelled:
        try:
            credits = await asyncio.shield(consume_task)
        except Exception:
            raise cancelled
        if not credits.get("whitelisted", False):
            await asyncio.shield(_refund(uid, cost, "vision_cancelled"))
        raise cancelled


# ---------------------------------------------------------------------------
# Response builder — always includes credit info
# ---------------------------------------------------------------------------
def ok_response(content: str, credits: dict) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "credits": {
            "remaining": credits["remaining"],
            "total":     CREDITS_PER_PERIOD,
            "reset_at":  credits["reset_at"],
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_notification_service: Optional[NotificationService] = None


def _get_notification_service() -> NotificationService:
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


async def _run_notification_operation(uid: str, operation: str, *args):
    loop = asyncio.get_event_loop()
    service = _get_notification_service()
    try:
        return await loop.run_in_executor(
            _executor, getattr(service, operation), uid, *args
        )
    except AccountDeletionPending:
        raise HTTPException(status_code=403, detail="Account deletion in progress")
    except NotificationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except InstallationOwnershipError:
        raise HTTPException(status_code=404, detail="Installation not found")
    except NotificationPolicyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception(
            "Notification operation failed operation=%s uid=%s", operation, uid
        )
        raise HTTPException(
            status_code=503,
            detail="Notification service is temporarily unavailable",
        )

@app.get("/health")
async def health():
    """Railway health-check probe."""
    return {"status": "ok"}


@app.post("/v1/notifications/devices/register")
async def register_notification_device(
    body: DeviceRegistrationRequest,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    return await _run_notification_operation(uid, "register_device", body)


@app.post("/v1/notifications/devices/unregister")
async def unregister_notification_device(
    body: DeviceUnregistrationRequest,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    return await _run_notification_operation(
        uid, "unregister_device", body.installation_id
    )


@app.post("/v1/notifications/preferences")
async def update_notification_settings(
    body: NotificationSettingsRequest,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    return await _run_notification_operation(uid, "update_settings", body)


@app.post("/v1/notifications/state")
async def synchronize_notification_state(
    body: NotificationStateRequest,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    return await _run_notification_operation(uid, "sync_state", body)


@app.post("/v1/notifications/events:batch")
async def ingest_notification_events(
    body: NotificationEventBatchRequest,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    return await _run_notification_operation(uid, "ingest_events", body)


@app.get("/v1/credits")
async def get_credits(authorization: str = Header(None)):
    """
    Return the caller's current credit balance.
    Safe to call on app open — consumes 0 credits.
    """
    uid     = await get_uid(authorization)
    credits = await read_credits(uid)
    secs    = max(0, credits["reset_at"] - int(datetime.now(timezone.utc).timestamp()))
    return {
        "remaining": credits["remaining"],
        "total":     CREDITS_PER_PERIOD,
        "reset_at":  credits["reset_at"],
        "resets_in": f"{secs // 3600}h {(secs % 3600) // 60}m",
    }


@app.post("/v1/admob/reward-intents")
async def create_admob_reward_intent(authorization: str = Header(None)):
    uid = await get_uid(authorization)
    result = await create_reward_intent(uid)
    if not result["ok"]:
        if result.get("deleted"):
            raise HTTPException(status_code=403, detail="Account deletion in progress")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "retry_after": result["retry_after"],
            },
        )
    return {
        "intent_id": result["intent_id"],
        "expires_at": result["expires_at"],
    }


@app.post("/v1/ai-reports", status_code=201)
async def create_ai_report_endpoint(
    body: AiReportRequest,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    result = await create_ai_report(uid, body.feature, body.reason, body.output)
    if not result["ok"]:
        if result.get("deleted"):
            raise HTTPException(status_code=403, detail="Account deletion in progress")
        raise HTTPException(
            status_code=429,
            detail="Please wait before submitting another report",
            headers={"Retry-After": str(result["retry_after"])},
        )
    return {"report_id": result["report_id"]}


@app.get("/v1/admob/reward-intents/{intent_id}")
async def get_admob_reward_intent(
    intent_id: str,
    authorization: str = Header(None),
):
    uid = await get_uid(authorization)
    if not _safe_identifier.fullmatch(intent_id):
        raise HTTPException(status_code=404, detail="Reward intent not found")
    result = await read_reward_intent(uid, intent_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Reward intent not found")
    response = {"status": result.get("status", "pending")}
    if response["status"] == "credited":
        response["credits"] = {
            "remaining": int(result["remaining"]),
            "total": CREDITS_PER_PERIOD,
            "reset_at": int(result["reset_at"]),
        }
    return response


@app.post("/v1/credits/reward")
async def reward_ad_credits(
    authorization: str = Header(None),
):
    await get_uid(authorization)
    raise HTTPException(
        status_code=410,
        detail="Client reward claims are disabled. Update Reviso to use AdMob SSV.",
    )


# ---------------------------------------------------------------------------
# SSV (Server-Side Verification) callback
# ---------------------------------------------------------------------------
@app.get("/v1/admob/ssv")
async def admob_ssv_callback(request: Request):
    raw_query = request.scope.get("query_string", b"")
    try:
        signed_parameters = await verify_admob_query(raw_query)
    except SsvKeyUnavailable as exc:
        logger.exception("AdMob SSV key service unavailable")
        raise HTTPException(status_code=503, detail=str(exc))
    except InvalidSsvCallback as exc:
        logger.warning("Rejected AdMob SSV callback: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid SSV callback")

    try:
        verified = validate_ssv_parameters(signed_parameters)
    except InvalidSsvCallback as exc:
        # The callback is genuinely from Google but cannot grant a reward. This
        # includes AdMob console tests and old app versions without an intent.
        logger.warning("Ignored signed AdMob SSV callback: %s", exc)
        return {"status": "ignored"}

    try:
        result = await grant_verified_reward(
            uid=str(verified["uid"]),
            intent_id=str(verified["intent_id"]),
            transaction_id=str(verified["transaction_id"]),
            ad_unit=str(verified["ad_unit"]),
            callback_timestamp_ms=int(verified["callback_timestamp_ms"]),
        )
    except Exception:
        logger.exception("AdMob SSV credit transaction failed")
        raise HTTPException(status_code=503, detail="Reward processing unavailable")

    logger.info("AdMob SSV processed status=%s", result["status"])
    return {"status": result["status"]}


@app.post("/v1/credits/add")
async def add_credits_endpoint(
    body: AddCreditsRequest,
    x_api_key: str = Header(None),
):
    """
    Admin-only — add credits after a verified ad-reward event.

    Recommended flow:
      1. User watches ad in your app (Google AdMob / Unity Ads etc.)
      2. Ad SDK fires reward callback
      3. Your app calls a Firebase Cloud Function with the user's ID token
      4. Firebase Function verifies the token server-side, gets uid
      5. Firebase Function POSTs to this endpoint with x-api-key + {uid, amount}

    NEVER expose APP_INTERNAL_AUTH_KEY to the mobile client.
    """
    verify_admin_key(x_api_key)
    try:
        result = await add_credits(body.uid, body.amount)
    except AccountDeletionInProgress:
        raise HTTPException(status_code=409, detail="Account deletion in progress")
    logger.info("Credits added uid=%s amount=%d new_total=%d", body.uid, body.amount, result["remaining"])
    return {
        "uid":       body.uid,
        "added":     body.amount,
        "remaining": result["remaining"],
        "reset_at":  result["reset_at"],
    }


@app.post("/v1/account/delete-data")
async def delete_account_data_endpoint(authorization: str = Header(None)):
    uid = await get_recent_uid(authorization)
    try:
        return await delete_account_data(uid)
    except Exception:
        logger.exception("Backend account deletion failed uid=%s", uid)
        raise HTTPException(
            status_code=503,
            detail="Account data deletion is temporarily unavailable",
        )


@app.post("/v1/account/delete-data/begin")
async def begin_account_deletion_endpoint(authorization: str = Header(None)):
    uid = await get_recent_uid(authorization)
    try:
        return await begin_account_deletion(uid)
    except Exception:
        logger.exception("Account deletion lock failed uid=%s", uid)
        raise HTTPException(
            status_code=503,
            detail="Account deletion is temporarily unavailable",
        )


@app.get("/v1/account/deletion-status")
async def account_deletion_status_endpoint(authorization: str = Header(None)):
    decoded = await _verify_token(authorization)
    uid = decoded["uid"]
    loop = asyncio.get_event_loop()
    deleting = await loop.run_in_executor(_executor, _deletion_requested_sync, uid)
    return {"deleting": deleting}


# ---------------------------------------------------------------------------
# Jev-based answer scoring (System One decision model)
#
# Same route/credits/response contract as the LLM scorer. Jev answers nine
# parallel rubric questions plus a genuine-attempt check; weights are applied
# in code. Gemini stays as automatic fallback and as the shadow baseline.
# ---------------------------------------------------------------------------
JEV_SCORE_MODEL = os.getenv("JEV_SCORE_MODEL", "typesafe/jev-1.13")
JEV_SCORE_TIMEOUT_SECONDS = 8.0
JEV_SCORE_MIN_CONFIDENCE = 0.5
JEV_SCORE_MIN_LOW_DIMS = 2
JEV_SCORE_SHADOW_TTL_SECONDS = 30 * 24 * 60 * 60

try:
    JEV_SCORE_PERCENT = int(os.getenv("JEV_SCORE_PERCENT", "0"))
except (TypeError, ValueError):
    JEV_SCORE_PERCENT = 0
JEV_SCORE_PERCENT = max(0, min(100, JEV_SCORE_PERCENT))
JEV_SCORE_SHADOW_ENABLED = (
    os.getenv("JEV_SCORE_SHADOW_ENABLED", "false").lower() == "true"
)

JEV_SCORE_LEVELS = [
    "Wrong or missing",
    "Partial",
    "Mostly complete",
    "Fully correct, even if worded differently from the reference",
]
JEV_SCORE_GENUINE_CRITERIA = {
    "true": "Genuine attempt at answering, even if factually wrong",
    "false": "Gibberish, joke, blank, off-topic, or explicit don't-know",
}
# Same dimensions and weights as SCORE_SYSTEM_PROMPT.
JEV_SCORE_DIMS = (
    ("accuracy", 0.30),
    ("completeness", 0.15),
    ("clarity", 0.12),
    ("grammar", 0.08),
    ("style", 0.08),
    ("originality", 0.07),
    ("structure", 0.08),
    ("terminology", 0.07),
    ("evidence", 0.05),
)


def _jev_score_questions() -> dict:
    questions = {
        name: {
            "type": "score",
            "instructions": f"Rate the student answer on {name}.",
            "criteria": JEV_SCORE_LEVELS,
        }
        for name, _ in JEV_SCORE_DIMS
    }
    questions["genuine"] = {
        "type": "noul",
        "instructions": (
            "Is the student answer a genuine attempt at answering, "
            "rather than gibberish, a joke, or blank?"
        ),
        "criteria": JEV_SCORE_GENUINE_CRITERIA,
    }
    return questions


def _jev_bucket(uid: str) -> int:
    digest = sha256(f"jev-score:{uid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _jev_enabled_for(uid: str) -> bool:
    return _jev_bucket(uid) < JEV_SCORE_PERCENT


def _jev_weighted_score(levels: dict) -> int:
    total = sum(
        weight * float(levels[name]) / 3 * 100 for name, weight in JEV_SCORE_DIMS
    )
    score = round(total)
    if float(levels.get("accuracy", 3)) == 0:
        score = min(score, 25)
    return score


async def _call_jev_decisions(payload: dict, timeout: float) -> dict:
    if _http_client is None:
        raise _VisionModelError("OpenRouter client is not initialized")
    try:
        response = await asyncio.wait_for(
            _http_client.post(
                "https://openrouter.ai/api/alpha/decisions",
                json=payload,
                timeout=httpx.Timeout(
                    timeout + 1.0,
                    connect=min(5.0, timeout),
                    write=min(5.0, timeout),
                    pool=min(5.0, timeout),
                ),
            ),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        raise _VisionRequestBudgetExceeded() from exc
    except httpx.HTTPError as exc:
        raise _VisionModelError(
            f"transport failed ({type(exc).__name__})"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise _VisionModelError(f"upstream returned HTTP {response.status_code}")
    try:
        return response.json()
    except (ValueError, TypeError) as exc:
        raise _VisionModelError("Malformed decisions upstream response") from exc


async def score_with_jev(
    question: str,
    reference_answer: str,
    student_answer: str,
) -> dict | None:
    """Score via Jev; return scoring dict or None to fall back to the LLM.

    Returned dict: {levels, weighted, minconf, low_dims, genuine, model}.
    None means the caller must use the Gemini path (error, timeout, or too
    many low-confidence dimensions).
    """
    payload = {
        "model": JEV_SCORE_MODEL,
        "state": {
            "question": question,
            "reference_answer": reference_answer,
            "student_answer": student_answer,
        },
        "questions": _jev_score_questions(),
    }
    try:
        body = await _call_jev_decisions(payload, JEV_SCORE_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except (_VisionModelError, _VisionRequestBudgetExceeded):
        return None
    except Exception:
        logger.exception("Unexpected Jev scoring failure")
        return None
    try:
        answers = body.get("answers", {})
        levels = {}
        confidences = {}
        for name, _ in JEV_SCORE_DIMS:
            answer = answers.get(name, {})
            level = answer.get("score")
            if not isinstance(level, (int, float)):
                return None
            levels[name] = float(level)
            confidence = answer.get("confidence")
            confidences[name] = (
                float(confidence) if isinstance(confidence, (int, float)) else 0.0
            )
        genuine_answer = answers.get("genuine", {})
        genuine = genuine_answer.get("noul", genuine_answer.get("value"))
        genuine = float(genuine) if isinstance(genuine, (int, float)) else 0.0
        low_dims = sum(
            1 for value in confidences.values() if value < JEV_SCORE_MIN_CONFIDENCE
        )
        if low_dims >= JEV_SCORE_MIN_LOW_DIMS:
            return None
        weighted = _jev_weighted_score(levels)
        return {
            "levels": levels,
            "weighted": weighted,
            "minconf": min(confidences.values()),
            "low_dims": low_dims,
            "genuine": genuine,
            "model": body.get("model", JEV_SCORE_MODEL),
        }
    except Exception:
        logger.exception("Jev scoring response malformed")
        return None


def _jev_score_content(result: dict) -> str:
    """Same JSON contract as /v1/score minus the retired note field."""
    dims = {
        name: int(round(float(result["levels"][name]) / 3 * 100))
        for name, _ in JEV_SCORE_DIMS
    }
    dims["score"] = int(result["weighted"])
    return json.dumps(dims)


def _build_score_shadow_record(
    uid: str,
    question: str,
    reference_answer: str,
    student_answer: str,
    llm_score: int | None,
    jev: dict | None,
    served_by: str,
    now: int,
) -> dict:
    return {
        "uid": uid,
        "question": question,
        "reference_answer": reference_answer,
        "student_answer": student_answer,
        "llm_score": llm_score,
        "jev_weighted": (jev or {}).get("weighted"),
        "jev_minconf": (jev or {}).get("minconf"),
        "jev_genuine": (jev or {}).get("genuine"),
        "jev_model": (jev or {}).get("model"),
        "served_by": served_by,
        "agreed": (
            llm_score is not None
            and jev is not None
            and abs(llm_score - jev["weighted"]) <= 10
        ),
        "created_at": now,
        "delete_after_at": now + JEV_SCORE_SHADOW_TTL_SECONDS,
    }


def _build_shadow_record(
    uid: str,
    question: str,
    reference_answer: str,
    student_answer: str,
    llm_score: int | None,
    jev: dict | None,
    served_by: str,
    now: int,
) -> dict:
    return {
        "uid": uid,
        "question": question,
        "reference_answer": reference_answer,
        "student_answer": student_answer,
        "llm_score": llm_score,
        "jev_weighted": (jev or {}).get("weighted"),
        "jev_minconf": (jev or {}).get("minconf"),
        "jev_genuine": (jev or {}).get("genuine"),
        "jev_model": (jev or {}).get("model"),
        "served_by": served_by,
        "agreed": (
            llm_score is not None
            and jev is not None
            and abs(llm_score - jev["weighted"]) <= 10
        ),
        "created_at": now,
        # Firestore TTL policy on delete_after_at retains 30 days for tuning.
        "delete_after_at": now + JEV_SCORE_SHADOW_TTL_SECONDS,
    }


def _record_score_shadow_sync(record: dict) -> None:
    db = fb_fs.client()
    db.collection("ai_score_shadow").add(record)


async def _record_score_shadow(record: dict) -> None:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, _record_score_shadow_sync, record)
    except Exception:
        logger.exception("score shadow write failed")


async def _shadow_compare_and_record(
    uid: str,
    question: str,
    reference_answer: str,
    student_answer: str,
    llm_content: str | None,
    served_by: str,
) -> None:
    try:
        jev = await asyncio.wait_for(
            score_with_jev(question, reference_answer, student_answer),
            timeout=JEV_SCORE_TIMEOUT_SECONDS + 5.0,
        )
    except Exception:
        jev = None
    llm_score = None
    try:
        if llm_content is not None:
            llm_score = int(json.loads(llm_content).get("score"))
    except (ValueError, TypeError, AttributeError):
        llm_score = None
    now = int(datetime.now(timezone.utc).timestamp())
    await _record_score_shadow(
        _build_shadow_record(
            uid, question, reference_answer, student_answer,
            llm_score, jev, served_by, now,
        )
    )


def _dispatch_score_shadow(
    uid: str,
    body: ScoreRequest,
    llm_content: str | None,
    served_by: str,
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("score shadow skipped: no running loop")
        return
    loop.create_task(
        _shadow_compare_and_record(
            uid, body.question, body.actual_answer, body.user_answer,
            llm_content, served_by,
        )
    )


@app.post("/v1/score")
async def score_answer(
    body: ScoreRequest,
    authorization: str = Header(None),
):
    uid     = await get_uid(authorization)
    cost    = ENDPOINT_COSTS["score"]
    credits = await consume_credits(uid, cost)   # 402 if not enough

    if _jev_enabled_for(uid):
        try:
            jev = await asyncio.wait_for(
                score_with_jev(
                    body.question, body.actual_answer, body.user_answer
                ),
                timeout=JEV_SCORE_TIMEOUT_SECONDS + 1.0,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            jev = None
            logger.exception("score jev failed uid=%s", uid)
        if jev is not None:
            content = _jev_score_content(jev)
            logger.info(
                "score jev ok uid=%s remaining=%d", uid, credits["remaining"]
            )
            return ok_response(content, credits)
        logger.info("score jev fallback to llm uid=%s", uid)

    user_prompt = (
        f"Question: {body.question}\n\n"
        f"Reference Answer: {body.actual_answer}\n\n"
        f"Student Answer: {body.user_answer}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": SCORE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "extra_body": {"reasoning": {"enabled": True}},
    }

    try:
        content = await call_with_fallback(payload, TEXT_MODEL, TEXT_FALLBACK_MODEL)
        logger.info("score ok uid=%s remaining=%d", uid, credits["remaining"])
        if JEV_SCORE_SHADOW_ENABLED:
            _dispatch_score_shadow(uid, body, content, served_by="llm")
        return ok_response(content, credits)
    except HTTPException as e:
        if e.status_code == 502:
            await _refund(uid, cost, "openrouter_502")
        raise
    except Exception:
        logger.exception("score error uid=%s", uid)
        await _refund(uid, cost, "server_error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/vision")
async def generate_cards_from_image(
    body: VisionRequest,
    authorization: str = Header(None),
):
    uid     = await get_uid(authorization)
    cost    = ENDPOINT_COSTS["vision"]
    credits = await consume_vision_credits(uid, cost)

    data_uri = f"data:{body.mime_type};base64,{body.image_data}"

    payload = {
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text",      "text": "Generate flashcards from this image"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vision_flashcards",
                "strict": True,
                "schema": VisionOutput.model_json_schema(),
            },
        },
        "provider": {"require_parameters": True},
        "reasoning": {"enabled": False},
        "temperature": 0.1,
        "max_tokens": VISION_MAX_TOKENS,
    }

    try:
        content = await asyncio.wait_for(
            call_validated_vision_with_fallback(payload),
            timeout=VISION_TOTAL_DEADLINE_SECONDS,
        )
        logger.info("vision ok uid=%s remaining=%d", uid, credits["remaining"])
        return ok_response(content, credits)
    except asyncio.CancelledError:
        logger.warning("vision request cancelled uid=%s", uid)
        if not credits.get("whitelisted", False):
            await asyncio.shield(_refund(uid, cost, "vision_cancelled"))
        raise
    except (asyncio.TimeoutError, _VisionDeadlineExceeded):
        logger.warning("vision deadline exceeded uid=%s", uid)
        if not credits.get("whitelisted", False):
            await _refund(uid, cost, "vision_timeout")
        raise HTTPException(
            status_code=504,
            detail="Card generation took too long. Please try again.",
        )
    except _VisionGenerationFailed:
        logger.warning("vision models failed uid=%s", uid)
        if not credits.get("whitelisted", False):
            await _refund(uid, cost, "vision_502")
        raise HTTPException(
            status_code=502,
            detail="Could not generate valid cards. Please try again.",
        )
    except Exception:
        logger.exception("vision error uid=%s", uid)
        if not credits.get("whitelisted", False):
            await _refund(uid, cost, "server_error")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/text")
async def generate_cards_from_text(
    body: TextCardsRequest,
    authorization: str = Header(None),
):
    uid       = await get_uid(authorization)
    base_cost = ENDPOINT_COSTS["text"]
    credits   = await consume_vision_credits(uid, base_cost)

    payload = {
        "messages": [
            {"role": "system", "content": TEXT_CARDS_SYSTEM_PROMPT},
            {"role": "user",   "content": body.text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "text_flashcards",
                "strict": True,
                "schema": TextCardsOutput.model_json_schema(),
            },
        },
        "provider": {"require_parameters": True},
        "reasoning": {"enabled": False},
        "temperature": 0.1,
        "max_tokens": TEXT_CARDS_MAX_TOKENS,
    }

    try:
        content = await asyncio.wait_for(
            call_validated_text_with_fallback(payload),
            timeout=TEXT_CARDS_TOTAL_DEADLINE_SECONDS,
        )
    except asyncio.CancelledError:
        logger.warning("text cards request cancelled uid=%s", uid)
        if not credits.get("whitelisted", False):
            await asyncio.shield(_refund(uid, base_cost, "text_cards_cancelled"))
        raise
    except (asyncio.TimeoutError, _VisionDeadlineExceeded):
        logger.warning("text cards deadline exceeded uid=%s", uid)
        if not credits.get("whitelisted", False):
            await _refund(uid, base_cost, "text_cards_timeout")
        raise HTTPException(
            status_code=504,
            detail="Card generation took too long. Please try again.",
        )
    except _VisionGenerationFailed:
        logger.warning("text cards models failed uid=%s", uid)
        if not credits.get("whitelisted", False):
            await _refund(uid, base_cost, "text_cards_502")
        raise HTTPException(
            status_code=502,
            detail="Could not generate valid cards. Please try again.",
        )
    except Exception:
        logger.exception("text cards error uid=%s", uid)
        if not credits.get("whitelisted", False):
            await _refund(uid, base_cost, "server_error")
        raise HTTPException(status_code=500, detail="Internal server error")

    # The card count is only known after validated generation, so the
    # per-card surcharge is charged after success. On the rare race where
    # the balance dipped meanwhile, deliver anyway and log the gap.
    card_count = len(json.loads(content)["cards"])
    total_cost = text_cards_cost(card_count)
    extra_cost = total_cost - base_cost
    if extra_cost > 0:
        try:
            extra = await asyncio.shield(try_consume_credits(uid, extra_cost))
        except Exception:
            extra = None
            logger.exception("text cards extra charge failed uid=%s", uid)
        if extra is not None and extra.get("ok"):
            credits = extra
        else:
            logger.warning(
                "text_cards_undercharge uid=%s cards=%d extra=%d remaining=%s",
                uid,
                card_count,
                extra_cost,
                (extra or {}).get("remaining"),
            )
    logger.info(
        "text cards ok uid=%s cards=%d remaining=%d",
        uid,
        card_count,
        credits["remaining"],
    )
    return ok_response(content, credits)


@app.post("/v1/chat")
async def secure_chat_proxy(
    body: ChatRequest,
    authorization: str = Header(None),
):
    uid     = await get_uid(authorization)
    cost    = ENDPOINT_COSTS["chat"]
    credits = await consume_credits(uid, cost)

    messages = []
    try:
        for msg in body.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                content_list = []
                for item in msg.content:
                    if item.type == "text":
                        content_list.append({"type": "text", "text": item.text or ""})
                    elif item.type == "base64":
                        raw  = item.data or ""
                        mime = item.mime_type or "image/jpeg"
                        uri  = raw if raw.startswith("data:") else f"data:{mime};base64,{raw}"
                        content_list.append({"type": "image_url", "image_url": {"url": uri}})
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported content type: {item.type}",
                        )
                messages.append({"role": msg.role, "content": content_list})
    except HTTPException:
        raise
    except Exception:
        logger.exception("chat message prep failed uid=%s", uid)
        await _refund(uid, cost, "bad_request")
        raise HTTPException(status_code=400, detail="Invalid request format")

    payload = {
        "model": TEXT_MODEL,
        "messages": messages,
        "extra_body": {"reasoning": {"enabled": True}},
    }

    try:
        content = await call_openrouter(payload)
        logger.info("chat ok uid=%s remaining=%d", uid, credits["remaining"])
        return ok_response(content, credits)
    except HTTPException as e:
        if e.status_code == 502:
            await _refund(uid, cost, "openrouter_502")
        raise
    except Exception:
        logger.exception("chat error uid=%s", uid)
        await _refund(uid, cost, "server_error")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),  # Railway injects $PORT automatically
        reload=False,
        workers=1,
        access_log=True,
    )
