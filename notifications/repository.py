from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from firebase_admin import firestore as fb_fs

from .constants import (
    ACCOUNT_DELETIONS,
    APP_ACTIVE_GRACE_SECONDS,
    AudiencePolicy,
    DISABLED_DEVICE_RETENTION_SECONDS,
    EVENT_RETENTION_SECONDS,
    JOB_RETENTION_SECONDS,
    JobStatus,
    LifecycleSegment,
    NOTIFICATION_DEVICES,
    NOTIFICATION_EVENTS,
    NOTIFICATION_JOBS,
    NOTIFICATION_SENDS,
    NOTIFICATION_STATES,
    NOTIFICATION_WORKER_STATUS,
    INFLUENCED_WINDOW_SECONDS,
    OUTCOME_WINDOW_SECONDS,
    PermissionStatus,
    SEND_RETENTION_SECONDS,
    SendStatus,
)
from .models import (
    DeviceRecord,
    DeviceRegistrationRequest,
    JobRecord,
    NotificationEvent,
    NotificationSettingsRequest,
    NotificationState,
    NotificationStateRequest,
    SendHistory,
)
from .scheduling import (
    as_utc,
    evaluation_deadline_at,
    learned_habit_minute,
    next_evaluation_at,
    scheduling_day_key,
)

UTC = timezone.utc


class NotificationRepositoryError(Exception):
    pass


class AccountDeletionPending(NotificationRepositoryError):
    pass


class NotificationConflict(NotificationRepositoryError):
    pass


class InstallationOwnershipError(NotificationRepositoryError):
    pass


class NotificationPolicyError(NotificationRepositoryError):
    pass


def _epoch(value: datetime) -> int:
    return int(as_utc(value).timestamp())


def _within_outcome_window(send_data: dict, event_at: datetime) -> bool:
    """True when a tap-linked event falls inside the accepted-send window.

    Guards against stale completions (e.g. a tap-linked session finishing
    days after the send) being credited to that template.
    """
    accepted = send_data.get("fcm_accepted_at")
    if accepted is None:
        return False
    delta = (as_utc(event_at) - _datetime(accepted)).total_seconds()
    return 0 <= delta <= OUTCOME_WINDOW_SECONDS


def _datetime(value, default: datetime | None = None) -> datetime:
    if value is None:
        if default is None:
            raise ValueError("missing timestamp")
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime.fromtimestamp(int(value), tz=UTC)


def _date(value, default: date | None = None) -> date:
    if value is None:
        if default is None:
            raise ValueError("missing date")
        return default
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def state_payload_hash(body: NotificationStateRequest) -> str:
    return _canonical_hash(body.model_dump(mode="json"))


def event_payload_hash(
    installation_id: str,
    event: NotificationEvent,
) -> str:
    return _canonical_hash(
        {
            "installation_id": installation_id,
            "event": event.model_dump(mode="json"),
        }
    )


def _audience_from_data(
    state_data: dict, user_data: dict, as_of: date | None = None
) -> AudiencePolicy:
    raw = state_data.get("audience_policy")
    try:
        existing = AudiencePolicy(raw)
    except (TypeError, ValueError):
        existing = AudiencePolicy.UNKNOWN
    birth_date_value = user_data.get("dateOfBirth")
    if not isinstance(birth_date_value, str):
        return existing
    try:
        birth_date = date.fromisoformat(birth_date_value)
    except ValueError:
        return existing
    today = as_of or datetime.now(UTC).date()
    age = (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )
    if age < 13:
        return AudiencePolicy.UNKNOWN
    if age < 18:
        return AudiencePolicy.TEEN
    return AudiencePolicy.ADULT


def _lifecycle_segment(
    body: NotificationStateRequest,
    current_day: date,
) -> LifecycleSegment:
    last_day = body.last_qualified_study_day
    inactive = (current_day - last_day).days if last_day is not None else 10_000
    if body.total_qualified_study_days == 1 and inactive <= 1:
        return LifecycleSegment.ACTIVATED
    if body.current_streak >= 3 and body.cards_studied_today == 0 and inactive >= 1:
        return LifecycleSegment.STREAK_AT_RISK
    if 2 <= inactive <= 6:
        return LifecycleSegment.RECENTLY_LAPSED
    if 7 <= inactive <= 29:
        return LifecycleSegment.AT_RISK_MONTHLY
    if inactive >= 30:
        return LifecycleSegment.DORMANT
    if body.current_streak <= 6:
        return LifecycleSegment.HABIT_BUILDER
    return LifecycleSegment.CURRENT


def state_from_data(
    uid: str, data: dict, now: datetime | None = None
) -> NotificationState:
    fallback_now = as_utc(now or datetime.now(UTC))
    timezone_name = str(data.get("iana_timezone", "UTC"))
    fallback_day = scheduling_day_key(fallback_now, timezone_name)
    sessions = data.get("successful_study_sessions", ()) or ()
    session_minutes = tuple(
        int(item["minute"])
        for item in sessions
        if isinstance(item, dict) and "minute" in item
    )
    return NotificationState(
        uid=uid,
        state_revision=int(data.get("state_revision", 0)),
        audience_policy=_audience_from_data(data, {}),
        reminders_enabled=bool(data.get("reminders_enabled", False)),
        preferred_study_minute=int(data.get("preferred_study_minute", 19 * 60)),
        quiet_hours_start_minute=int(data.get("quiet_hours_start_minute", 8 * 60)),
        quiet_hours_end_minute=int(data.get("quiet_hours_end_minute", 22 * 60 + 30)),
        iana_timezone=timezone_name,
        scheduling_day_key=_date(data.get("scheduling_day_key"), fallback_day),
        last_study_at=(
            _datetime(data["last_study_at"])
            if data.get("last_study_at") is not None
            else None
        ),
        last_qualified_study_day=(
            _date(data["last_qualified_study_day"])
            if data.get("last_qualified_study_day") is not None
            else None
        ),
        current_streak=int(data.get("current_streak", 0)),
        daily_goal=int(data.get("daily_goal", 10)),
        cards_studied_today=int(data.get("cards_studied_today", 0)),
        actionable_queue_bucket=data.get("actionable_queue_bucket", "none"),
        oldest_overdue_bucket=data.get("oldest_overdue_bucket", "none"),
        total_qualified_study_days=int(data.get("total_qualified_study_days", 0)),
        qualified_study_days_last_7=int(data.get("qualified_study_days_last_7", 0)),
        successful_study_minutes=session_minutes,
        learned_study_minute=data.get("learned_study_minute"),
        review_event_balances={
            str(day_key): max(0, int(count))
            for day_key, count in (data.get("review_event_balances", {}) or {}).items()
            if isinstance(day_key, str) and isinstance(count, int)
        },
        consecutive_ignored_count=int(data.get("consecutive_ignored_count", 0)),
        app_active_until=(
            _datetime(data["app_active_until"])
            if data.get("app_active_until") is not None
            else None
        ),
        last_client_event_at=_datetime(data.get("last_client_event_at"), fallback_now),
        updated_at=_datetime(data.get("updated_at"), fallback_now),
    )


def device_from_data(installation_id: str, data: dict) -> DeviceRecord:
    return DeviceRecord(
        installation_id=installation_id,
        uid=str(data.get("uid", "")),
        fcm_token=data.get("fcm_token"),
        platform=str(data.get("platform", "android")),
        enabled=bool(data.get("enabled", False)),
        permission_status=data.get("permission_status", "not_determined"),
        iana_timezone=str(data.get("iana_timezone", "UTC")),
        last_seen_at=_datetime(
            data.get("last_seen_at"), datetime.fromtimestamp(0, UTC)
        ),
    )


def send_history_from_data(send_key: str, data: dict) -> SendHistory:
    return SendHistory(
        send_key=send_key,
        notification_kind=data["notification_kind"],
        copy_id=str(data["copy_id"]),
        scheduling_day_key=_date(data["scheduling_day_key"]),
        sent_at=_datetime(data.get("fcm_accepted_at") or data.get("attempted_at")),
        send_status=data.get("send_status", SendStatus.RESERVED.value),
        tone=str(data.get("tone", "neutral")),
        comeback_milestone=data.get("comeback_milestone"),
    )


def job_from_data(job_id: str, data: dict) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        uid=str(data["uid"]),
        status=data["status"],
        evaluate_at=_datetime(data["evaluate_at"]),
        deadline_at=_datetime(data["deadline_at"]),
        lease_owner=data.get("lease_owner"),
        lease_until=(
            _datetime(data["lease_until"])
            if data.get("lease_until") is not None
            else None
        ),
        attempt_count=int(data.get("attempt_count", 0)),
    )


@dataclass(frozen=True)
class WorkerContext:
    state: NotificationState
    device: DeviceRecord | None
    deletion_pending: bool
    history: tuple[SendHistory, ...]
    existing_send_keys: frozenset[str]


class NotificationRepository:
    def __init__(
        self,
        db=None,
        now_factory: Callable[[], datetime] | None = None,
    ):
        self.db = db or fb_fs.client()
        self._now_factory = now_factory or (lambda: datetime.now(UTC))

    def now(self) -> datetime:
        return as_utc(self._now_factory())

    def record_worker_poll(
        self, worker_id: str, completed_at: datetime, processed_jobs: int
    ) -> None:
        completed_at = as_utc(completed_at)
        self.db.collection(NOTIFICATION_WORKER_STATUS).document("current").set(
            {
                "worker_id": worker_id,
                "last_success_at": _epoch(completed_at),
                "processed_jobs": processed_jobs,
                "updated_at": _epoch(completed_at),
            }
        )

    @staticmethod
    def _job_id(uid: str, evaluate_at: datetime) -> str:
        return sha256(f"{uid}:{_epoch(evaluate_at)}".encode("utf-8")).hexdigest()

    def _job_fields(
        self,
        uid: str,
        state: NotificationState,
        evaluate_at: datetime,
        now: datetime,
    ) -> tuple[str, dict]:
        job_id = self._job_id(uid, evaluate_at)
        return job_id, {
            "job_id": job_id,
            "uid": uid,
            "status": JobStatus.QUEUED.value,
            "evaluate_at": _epoch(evaluate_at),
            "deadline_at": _epoch(evaluation_deadline_at(state, evaluate_at)),
            "lease_owner": None,
            "lease_until": None,
            "attempt_count": 0,
            "last_error_code": None,
            "created_at": _epoch(now),
            "updated_at": _epoch(now),
        }

    def register_device(self, uid: str, body: DeviceRegistrationRequest) -> dict:
        now = self.now()
        device_ref = self.db.collection(NOTIFICATION_DEVICES).document(
            body.installation_id
        )
        deletion_ref = self.db.collection(ACCOUNT_DELETIONS).document(uid)
        state_ref = self.db.collection(NOTIFICATION_STATES).document(uid)
        user_ref = self.db.collection("users").document(uid)

        @fb_fs.transactional
        def transaction_body(transaction):
            if deletion_ref.get(transaction=transaction).exists:
                raise AccountDeletionPending()
            device_snapshot = device_ref.get(transaction=transaction)
            state_snapshot = state_ref.get(transaction=transaction)
            user_snapshot = user_ref.get(transaction=transaction)
            existing = device_snapshot.to_dict() or {}
            state_data = state_snapshot.to_dict() or {}
            user_data = user_snapshot.to_dict() or {}
            audience = _audience_from_data(state_data, user_data, now.date())
            enabled = body.permission_status == PermissionStatus.AUTHORIZED
            offset = int(
                now.astimezone(ZoneInfo(body.iana_timezone)).utcoffset().total_seconds()
                // 60
            )
            device_data = {
                "installation_id": body.installation_id,
                "uid": uid,
                "platform": body.platform,
                "app_version": body.app_version,
                "locale": body.locale,
                "iana_timezone": body.iana_timezone,
                "utc_offset_minutes": offset,
                "enabled": enabled,
                "permission_status": body.permission_status.value,
                "last_seen_at": _epoch(now),
                "token_updated_at": None,
                "delete_after_at": (
                    None if enabled else _epoch(now) + DISABLED_DEVICE_RETENTION_SECONDS
                ),
                "created_at": existing.get("created_at", _epoch(now)),
                "updated_at": _epoch(now),
            }
            if enabled:
                device_data["fcm_token"] = body.fcm_token
                device_data["token_updated_at"] = (
                    existing.get("token_updated_at")
                    if existing.get("fcm_token") == body.fcm_token
                    and existing.get("token_updated_at") is not None
                    else _epoch(now)
                )
            transaction.set(device_ref, device_data)
            if not state_snapshot.exists:
                transaction.set(
                    state_ref,
                    {
                        "uid": uid,
                        "state_revision": 0,
                        "audience_policy": audience.value,
                        "reminders_enabled": False,
                        "preferred_study_minute": 19 * 60,
                        "quiet_hours_start_minute": 8 * 60,
                        "quiet_hours_end_minute": 22 * 60 + 30,
                        "iana_timezone": body.iana_timezone,
                        "scheduling_day_key": scheduling_day_key(
                            now, body.iana_timezone
                        ).isoformat(),
                        "actionable_queue_bucket": "none",
                        "oldest_overdue_bucket": "none",
                        "last_client_event_at": _epoch(now),
                        "updated_at": _epoch(now),
                        "created_at": _epoch(now),
                    },
                )
            return enabled

        enabled = transaction_body(self.db.transaction())
        return {
            "installation_id": body.installation_id,
            "enabled": enabled,
            "permission_status": body.permission_status.value,
        }

    def unregister_device(self, uid: str, installation_id: str) -> dict:
        now = self.now()
        ref = self.db.collection(NOTIFICATION_DEVICES).document(installation_id)
        deletion_ref = self.db.collection(ACCOUNT_DELETIONS).document(uid)

        @fb_fs.transactional
        def transaction_body(transaction):
            if deletion_ref.get(transaction=transaction).exists:
                raise AccountDeletionPending()
            snapshot = ref.get(transaction=transaction)
            data = snapshot.to_dict() or {}
            if not snapshot.exists or data.get("uid") != uid:
                return False
            transaction.update(
                ref,
                {
                    "enabled": False,
                    "fcm_token": fb_fs.DELETE_FIELD,
                    "disabled_reason": "unregistered",
                    "delete_after_at": (
                        _epoch(now) + DISABLED_DEVICE_RETENTION_SECONDS
                    ),
                    "updated_at": _epoch(now),
                },
            )
            return True

        found = transaction_body(self.db.transaction())
        return {"installation_id": installation_id, "enabled": False, "found": found}

    def update_settings(self, uid: str, body: NotificationSettingsRequest) -> dict:
        now = self.now()
        deletion_ref = self.db.collection(ACCOUNT_DELETIONS).document(uid)
        device_ref = self.db.collection(NOTIFICATION_DEVICES).document(
            body.installation_id
        )
        state_ref = self.db.collection(NOTIFICATION_STATES).document(uid)
        user_ref = self.db.collection("users").document(uid)

        @fb_fs.transactional
        def transaction_body(transaction):
            if deletion_ref.get(transaction=transaction).exists:
                raise AccountDeletionPending()
            device_snapshot = device_ref.get(transaction=transaction)
            state_snapshot = state_ref.get(transaction=transaction)
            user_snapshot = user_ref.get(transaction=transaction)
            device_data = device_snapshot.to_dict() or {}
            if not device_snapshot.exists or device_data.get("uid") != uid:
                raise InstallationOwnershipError()
            if body.reminders_enabled and (
                not device_data.get("enabled")
                or device_data.get("permission_status")
                != PermissionStatus.AUTHORIZED.value
            ):
                raise NotificationPolicyError(
                    "an authorized installation is required to enable reminders"
                )

            state_data = state_snapshot.to_dict() or {}
            audience = _audience_from_data(
                state_data, user_snapshot.to_dict() or {}, now.date()
            )
            if body.reminders_enabled and audience != AudiencePolicy.ADULT:
                if not 8 * 60 <= body.preferred_study_minute <= 21 * 60 + 30:
                    raise NotificationPolicyError(
                        "preferred study time is outside the age-safe window"
                    )

            merged = {
                **state_data,
                "uid": uid,
                "state_revision": int(state_data.get("state_revision", 0)),
                "audience_policy": audience.value,
                "reminders_enabled": body.reminders_enabled,
                "preferred_study_minute": body.preferred_study_minute,
                "quiet_hours_start_minute": body.quiet_hours_start_minute,
                "quiet_hours_end_minute": body.quiet_hours_end_minute,
                "iana_timezone": body.iana_timezone,
                "scheduling_day_key": state_data.get(
                    "scheduling_day_key",
                    scheduling_day_key(now, body.iana_timezone).isoformat(),
                ),
                "actionable_queue_bucket": state_data.get(
                    "actionable_queue_bucket", "none"
                ),
                "oldest_overdue_bucket": state_data.get(
                    "oldest_overdue_bucket", "none"
                ),
                "last_client_event_at": state_data.get(
                    "last_client_event_at", _epoch(now)
                ),
                "updated_at": state_data.get("updated_at", _epoch(now)),
                "settings_updated_at": _epoch(now),
                "created_at": state_data.get("created_at", _epoch(now)),
            }
            internal_state = state_from_data(uid, merged, now)
            evaluate_at = next_evaluation_at(internal_state, now)
            job_ref = None
            job_data = None
            job_exists = False
            if evaluate_at is not None:
                job_id, job_data = self._job_fields(
                    uid, internal_state, evaluate_at, now
                )
                job_ref = self.db.collection(NOTIFICATION_JOBS).document(job_id)
                job_exists = job_ref.get(transaction=transaction).exists

            merged["next_evaluation_at"] = (
                _epoch(evaluate_at) if evaluate_at is not None else None
            )
            transaction.set(state_ref, merged)
            transaction.update(
                device_ref,
                {
                    "iana_timezone": body.iana_timezone,
                    "last_seen_at": _epoch(now),
                    "updated_at": _epoch(now),
                },
            )
            if job_ref is not None and not job_exists:
                transaction.set(job_ref, job_data)
            return evaluate_at

        evaluate_at = transaction_body(self.db.transaction())
        return {
            "reminders_enabled": body.reminders_enabled,
            "next_evaluation_at": (
                evaluate_at.isoformat() if evaluate_at is not None else None
            ),
        }

    def sync_state(self, uid: str, body: NotificationStateRequest) -> dict:
        now = self.now()
        payload_hash = state_payload_hash(body)
        deletion_ref = self.db.collection(ACCOUNT_DELETIONS).document(uid)
        state_ref = self.db.collection(NOTIFICATION_STATES).document(uid)
        user_ref = self.db.collection("users").document(uid)

        @fb_fs.transactional
        def transaction_body(transaction):
            if deletion_ref.get(transaction=transaction).exists:
                raise AccountDeletionPending()
            snapshot = state_ref.get(transaction=transaction)
            user_snapshot = user_ref.get(transaction=transaction)
            existing = snapshot.to_dict() or {}
            stored_revision = int(existing.get("state_revision", -1))
            stored_hash = existing.get("state_payload_hash")
            if body.state_revision < stored_revision:
                raise NotificationConflict("state revision is stale")
            if body.state_revision == stored_revision and stored_hash is not None:
                if stored_hash != payload_hash:
                    raise NotificationConflict(
                        "state revision was already used with different data"
                    )
                return None, True

            timezone_name = body.iana_timezone or existing.get("iana_timezone", "UTC")
            if (
                scheduling_day_key(body.client_event_at, timezone_name)
                != body.scheduling_day_key
            ):
                raise NotificationConflict(
                    "scheduling_day_key does not match client_event_at"
                )
            effective_last_day = body.last_qualified_study_day
            if effective_last_day is None and body.last_study_at is not None:
                effective_last_day = scheduling_day_key(
                    body.last_study_at, timezone_name
                )
            effective_total_days = max(
                body.total_qualified_study_days,
                body.current_streak,
                1 if effective_last_day is not None else 0,
            )
            effective_body = body.model_copy(
                update={
                    "last_qualified_study_day": effective_last_day,
                    "total_qualified_study_days": effective_total_days,
                }
            )
            audience = _audience_from_data(
                existing, user_snapshot.to_dict() or {}, now.date()
            )
            sessions_by_day = {
                item["day"]: item
                for item in existing.get("successful_study_sessions", ()) or ()
                if isinstance(item, dict) and "day" in item and "minute" in item
            }
            if effective_last_day is not None and body.last_study_at is not None:
                local_study = as_utc(body.last_study_at).astimezone(
                    ZoneInfo(timezone_name)
                )
                sessions_by_day[effective_last_day.isoformat()] = {
                    "day": effective_last_day.isoformat(),
                    "minute": local_study.hour * 60 + local_study.minute,
                }
            sessions = sorted(sessions_by_day.values(), key=lambda item: item["day"])[
                -14:
            ]
            learned = learned_habit_minute(
                [int(item["minute"]) for item in sessions],
                int(existing.get("preferred_study_minute", 19 * 60)),
                int(existing.get("quiet_hours_start_minute", 8 * 60)),
                int(existing.get("quiet_hours_end_minute", 22 * 60 + 30)),
            )
            current_day = scheduling_day_key(now, timezone_name)
            merged = {
                **existing,
                "uid": uid,
                "state_revision": body.state_revision,
                "state_payload_hash": payload_hash,
                "audience_policy": audience.value,
                "reminders_enabled": bool(existing.get("reminders_enabled", False)),
                "preferred_study_minute": int(
                    existing.get("preferred_study_minute", 19 * 60)
                ),
                "quiet_hours_start_minute": int(
                    existing.get("quiet_hours_start_minute", 8 * 60)
                ),
                "quiet_hours_end_minute": int(
                    existing.get("quiet_hours_end_minute", 22 * 60 + 30)
                ),
                "iana_timezone": timezone_name,
                "scheduling_day_key": body.scheduling_day_key.isoformat(),
                "last_study_at": (
                    _epoch(body.last_study_at)
                    if body.last_study_at is not None
                    else None
                ),
                "last_qualified_study_day": (
                    effective_last_day.isoformat()
                    if effective_last_day is not None
                    else None
                ),
                "current_streak": body.current_streak,
                "daily_goal": body.daily_goal,
                "cards_studied_today": body.cards_studied_today,
                "actionable_queue_bucket": body.actionable_queue_bucket.value,
                "oldest_overdue_bucket": body.oldest_overdue_bucket.value,
                "total_qualified_study_days": effective_total_days,
                "qualified_study_days_last_7": body.qualified_study_days_last_7,
                "lifecycle_segment": _lifecycle_segment(
                    effective_body, current_day
                ).value,
                "successful_study_sessions": sessions,
                "learned_study_minute": learned,
                "app_active_until": (
                    _epoch(now + timedelta(seconds=APP_ACTIVE_GRACE_SECONDS))
                    if body.app_active
                    else None
                ),
                "last_client_event_at": _epoch(body.client_event_at),
                "received_at": _epoch(now),
                "updated_at": _epoch(now),
                "created_at": existing.get("created_at", _epoch(now)),
            }
            internal_state = state_from_data(uid, merged, now)
            evaluate_at = next_evaluation_at(internal_state, now)
            job_ref = None
            job_data = None
            job_exists = False
            if evaluate_at is not None:
                job_id, job_data = self._job_fields(
                    uid, internal_state, evaluate_at, now
                )
                job_ref = self.db.collection(NOTIFICATION_JOBS).document(job_id)
                job_exists = job_ref.get(transaction=transaction).exists
            merged["next_evaluation_at"] = (
                _epoch(evaluate_at) if evaluate_at is not None else None
            )
            transaction.set(state_ref, merged)
            if job_ref is not None and not job_exists:
                transaction.set(job_ref, job_data)
            return evaluate_at, False

        evaluate_at, idempotent = transaction_body(self.db.transaction())
        return {
            "state_revision": body.state_revision,
            "idempotent": idempotent,
            "next_evaluation_at": (
                evaluate_at.isoformat() if evaluate_at is not None else None
            ),
        }

    def ingest_events(
        self,
        uid: str,
        installation_id: str,
        events: Iterable[NotificationEvent],
    ) -> dict:
        now = self.now()
        events = tuple(events)
        hashes = {
            event.event_id: event_payload_hash(installation_id, event)
            for event in events
        }
        deletion_ref = self.db.collection(ACCOUNT_DELETIONS).document(uid)
        device_ref = self.db.collection(NOTIFICATION_DEVICES).document(installation_id)
        state_ref = self.db.collection(NOTIFICATION_STATES).document(uid)
        event_refs = {
            event.event_id: self.db.collection(NOTIFICATION_EVENTS).document(
                event.event_id
            )
            for event in events
        }
        notification_ids = {
            str(event.minimal_payload["notification_id"])
            for event in events
            if "notification_id" in event.minimal_payload
        }
        send_refs = {
            send_id: self.db.collection(NOTIFICATION_SENDS).document(send_id)
            for send_id in notification_ids
        }
        # Influenced fallback discovery (Duolingo-style view-through): a
        # manual study carries no notification_id, so find unresolved
        # accepted sends up front and fetch them inside the transaction
        # below. Equality-only filters need no composite index; the 2h
        # window itself is applied in code per event.
        fallback_ids: list[str] = []
        if any(
            event.event_type.value == "review_applied"
            and "notification_id" not in event.minimal_payload
            for event in events
        ):
            try:
                for snapshot in (
                    self.db.collection(NOTIFICATION_SENDS)
                    .where("uid", "==", uid)
                    .where("outcome_resolved", "==", False)
                    .limit(25)
                    .stream()
                ):
                    data = snapshot.to_dict() or {}
                    if (
                        data.get("send_status") == SendStatus.FCM_ACCEPTED.value
                        and data.get("tap_at") is None
                        and data.get("fcm_accepted_at") is not None
                    ):
                        fallback_ids.append(snapshot.id)
            except Exception:
                fallback_ids = []
        for send_id in fallback_ids:
            if send_id not in send_refs:
                send_refs[send_id] = self.db.collection(NOTIFICATION_SENDS).document(
                    send_id
                )

        @fb_fs.transactional
        def transaction_body(transaction):
            if deletion_ref.get(transaction=transaction).exists:
                raise AccountDeletionPending()
            device_snapshot = device_ref.get(transaction=transaction)
            state_snapshot = state_ref.get(transaction=transaction)
            event_snapshots = {
                event_id: ref.get(transaction=transaction)
                for event_id, ref in event_refs.items()
            }
            send_snapshots = {
                send_id: ref.get(transaction=transaction)
                for send_id, ref in send_refs.items()
            }
            device_data = device_snapshot.to_dict() or {}
            if not device_snapshot.exists or device_data.get("uid") != uid:
                raise InstallationOwnershipError()

            for event_id, snapshot in event_snapshots.items():
                if (
                    snapshot.exists
                    and (snapshot.to_dict() or {}).get("payload_hash")
                    != hashes[event_id]
                ):
                    raise NotificationConflict(
                        "event_id was already used with different data"
                    )

            accepted = 0
            duplicates = 0
            reset_ignored = False
            app_active_until = None
            influenced_done: set[str] = set()
            state_data = state_snapshot.to_dict() or {}
            timezone_name = str(state_data.get("iana_timezone", "UTC"))
            balances = {
                str(day_key): max(0, int(count))
                for day_key, count in (
                    state_data.get("review_event_balances", {}) or {}
                ).items()
                if isinstance(day_key, str) and isinstance(count, int)
            }
            review_balance_changed = False
            for event in events:
                snapshot = event_snapshots[event.event_id]
                if snapshot.exists:
                    duplicates += 1
                    continue
                transaction.set(
                    event_refs[event.event_id],
                    {
                        "event_id": event.event_id,
                        "uid": uid,
                        "installation_id": installation_id,
                        "event_type": event.event_type.value,
                        "operation_id": event.operation_id,
                        "state_revision": event.state_revision,
                        "event_at": _epoch(event.event_at),
                        "received_at": _epoch(now),
                        "delete_after_at": _epoch(now) + EVENT_RETENTION_SECONDS,
                        "payload_hash": hashes[event.event_id],
                        "minimal_payload": event.minimal_payload,
                    },
                )
                accepted += 1

                send_id = event.minimal_payload.get("notification_id")
                if send_id is not None:
                    send_id = str(send_id)
                    send_snapshot = send_snapshots.get(send_id)
                    send_data = (
                        send_snapshot.to_dict()
                        if send_snapshot is not None
                        else None
                    )
                    if send_data and send_data.get("uid") == uid:
                        field = {
                            "app_opened_from_notification": "tap_at",
                            "study_session_started": "review_started_at",
                            "study_session_completed": "review_completed_at",
                        }.get(event.event_type.value)
                        if field is not None and _within_outcome_window(
                            send_data, event.event_at
                        ):
                            updates: dict = {
                                field: _epoch(event.event_at),
                                "outcome_resolved": True,
                                "updated_at": _epoch(now),
                            }
                            if event.event_type.value == "study_session_completed":
                                # Tap path owns attribution: a tap-linked
                                # completion upgrades even an influenced send.
                                updates["outcome"] = "tapped_success"
                            transaction.update(send_refs[send_id], updates)
                elif event.event_type.value == "review_applied":
                    # Influenced fallback: manual study within 2h of an
                    # accepted send credits that template without a tap.
                    # First qualifying review wins per send per batch.
                    event_ts = _epoch(event.event_at)
                    best_id: str | None = None
                    best_accepted = -1
                    for candidate_id in fallback_ids:
                        if candidate_id in influenced_done:
                            continue
                        candidate_snapshot = send_snapshots.get(candidate_id)
                        candidate = (
                            candidate_snapshot.to_dict()
                            if candidate_snapshot is not None
                            else None
                        )
                        if not candidate or candidate.get("uid") != uid:
                            continue
                        if candidate.get("outcome_resolved"):
                            continue
                        if candidate.get("tap_at") is not None:
                            continue
                        if (
                            candidate.get("send_status")
                            != SendStatus.FCM_ACCEPTED.value
                        ):
                            continue
                        accepted_at = candidate.get("fcm_accepted_at")
                        if accepted_at is None:
                            continue
                        accepted_ts = int(accepted_at)
                        if (
                            accepted_ts
                            <= event_ts
                            <= accepted_ts + INFLUENCED_WINDOW_SECONDS
                            and accepted_ts > best_accepted
                        ):
                            best_id = candidate_id
                            best_accepted = accepted_ts
                    if best_id is not None:
                        transaction.update(
                            send_refs[best_id],
                            {
                                "influenced_studied_at": event_ts,
                                "outcome": "influenced_success",
                                "outcome_resolved": True,
                                "updated_at": _epoch(now),
                            },
                        )
                        influenced_done.add(best_id)
                if event.event_type.value == "study_session_completed":
                    reset_ignored = True
                if event.event_type.value == "app_opened_from_notification":
                    app_active_until = _epoch(
                        now + timedelta(seconds=APP_ACTIVE_GRACE_SECONDS)
                    )
                if event.event_type.value in {"review_applied", "review_undone"}:
                    day_key = str(
                        event.minimal_payload.get("scheduling_day_key")
                        or scheduling_day_key(event.event_at, timezone_name).isoformat()
                    )
                    delta = 1 if event.event_type.value == "review_applied" else -1
                    balances[day_key] = max(0, balances.get(day_key, 0) + delta)
                    review_balance_changed = True
                    if event.event_type.value == "review_applied":
                        reset_ignored = True
                        app_active_until = _epoch(
                            now + timedelta(seconds=APP_ACTIVE_GRACE_SECONDS)
                        )

            if state_snapshot.exists and (
                reset_ignored or app_active_until is not None or review_balance_changed
            ):
                state_updates = {"operational_updated_at": _epoch(now)}
                if reset_ignored:
                    state_updates["consecutive_ignored_count"] = 0
                if app_active_until is not None:
                    state_updates["app_active_until"] = app_active_until
                if review_balance_changed:
                    current_day = scheduling_day_key(now, timezone_name)
                    retained_balances = {}
                    for day_key, count in balances.items():
                        try:
                            balance_day = date.fromisoformat(day_key)
                        except ValueError:
                            continue
                        if count > 0 and balance_day >= current_day - timedelta(
                            days=30
                        ):
                            retained_balances[day_key] = count
                    state_updates["review_event_balances"] = retained_balances
                transaction.update(state_ref, state_updates)
            transaction.update(
                device_ref,
                {"last_seen_at": _epoch(now), "updated_at": _epoch(now)},
            )
            return accepted, duplicates

        accepted, duplicates = transaction_body(self.db.transaction())
        return {"accepted": accepted, "duplicates": duplicates}

    def scan_due_states(self, now: datetime, limit: int = 100) -> int:
        now = as_utc(now)
        snapshots = list(
            self.db.collection(NOTIFICATION_STATES)
            .where("next_evaluation_at", "<=", _epoch(now))
            .limit(limit)
            .stream()
        )
        created = 0
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            if not data.get("reminders_enabled"):
                continue
            uid = snapshot.id
            state = state_from_data(uid, data, now)
            evaluate_at = _datetime(data.get("next_evaluation_at"), now)
            job_id, job_data = self._job_fields(uid, state, evaluate_at, now)
            job_ref = self.db.collection(NOTIFICATION_JOBS).document(job_id)

            @fb_fs.transactional
            def create_if_missing(transaction):
                current_state = (
                    self.db.collection(NOTIFICATION_STATES)
                    .document(uid)
                    .get(transaction=transaction)
                )
                job_snapshot = job_ref.get(transaction=transaction)
                if (
                    not current_state.exists
                    or not (current_state.to_dict() or {}).get("reminders_enabled")
                    or job_snapshot.exists
                ):
                    return False
                transaction.set(job_ref, job_data)
                return True

            if create_if_missing(self.db.transaction()):
                created += 1
        return created

    def claim_due_jobs(
        self,
        worker_id: str,
        now: datetime,
        batch_size: int,
        lease_seconds: int,
    ) -> tuple[JobRecord, ...]:
        now = as_utc(now)
        collection = self.db.collection(NOTIFICATION_JOBS)
        query_limit = max(batch_size * 2, batch_size)
        queued = list(
            collection.where("status", "==", JobStatus.QUEUED.value)
            .where("evaluate_at", "<=", _epoch(now))
            .limit(query_limit)
            .stream()
        )
        expired_leases = list(
            collection.where("status", "==", JobStatus.PROCESSING.value)
            .where("lease_until", "<=", _epoch(now))
            .limit(query_limit)
            .stream()
        )
        snapshots_by_id = {
            snapshot.id: snapshot for snapshot in (*queued, *expired_leases)
        }
        snapshots = sorted(
            snapshots_by_id.values(),
            key=lambda snapshot: int((snapshot.to_dict() or {}).get("evaluate_at", 0)),
        )
        claimed = []
        for snapshot in snapshots:
            if len(claimed) >= batch_size:
                break
            ref = snapshot.reference

            @fb_fs.transactional
            def claim(transaction):
                current = ref.get(transaction=transaction)
                if not current.exists:
                    return None
                data = current.to_dict() or {}
                status = data.get("status")
                lease_until = int(data.get("lease_until") or 0)
                claimable = status == JobStatus.QUEUED.value or (
                    status == JobStatus.PROCESSING.value and lease_until <= _epoch(now)
                )
                if not claimable or int(data.get("evaluate_at", 0)) > _epoch(now):
                    return None
                transaction.update(
                    ref,
                    {
                        "status": JobStatus.PROCESSING.value,
                        "lease_owner": worker_id,
                        "lease_until": _epoch(now) + lease_seconds,
                        "attempt_count": int(data.get("attempt_count", 0)) + 1,
                        "updated_at": _epoch(now),
                    },
                )
                data.update(
                    {
                        "status": JobStatus.PROCESSING.value,
                        "lease_owner": worker_id,
                        "lease_until": _epoch(now) + lease_seconds,
                        "attempt_count": int(data.get("attempt_count", 0)) + 1,
                    }
                )
                return job_from_data(current.id, data)

            job = claim(self.db.transaction())
            if job is not None:
                claimed.append(job)
        return tuple(claimed)

    def worker_context(self, uid: str, now: datetime) -> WorkerContext | None:
        now = as_utc(now)
        state_snapshot = self.db.collection(NOTIFICATION_STATES).document(uid).get()
        if not state_snapshot.exists:
            return None
        state = state_from_data(uid, state_snapshot.to_dict() or {}, now)
        deleting = self.db.collection(ACCOUNT_DELETIONS).document(uid).get().exists
        device_snapshots = list(
            self.db.collection(NOTIFICATION_DEVICES)
            .where("uid", "==", uid)
            .limit(100)
            .stream()
        )
        devices = [
            device_from_data(snapshot.id, snapshot.to_dict() or {})
            for snapshot in device_snapshots
        ]
        eligible_devices = [
            device
            for device in devices
            if device.enabled
            and device.fcm_token
            and device.permission_status == PermissionStatus.AUTHORIZED
        ]
        device = max(eligible_devices, key=lambda item: item.last_seen_at, default=None)

        cutoff = now - timedelta(days=30)
        send_snapshots = list(
            self.db.collection(NOTIFICATION_SENDS)
            .where("uid", "==", uid)
            .where("attempted_at", ">=", _epoch(cutoff))
            .limit(200)
            .stream()
        )
        history = []
        existing = set()
        for snapshot in send_snapshots:
            data = snapshot.to_dict() or {}
            try:
                item = send_history_from_data(snapshot.id, data)
            except (KeyError, TypeError, ValueError):
                continue
            if item.sent_at >= cutoff:
                history.append(item)
            if item.send_status == SendStatus.FCM_ACCEPTED:
                existing.add(item.send_key)
        history.sort(key=lambda item: item.sent_at)
        return WorkerContext(
            state=state,
            device=device,
            deletion_pending=deleting,
            history=tuple(history),
            existing_send_keys=frozenset(existing),
        )

    def reserve_send(
        self,
        job: JobRecord,
        worker_id: str,
        state: NotificationState,
        device: DeviceRecord,
        send_data: dict,
    ) -> str:
        now = self.now()
        job_ref = self.db.collection(NOTIFICATION_JOBS).document(job.job_id)
        deletion_ref = self.db.collection(ACCOUNT_DELETIONS).document(job.uid)
        state_ref = self.db.collection(NOTIFICATION_STATES).document(job.uid)
        device_ref = self.db.collection(NOTIFICATION_DEVICES).document(
            device.installation_id
        )
        send_ref = self.db.collection(NOTIFICATION_SENDS).document(
            send_data["send_key"]
        )

        @fb_fs.transactional
        def transaction_body(transaction):
            job_snapshot = job_ref.get(transaction=transaction)
            deletion_snapshot = deletion_ref.get(transaction=transaction)
            state_snapshot = state_ref.get(transaction=transaction)
            device_snapshot = device_ref.get(transaction=transaction)
            send_snapshot = send_ref.get(transaction=transaction)
            job_data = job_snapshot.to_dict() or {}
            current_state = state_snapshot.to_dict() or {}
            current_device = device_snapshot.to_dict() or {}
            if deletion_snapshot.exists:
                return "deleted"
            if (
                not job_snapshot.exists
                or job_data.get("status") != JobStatus.PROCESSING.value
                or job_data.get("lease_owner") != worker_id
            ):
                return "lease_lost"
            if (
                not state_snapshot.exists
                or int(current_state.get("state_revision", -1)) != state.state_revision
                or int(current_state.get("updated_at", -1)) != _epoch(state.updated_at)
                or not current_state.get("reminders_enabled")
            ):
                return "state_changed"
            if (
                not device_snapshot.exists
                or not current_device.get("enabled")
                or current_device.get("uid") != job.uid
                or current_device.get("fcm_token") != device.fcm_token
            ):
                return "device_changed"
            if send_snapshot.exists:
                existing = send_snapshot.to_dict() or {}
                if existing.get("send_status") == SendStatus.FCM_ACCEPTED.value:
                    return "duplicate"
                if existing.get("job_id") != job.job_id:
                    return "duplicate"
                transaction.update(
                    send_ref,
                    {
                        "send_status": SendStatus.RESERVED.value,
                        "attempted_at": _epoch(now),
                        "updated_at": _epoch(now),
                    },
                )
                return "reserved"

            transaction.set(
                send_ref,
                {
                    **send_data,
                    "send_id": send_data["send_key"],
                    "job_id": job.job_id,
                    "uid": job.uid,
                    "installation_id": device.installation_id,
                    "scheduled_at": _epoch(job.evaluate_at),
                    "attempted_at": _epoch(now),
                    "fcm_accepted_at": None,
                    "send_status": SendStatus.RESERVED.value,
                    "error_code": None,
                    "tap_at": None,
                    "review_started_at": None,
                    "review_completed_at": None,
                    "influenced_studied_at": None,
                    "outcome": None,
                    "outcome_resolved": False,
                    "created_at": _epoch(now),
                    "delete_after_at": _epoch(now) + SEND_RETENTION_SECONDS,
                    "updated_at": _epoch(now),
                },
            )
            return "reserved"

        return transaction_body(self.db.transaction())

    def can_send_now(
        self,
        job: JobRecord,
        worker_id: str,
        state: NotificationState,
        device: DeviceRecord,
    ) -> bool:
        now = self.now()
        if self.db.collection(ACCOUNT_DELETIONS).document(job.uid).get().exists:
            return False
        job_snapshot = self.db.collection(NOTIFICATION_JOBS).document(job.job_id).get()
        state_snapshot = self.db.collection(NOTIFICATION_STATES).document(job.uid).get()
        device_snapshot = (
            self.db.collection(NOTIFICATION_DEVICES)
            .document(device.installation_id)
            .get()
        )
        current_job = job_snapshot.to_dict() or {}
        current_state = state_snapshot.to_dict() or {}
        current_device = device_snapshot.to_dict() or {}
        return bool(
            job_snapshot.exists
            and current_job.get("status") == JobStatus.PROCESSING.value
            and current_job.get("lease_owner") == worker_id
            and int(current_job.get("lease_until") or 0) > _epoch(now)
            and int(current_job.get("deadline_at") or 0) >= _epoch(now)
            and state_snapshot.exists
            and device_snapshot.exists
            and current_state.get("reminders_enabled")
            and int(current_state.get("state_revision", -1)) == state.state_revision
            and int(current_state.get("updated_at", -1)) == _epoch(state.updated_at)
            and current_device.get("enabled")
            and current_device.get("uid") == job.uid
            and current_device.get("fcm_token") == device.fcm_token
        )

    def mark_send_accepted(self, send_key: str, accepted_at: datetime) -> None:
        accepted_at = as_utc(accepted_at)
        self.db.collection(NOTIFICATION_SENDS).document(send_key).update(
            {
                "send_status": SendStatus.FCM_ACCEPTED.value,
                "fcm_accepted_at": _epoch(accepted_at),
                "ignore_after_at": _epoch(accepted_at) + OUTCOME_WINDOW_SECONDS,
                "error_code": None,
                "updated_at": _epoch(accepted_at),
            }
        )

    def mark_send_retrying(self, send_key: str, error_code: str) -> None:
        now = self.now()
        self.db.collection(NOTIFICATION_SENDS).document(send_key).update(
            {
                "send_status": SendStatus.RETRYING.value,
                "error_code": error_code,
                "updated_at": _epoch(now),
            }
        )

    def mark_send_permanent_failure(self, send_key: str, error_code: str) -> None:
        now = self.now()
        self.db.collection(NOTIFICATION_SENDS).document(send_key).update(
            {
                "send_status": SendStatus.FAILED_PERMANENT.value,
                "error_code": error_code,
                "outcome_resolved": True,
                "updated_at": _epoch(now),
            }
        )

    def disable_invalid_device(
        self,
        uid: str,
        device: DeviceRecord,
        error_code: str,
    ) -> None:
        now = self.now()
        ref = self.db.collection(NOTIFICATION_DEVICES).document(device.installation_id)

        @fb_fs.transactional
        def transaction_body(transaction):
            snapshot = ref.get(transaction=transaction)
            data = snapshot.to_dict() or {}
            if (
                snapshot.exists
                and data.get("uid") == uid
                and data.get("fcm_token") == device.fcm_token
            ):
                transaction.update(
                    ref,
                    {
                        "enabled": False,
                        "fcm_token": fb_fs.DELETE_FIELD,
                        "disabled_reason": error_code,
                        "delete_after_at": (
                            _epoch(now) + DISABLED_DEVICE_RETENTION_SECONDS
                        ),
                        "updated_at": _epoch(now),
                    },
                )

        transaction_body(self.db.transaction())

    def finish_job(
        self,
        job: JobRecord,
        worker_id: str,
        status: JobStatus,
        reason: str | None,
        state: NotificationState | None,
        history: Iterable[SendHistory] = (),
    ) -> bool:
        now = self.now()
        next_at = next_evaluation_at(state, now, history) if state is not None else None
        job_ref = self.db.collection(NOTIFICATION_JOBS).document(job.job_id)
        state_ref = self.db.collection(NOTIFICATION_STATES).document(job.uid)
        next_job_ref = None
        next_job_data = None
        if state is not None and next_at is not None:
            next_job_id, next_job_data = self._job_fields(job.uid, state, next_at, now)
            next_job_ref = self.db.collection(NOTIFICATION_JOBS).document(next_job_id)

        @fb_fs.transactional
        def transaction_body(transaction):
            job_snapshot = job_ref.get(transaction=transaction)
            state_snapshot = state_ref.get(transaction=transaction)
            next_job_snapshot = (
                next_job_ref.get(transaction=transaction)
                if next_job_ref is not None
                else None
            )
            data = job_snapshot.to_dict() or {}
            if (
                not job_snapshot.exists
                or data.get("status") != JobStatus.PROCESSING.value
                or data.get("lease_owner") != worker_id
            ):
                return False
            current_state = state_snapshot.to_dict() or {}
            state_unchanged = (
                state is not None
                and state_snapshot.exists
                and (
                    int(current_state.get("state_revision", -1)) == state.state_revision
                    and int(current_state.get("updated_at", -1))
                    == _epoch(state.updated_at)
                )
            )
            transaction.update(
                job_ref,
                {
                    "status": status.value,
                    "lease_owner": None,
                    "lease_until": None,
                    "last_error_code": reason,
                    "delete_after_at": _epoch(now) + JOB_RETENTION_SECONDS,
                    "updated_at": _epoch(now),
                },
            )
            if state_unchanged:
                transaction.update(
                    state_ref,
                    {
                        "next_evaluation_at": (
                            _epoch(next_at) if next_at is not None else None
                        ),
                        "operational_updated_at": _epoch(now),
                    },
                )
            if (
                next_job_ref is not None
                and next_job_snapshot is not None
                and not next_job_snapshot.exists
                and state_unchanged
            ):
                transaction.set(next_job_ref, next_job_data)
            return True

        return transaction_body(self.db.transaction())

    def purge_expired_records(
        self, now: datetime, limit_per_collection: int = 100
    ) -> int:
        now = as_utc(now)
        cutoff = _epoch(now)
        deleted = 0
        for collection_name in (
            NOTIFICATION_DEVICES,
            NOTIFICATION_EVENTS,
            NOTIFICATION_JOBS,
            NOTIFICATION_SENDS,
        ):
            snapshots = list(
                self.db.collection(collection_name)
                .where("delete_after_at", "<=", cutoff)
                .limit(limit_per_collection)
                .stream()
            )
            for snapshot in snapshots:
                ref = snapshot.reference

                @fb_fs.transactional
                def delete_if_still_expired(transaction):
                    current = ref.get(transaction=transaction)
                    data = current.to_dict() or {}
                    if (
                        not current.exists
                        or int(data.get("delete_after_at") or 0) > cutoff
                    ):
                        return False
                    transaction.delete(ref)
                    return True

                if delete_if_still_expired(self.db.transaction()):
                    deleted += 1
        return deleted

    def retry_job(
        self,
        job: JobRecord,
        worker_id: str,
        error_code: str,
        retry_at: datetime,
    ) -> bool:
        now = self.now()
        ref = self.db.collection(NOTIFICATION_JOBS).document(job.job_id)

        @fb_fs.transactional
        def transaction_body(transaction):
            snapshot = ref.get(transaction=transaction)
            data = snapshot.to_dict() or {}
            if (
                not snapshot.exists
                or data.get("status") != JobStatus.PROCESSING.value
                or data.get("lease_owner") != worker_id
            ):
                return False
            transaction.update(
                ref,
                {
                    "status": JobStatus.QUEUED.value,
                    "evaluate_at": _epoch(retry_at),
                    "lease_owner": None,
                    "lease_until": None,
                    "last_error_code": error_code,
                    "updated_at": _epoch(now),
                },
            )
            return True

        return transaction_body(self.db.transaction())

    def reconcile_ignored_sends(self, now: datetime, limit: int = 100) -> int:
        now = as_utc(now)
        snapshots = list(
            self.db.collection(NOTIFICATION_SENDS)
            .where("outcome_resolved", "==", False)
            .where("ignore_after_at", "<=", _epoch(now))
            .limit(limit)
            .stream()
        )
        reconciled = 0
        for snapshot in snapshots:
            send_ref = snapshot.reference

            @fb_fs.transactional
            def reconcile(transaction):
                send_snapshot = send_ref.get(transaction=transaction)
                send_data = send_snapshot.to_dict() or {}
                if (
                    not send_snapshot.exists
                    or send_data.get("send_status") != SendStatus.FCM_ACCEPTED.value
                    or send_data.get("outcome_resolved")
                    or int(send_data.get("ignore_after_at", 0)) > _epoch(now)
                ):
                    return False
                uid = send_data.get("uid")
                state_ref = self.db.collection(NOTIFICATION_STATES).document(uid)
                state_snapshot = state_ref.get(transaction=transaction)
                transaction.update(
                    send_ref,
                    {
                        "outcome_resolved": True,
                        "outcome": "ignored",
                        "updated_at": _epoch(now),
                    },
                )
                if state_snapshot.exists:
                    state_data = state_snapshot.to_dict() or {}
                    transaction.update(
                        state_ref,
                        {
                            "consecutive_ignored_count": int(
                                state_data.get("consecutive_ignored_count", 0)
                            )
                            + 1,
                            "operational_updated_at": _epoch(now),
                        },
                    )
                return True

            if reconcile(self.db.transaction()):
                reconciled += 1
        return reconciled
