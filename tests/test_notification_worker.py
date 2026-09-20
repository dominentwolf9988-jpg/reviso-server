from datetime import date, datetime, timedelta, timezone

import pytest
from firebase_admin import messaging

from notifications.constants import (
    AudiencePolicy,
    JobStatus,
    NotificationKind,
    SuppressionReason,
)
from notifications.models import (
    DeviceRecord,
    JobRecord,
    NotificationState,
)
from notifications.repository import WorkerContext
from notifications.sender import (
    FCMSender,
    OutboundNotification,
    PermanentTokenError,
    TransientSendError,
)
from notifications.worker import NotificationWorker

UTC = timezone.utc
NOW = datetime(2026, 8, 31, 19, 0, tzinfo=UTC)


def _state(**updates):
    values = {
        "uid": "firebase-user-123",
        "state_revision": 5,
        "audience_policy": AudiencePolicy.TEEN,
        "reminders_enabled": True,
        "preferred_study_minute": 19 * 60,
        "quiet_hours_start_minute": 8 * 60,
        "quiet_hours_end_minute": 21 * 60 + 30,
        "iana_timezone": "UTC",
        "scheduling_day_key": date(2026, 8, 31),
        "last_study_at": NOW - timedelta(days=1),
        "last_qualified_study_day": date(2026, 8, 30),
        "current_streak": 2,
        "daily_goal": 10,
        "cards_studied_today": 0,
        "actionable_queue_bucket": "6_10",
        "oldest_overdue_bucket": "1_3_days",
        "total_qualified_study_days": 8,
        "qualified_study_days_last_7": 5,
        "last_client_event_at": NOW - timedelta(minutes=2),
        "updated_at": NOW - timedelta(minutes=2),
    }
    values.update(updates)
    return NotificationState(**values)


def _device(**updates):
    values = {
        "installation_id": "installation-123",
        "uid": "firebase-user-123",
        "fcm_token": "fcm-token-value-with-enough-length",
        "enabled": True,
        "permission_status": "authorized",
        "iana_timezone": "UTC",
        "last_seen_at": NOW - timedelta(minutes=1),
    }
    values.update(updates)
    return DeviceRecord(**values)


def _job(**updates):
    values = {
        "job_id": "job-identifier-123",
        "uid": "firebase-user-123",
        "status": JobStatus.PROCESSING,
        "evaluate_at": NOW,
        "deadline_at": NOW + timedelta(hours=2),
        "lease_owner": "worker-1",
        "lease_until": NOW + timedelta(minutes=2),
        "attempt_count": 1,
    }
    values.update(updates)
    return JobRecord(**values)


class _FakeRepository:
    def __init__(self, context=None):
        self.context = context or WorkerContext(
            state=_state(),
            device=_device(),
            deletion_pending=False,
            history=(),
            existing_send_keys=frozenset(),
        )
        self.job = _job()
        self.reservation = "reserved"
        self.send_guard = True
        self.reserved = []
        self.accepted = []
        self.permanent = []
        self.retrying = []
        self.disabled = []
        self.finished = []
        self.retried = []
        self.polls = []

    def now(self):
        return NOW

    def reconcile_ignored_sends(self, now):
        return 0

    def purge_expired_records(self, now):
        return 0

    def record_worker_poll(self, worker_id, completed_at, processed_jobs):
        self.polls.append((worker_id, completed_at, processed_jobs))

    def scan_due_states(self, now):
        return 0

    def claim_due_jobs(self, worker_id, now, batch_size, lease_seconds):
        return (self.job,)

    def worker_context(self, uid, now):
        return self.context

    def reserve_send(self, job, worker_id, state, device, send_data):
        self.reserved.append(send_data)
        return self.reservation

    def can_send_now(self, job, worker_id, state, device):
        return self.send_guard

    def mark_send_accepted(self, send_key, accepted_at):
        self.accepted.append(send_key)

    def mark_send_permanent_failure(self, send_key, error_code):
        self.permanent.append((send_key, error_code))

    def mark_send_retrying(self, send_key, error_code):
        self.retrying.append((send_key, error_code))

    def disable_invalid_device(self, uid, device, error_code):
        self.disabled.append((uid, device.installation_id, error_code))

    def finish_job(self, *args):
        self.finished.append(args)
        return True

    def retry_job(self, *args):
        self.retried.append(args)
        return True


class _FakeSender:
    def __init__(self, error=None):
        self.error = error
        self.outbound = []

    def send(self, outbound):
        self.outbound.append(outbound)
        if self.error:
            raise self.error
        return "projects/test/messages/123"


def _worker(repository, sender):
    return NotificationWorker(repository, sender, "worker-1", experiment_salt="test")


def test_worker_reserves_sends_and_completes_in_two_phases():
    repository = _FakeRepository()
    sender = _FakeSender()

    assert _worker(repository, sender).run_once() == 1

    assert len(repository.reserved) == 1
    assert len(sender.outbound) == 1
    assert repository.accepted == [repository.reserved[0]["send_key"]]
    assert repository.finished[-1][2] == JobStatus.COMPLETED
    assert sender.outbound[0].notification_kind == NotificationKind.HABIT_CUE
    assert repository.polls == [("worker-1", NOW, 1)]


def test_worker_rechecks_deletion_and_state_immediately_before_send():
    deleting = _FakeRepository()
    deleting.context = WorkerContext(
        state=_state(),
        device=_device(),
        deletion_pending=True,
        history=(),
        existing_send_keys=frozenset(),
    )
    sender = _FakeSender()
    _worker(deleting, sender).run_once()
    assert sender.outbound == []
    assert deleting.finished[-1][3] == SuppressionReason.ACCOUNT_DELETION_PENDING.value

    changed = _FakeRepository()
    changed.send_guard = False
    _worker(changed, sender).run_once()
    assert changed.permanent[-1][1] == "pre_send_guard"
    assert len(sender.outbound) == 0


def test_permanent_token_error_disables_and_cleans_up_installation():
    repository = _FakeRepository()
    sender = _FakeSender(PermanentTokenError("unregistered"))

    _worker(repository, sender).run_once()

    assert repository.disabled == [
        ("firebase-user-123", "installation-123", "unregistered")
    ]
    assert repository.permanent[-1][1] == "unregistered"
    assert repository.finished[-1][2] == JobStatus.FAILED


def test_transient_send_error_requeues_with_bounded_backoff():
    repository = _FakeRepository()
    sender = _FakeSender(TransientSendError("unavailable"))

    _worker(repository, sender).run_once()

    assert repository.retrying[-1][1] == "unavailable"
    assert len(repository.retried) == 1
    retry_at = repository.retried[0][-1]
    assert NOW < retry_at < repository.job.deadline_at


def test_worker_suppresses_if_state_changed_to_studied_before_claim_processing():
    state = _state(
        cards_studied_today=1,
        last_qualified_study_day=date(2026, 8, 31),
    )
    repository = _FakeRepository(
        WorkerContext(
            state=state,
            device=_device(),
            deletion_pending=False,
            history=(),
            existing_send_keys=frozenset(),
        )
    )
    sender = _FakeSender()

    _worker(repository, sender).run_once()

    assert repository.reserved == []
    assert sender.outbound == []
    assert repository.finished[-1][3] == SuppressionReason.ALREADY_STUDIED.value


def test_fcm_sender_uses_allowlisted_global_review_data_without_card_content():
    captured = []

    def send_function(message, app=None):
        captured.append(message)
        return "projects/test/messages/123"

    outbound = OutboundNotification(
        token="secret-device-token",
        notification_id="a" * 64,
        notification_kind=NotificationKind.HABIT_CUE,
        scheduling_day_key="2026-08-31",
        copy_id="habit_safe_1",
        experiment_id="none",
        variant_id="fixed",
        title="Your review is ready",
        body="A short review is ready.",
        sent_at=NOW,
        deadline_at=NOW + timedelta(hours=1),
    )

    result = FCMSender(send_function=send_function).send(outbound)

    assert result == "projects/test/messages/123"
    message = captured[0]
    assert message.data["route"] == "global_review"
    assert set(message.data) == {
        "schema_version",
        "notification_id",
        "notification_kind",
        "route",
        "scheduling_day_key",
        "copy_id",
        "experiment_id",
        "variant_id",
        "sent_at",
    }
    assert "secret-device-token" not in message.data.values()
    assert not {"card_id", "deck_id", "card_front", "card_back"} & set(message.data)
    assert message.android.collapse_key == "habit_cue:2026-08-31"
    # Registration tokens must go in `token`, not `fid`.
    # `fid` expects a Firebase Installation ID and rejects tokens
    # with UNREGISTERED. See
    # https://firebase.google.com/docs/cloud-messaging/send/admin-sdk
    assert message.token == "secret-device-token"
    assert message.fid is None


def test_fcm_unregistered_error_is_classified_as_permanent():
    def send_function(message, app=None):
        raise messaging.UnregisteredError("UNREGISTERED")

    outbound = OutboundNotification(
        token="secret-device-token",
        notification_id="a" * 64,
        notification_kind=NotificationKind.HABIT_CUE,
        scheduling_day_key="2026-08-31",
        copy_id="habit_safe_1",
        experiment_id="none",
        variant_id="fixed",
        title="Your review is ready",
        body="A short review is ready.",
        sent_at=NOW,
        deadline_at=NOW + timedelta(hours=1),
    )

    with pytest.raises(PermanentTokenError) as error:
        FCMSender(send_function=send_function).send(outbound)
    assert error.value.code == "unregistered"
