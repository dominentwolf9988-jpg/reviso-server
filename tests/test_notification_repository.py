from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from firebase_admin import firestore as fb_fs

from notifications.constants import JobStatus
from notifications.models import (
    DeviceRegistrationRequest,
    NotificationEvent,
    NotificationStateRequest,
)
from notifications.repository import (
    NotificationConflict,
    NotificationRepository,
    device_from_data,
    job_from_data,
    state_from_data,
)

UTC = timezone.utc
NOW = datetime(2026, 8, 31, 14, 30, tzinfo=UTC)


class _Snapshot:
    def __init__(self, reference):
        self.reference = reference
        self.id = reference.id
        self.exists = reference.id in reference.collection.documents

    def to_dict(self):
        if not self.exists:
            return None
        return dict(self.reference.collection.documents[self.id])


class _DocumentReference:
    def __init__(self, collection, document_id):
        self.collection = collection
        self.id = document_id

    def get(self, transaction=None):
        return _Snapshot(self)

    def set(self, data, merge=False):
        if merge:
            self.collection.documents.setdefault(self.id, {}).update(dict(data))
        else:
            self.collection.documents[self.id] = dict(data)

    def update(self, data):
        target = self.collection.documents[self.id]
        for key, value in data.items():
            if value is fb_fs.DELETE_FIELD:
                target.pop(key, None)
            else:
                target[key] = value

    def delete(self):
        self.collection.documents.pop(self.id, None)


class _Query:
    def __init__(self, collection, filters=()):
        self.collection = collection
        self.filters = filters
        self.result_limit = None

    def where(self, field, operator, value):
        return _Query(self.collection, (*self.filters, (field, operator, value)))

    def limit(self, value):
        self.result_limit = value
        return self

    def stream(self):
        snapshots = []
        for document_id, data in self.collection.documents.items():
            matches = True
            for field, operator, value in self.filters:
                actual = data.get(field)
                if operator == "==":
                    matches = matches and actual == value
                elif operator == "<=":
                    matches = matches and actual is not None and actual <= value
                elif operator == ">=":
                    matches = matches and actual is not None and actual >= value
                else:
                    raise AssertionError(f"unsupported query operator {operator}")
            if matches:
                snapshots.append(
                    _Snapshot(_DocumentReference(self.collection, document_id))
                )
        if self.result_limit is not None:
            snapshots = snapshots[: self.result_limit]
        return snapshots


class _Collection:
    def __init__(self, documents=None):
        self.documents = documents or {}

    def document(self, document_id):
        return _DocumentReference(self, document_id)

    def where(self, field, operator, value):
        return _Query(self, ((field, operator, value),))


class _Transaction:
    def __init__(self):
        self._read_only = False
        self._max_attempts = 1
        self._id = None

    def _clean_up(self):
        self._id = None

    def _begin(self, retry_id=None):
        self._id = uuid4().bytes

    def _commit(self):
        return None

    def _rollback(self):
        return None

    def set(self, reference, data, merge=False):
        reference.set(data, merge=merge)

    def update(self, reference, data):
        reference.update(data)

    def delete(self, reference):
        reference.delete()


class _Firestore:
    def __init__(self, collections=None):
        self.collections = collections or {}

    def collection(self, name):
        return self.collections.setdefault(name, _Collection())

    def transaction(self):
        return _Transaction()


def _repository(collections=None, now=NOW):
    firestore = _Firestore(collections)
    return NotificationRepository(firestore, now_factory=lambda: now), firestore


def _registration(token="fcm-token-value-with-enough-length"):
    return DeviceRegistrationRequest(
        installation_id="installation-123",
        fcm_token=token,
        platform="android",
        app_version="1.2.3",
        locale="en-IN",
        iana_timezone="Asia/Kolkata",
        permission_status="authorized",
    )


def _state(revision=1, cards=0):
    return NotificationStateRequest(
        state_revision=revision,
        scheduling_day_key="2026-08-31",
        last_study_at="2026-08-30T14:00:00Z" if cards == 0 else "2026-08-31T14:00:00Z",
        current_streak=2,
        daily_goal=10,
        cards_studied_today=cards,
        actionable_queue_bucket="6_10",
        oldest_overdue_bucket="1_3_days",
        client_event_at="2026-08-31T14:10:00Z",
    )


def test_device_registration_moves_ownership_and_never_returns_token():
    collections = {
        "notification_devices": _Collection(
            {
                "installation-123": {
                    "uid": "old-user",
                    "fcm_token": "old-token",
                    "created_at": 100,
                }
            }
        ),
        "users": _Collection(
            {"new-user": {"dateOfBirth": "2000-01-01", "audienceBand": "teen"}}
        ),
    }
    repository, firestore = _repository(collections)

    result = repository.register_device("new-user", _registration())

    stored = firestore.collection("notification_devices").documents["installation-123"]
    assert stored["uid"] == "new-user"
    assert stored["created_at"] == 100
    assert "fcm_token" not in result
    assert (
        firestore.collection("notification_states").documents["new-user"][
            "audience_policy"
        ]
        == "adult"
    )


def test_unregister_is_idempotent_and_removes_token_material():
    repository, firestore = _repository(
        {
            "notification_devices": _Collection(
                {
                    "installation-123": {
                        "uid": "user-12345",
                        "enabled": True,
                        "fcm_token": "secret-token",
                    }
                }
            )
        }
    )

    first = repository.unregister_device("user-12345", "installation-123")
    second = repository.unregister_device("user-12345", "installation-123")

    stored = firestore.collection("notification_devices").documents["installation-123"]
    assert first["found"] is True
    assert second["found"] is True
    assert stored["enabled"] is False
    assert "fcm_token" not in stored


def test_state_revisions_are_idempotent_and_reject_changed_payloads():
    repository, _ = _repository({"users": _Collection({"user-12345": {}})})
    body = _state()

    first = repository.sync_state("user-12345", body)
    duplicate = repository.sync_state("user-12345", body)

    assert first["idempotent"] is False
    assert duplicate["idempotent"] is True
    with pytest.raises(NotificationConflict):
        repository.sync_state(
            "user-12345", body.model_copy(update={"current_streak": 3})
        )
    with pytest.raises(NotificationConflict):
        repository.sync_state("user-12345", _state(revision=0))


def test_event_ids_deduplicate_and_hash_conflicts_are_rejected():
    repository, firestore = _repository(
        {
            "notification_devices": _Collection(
                {"installation-123": {"uid": "user-12345", "enabled": True}}
            ),
            "notification_states": _Collection(
                {
                    "user-12345": {
                        "uid": "user-12345",
                        "iana_timezone": "UTC",
                        "state_revision": 1,
                    }
                }
            ),
        }
    )
    event = NotificationEvent(
        event_id="operation-123:applied",
        event_type="review_applied",
        operation_id="operation-123",
        event_at="2026-08-31T14:10:00Z",
        state_revision=2,
        minimal_payload={"scheduling_day_key": "2026-08-31"},
    )

    first = repository.ingest_events("user-12345", "installation-123", (event,))
    duplicate = repository.ingest_events("user-12345", "installation-123", (event,))

    assert first == {"accepted": 1, "duplicates": 0}
    assert duplicate == {"accepted": 0, "duplicates": 1}
    assert firestore.collection("notification_events").documents[
        "operation-123:applied"
    ]["minimal_payload"] == {"scheduling_day_key": "2026-08-31"}
    assert firestore.collection("notification_states").documents["user-12345"][
        "review_event_balances"
    ] == {"2026-08-31": 1}
    with pytest.raises(NotificationConflict):
        repository.ingest_events(
            "user-12345",
            "installation-123",
            (event.model_copy(update={"state_revision": 3}),),
        )

    undone = NotificationEvent(
        event_id="operation-123:undone",
        event_type="review_undone",
        operation_id="operation-123",
        event_at="2026-08-31T14:20:00Z",
        state_revision=3,
        minimal_payload={"scheduling_day_key": "2026-08-31"},
    )
    repository.ingest_events("user-12345", "installation-123", (undone,))
    assert (
        firestore.collection("notification_states").documents["user-12345"][
            "review_event_balances"
        ]
        == {}
    )


def test_job_leases_exclude_contention_and_expire_for_recovery():
    evaluate_at = int((NOW - timedelta(minutes=1)).timestamp())
    deadline_at = int((NOW + timedelta(hours=1)).timestamp())
    repository, _ = _repository(
        {
            "notification_jobs": _Collection(
                {
                    "job-123": {
                        "job_id": "job-123",
                        "uid": "user-12345",
                        "status": "queued",
                        "evaluate_at": evaluate_at,
                        "deadline_at": deadline_at,
                        "attempt_count": 0,
                    }
                }
            )
        }
    )

    first = repository.claim_due_jobs("worker-a", NOW, 10, 120)
    contended = repository.claim_due_jobs("worker-b", NOW, 10, 120)

    assert len(first) == 1
    assert contended == ()

    recovering = NotificationRepository(
        repository.db, now_factory=lambda: NOW + timedelta(seconds=121)
    )
    recovered = recovering.claim_due_jobs(
        "worker-b", NOW + timedelta(seconds=121), 10, 120
    )
    assert len(recovered) == 1
    assert recovered[0].lease_owner == "worker-b"
    assert recovered[0].attempt_count == 2


def test_final_send_guard_requires_current_lease_state_and_device():
    now_epoch = int(NOW.timestamp())
    state_data = {
        "uid": "user-12345",
        "state_revision": 4,
        "reminders_enabled": True,
        "iana_timezone": "UTC",
        "scheduling_day_key": "2026-08-31",
        "actionable_queue_bucket": "6_10",
        "last_client_event_at": now_epoch,
        "updated_at": now_epoch,
    }
    device_data = {
        "uid": "user-12345",
        "fcm_token": "secret-token",
        "enabled": True,
        "permission_status": "authorized",
        "iana_timezone": "UTC",
        "last_seen_at": now_epoch,
    }
    job_data = {
        "uid": "user-12345",
        "status": "processing",
        "evaluate_at": now_epoch - 60,
        "deadline_at": now_epoch + 3600,
        "lease_owner": "worker-a",
        "lease_until": now_epoch + 120,
        "attempt_count": 1,
    }
    repository, firestore = _repository(
        {
            "notification_states": _Collection({"user-12345": state_data}),
            "notification_devices": _Collection({"installation-123": device_data}),
            "notification_jobs": _Collection({"job-123": job_data}),
        }
    )
    state = state_from_data("user-12345", state_data, NOW)
    device = device_from_data("installation-123", device_data)
    job = job_from_data("job-123", job_data)

    assert repository.can_send_now(job, "worker-a", state, device)

    firestore.collection("notification_jobs").documents["job-123"][
        "lease_owner"
    ] = "worker-b"
    assert not repository.can_send_now(job, "worker-a", state, device)


def test_finishing_stale_work_does_not_overwrite_newer_state_schedule():
    now_epoch = int(NOW.timestamp())
    stale_state_data = {
        "uid": "user-12345",
        "state_revision": 4,
        "reminders_enabled": True,
        "preferred_study_minute": 19 * 60,
        "iana_timezone": "UTC",
        "scheduling_day_key": "2026-08-31",
        "actionable_queue_bucket": "6_10",
        "last_client_event_at": now_epoch - 60,
        "updated_at": now_epoch - 60,
    }
    current_state_data = {
        **stale_state_data,
        "state_revision": 5,
        "updated_at": now_epoch,
        "next_evaluation_at": now_epoch + 7200,
    }
    job_data = {
        "uid": "user-12345",
        "status": "processing",
        "evaluate_at": now_epoch - 60,
        "deadline_at": now_epoch + 3600,
        "lease_owner": "worker-a",
        "lease_until": now_epoch + 120,
        "attempt_count": 1,
    }
    repository, firestore = _repository(
        {
            "notification_states": _Collection({"user-12345": current_state_data}),
            "notification_jobs": _Collection({"job-123": job_data}),
        }
    )
    stale_state = state_from_data("user-12345", stale_state_data, NOW)
    job = job_from_data("job-123", job_data)

    assert repository.finish_job(
        job,
        "worker-a",
        JobStatus.SUPPRESSED,
        "state_changed",
        stale_state,
    )
    current = firestore.collection("notification_states").documents["user-12345"]
    assert current["next_evaluation_at"] == now_epoch + 7200


def test_expired_operational_records_are_removed_without_touching_live_data():
    now_epoch = int(NOW.timestamp())
    repository, firestore = _repository(
        {
            "notification_events": _Collection(
                {
                    "expired-event": {"delete_after_at": now_epoch - 1},
                    "live-event": {"delete_after_at": now_epoch + 1},
                }
            )
        }
    )

    assert repository.purge_expired_records(NOW) == 1
    assert set(firestore.collection("notification_events").documents) == {"live-event"}
