"""Duolingo-style influenced attribution: manual study within 2h of a send.

Tap-linked events keep their columns; a manual review (no notification_id)
inside INFLUENCED_WINDOW_SECONDS credits the latest unresolved accepted
send as influenced_success without touching tap_at.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from firebase_admin import firestore as fb_fs

from notifications.constants import INFLUENCED_WINDOW_SECONDS, SendStatus
from notifications.models import NotificationEvent
from notifications.repository import NotificationRepository

UTC = timezone.utc
NOW = datetime(2026, 8, 31, 14, 30, tzinfo=UTC)
SEND_KEY = "a" * 64


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


def _epoch(value):
    return int(value.timestamp())


def _send(accepted_at, **overrides):
    doc = {
        "uid": "user-12345",
        "send_status": SendStatus.FCM_ACCEPTED.value,
        "fcm_accepted_at": _epoch(accepted_at),
        "tap_at": None,
        "review_started_at": None,
        "review_completed_at": None,
        "influenced_studied_at": None,
        "outcome": None,
        "outcome_resolved": False,
        "ignore_after_at": _epoch(accepted_at) + 24 * 60 * 60,
    }
    doc.update(overrides)
    return doc


def _collections(send_docs):
    return {
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
        "notification_sends": _Collection(dict(send_docs)),
    }


def _repository(collections, now=NOW):
    firestore = _Firestore(collections)
    return NotificationRepository(firestore, now_factory=lambda: now), firestore


def _review(event_id, at, operation_id="operation-123"):
    return NotificationEvent(
        event_id=event_id,
        event_type="review_applied",
        operation_id=operation_id,
        event_at=at,
        state_revision=2,
        minimal_payload={"scheduling_day_key": "2026-08-31"},
    )


def _session_completed(event_id, at):
    return NotificationEvent(
        event_id=event_id,
        event_type="study_session_completed",
        event_at=at,
        state_revision=3,
        minimal_payload={
            "notification_id": SEND_KEY,
            "session_id": "session-12345",
        },
    )


def test_influenced_window_is_two_hours():
    assert INFLUENCED_WINDOW_SECONDS == 2 * 60 * 60


def test_manual_review_inside_window_credits_influenced_success():
    accepted_at = NOW - timedelta(minutes=30)
    repository, firestore = _repository(_collections({SEND_KEY: _send(accepted_at)}))

    result = repository.ingest_events(
        "user-12345",
        "installation-123",
        (_review("operation-123:applied", NOW),),
    )

    assert result == {"accepted": 1, "duplicates": 0}
    stored = firestore.collection("notification_sends").documents[SEND_KEY]
    assert stored["influenced_studied_at"] == _epoch(NOW)
    assert stored["outcome"] == "influenced_success"
    assert stored["outcome_resolved"] is True
    assert stored["tap_at"] is None
    assert (
        firestore.collection("notification_states").documents["user-12345"][
            "consecutive_ignored_count"
        ]
        == 0
    )


def test_manual_review_outside_window_leaves_send_untouched():
    accepted_at = NOW - timedelta(hours=3)
    repository, firestore = _repository(_collections({SEND_KEY: _send(accepted_at)}))

    repository.ingest_events(
        "user-12345",
        "installation-123",
        (_review("operation-123:applied", NOW),),
    )

    stored = firestore.collection("notification_sends").documents[SEND_KEY]
    assert stored["influenced_studied_at"] is None
    assert stored["outcome"] is None
    assert stored["outcome_resolved"] is False


def test_fallback_skips_tapped_sends():
    accepted_at = NOW - timedelta(minutes=30)
    repository, firestore = _repository(
        _collections(
            {
                SEND_KEY: _send(
                    accepted_at,
                    tap_at=_epoch(accepted_at) + 60,
                    outcome_resolved=False,
                )
            }
        )
    )

    repository.ingest_events(
        "user-12345",
        "installation-123",
        (_review("operation-123:applied", NOW),),
    )

    stored = firestore.collection("notification_sends").documents[SEND_KEY]
    assert stored["influenced_studied_at"] is None
    assert stored["outcome"] is None


def test_tap_linked_completion_upgrades_influenced_to_tapped():
    accepted_at = NOW - timedelta(minutes=90)
    repository, firestore = _repository(_collections({SEND_KEY: _send(accepted_at)}))

    repository.ingest_events(
        "user-12345",
        "installation-123",
        (_review("operation-123:applied", NOW - timedelta(minutes=30)),),
    )
    repository.ingest_events(
        "user-12345",
        "installation-123",
        (_session_completed("session-12345:completed", NOW),),
    )

    stored = firestore.collection("notification_sends").documents[SEND_KEY]
    assert stored["influenced_studied_at"] is not None
    assert stored["review_completed_at"] == _epoch(NOW)
    assert stored["outcome"] == "tapped_success"


def test_stale_tap_linked_completion_is_not_credited():
    accepted_at = NOW - timedelta(hours=25)
    repository, firestore = _repository(_collections({SEND_KEY: _send(accepted_at)}))

    repository.ingest_events(
        "user-12345",
        "installation-123",
        (_session_completed("session-12345:completed", NOW),),
    )

    stored = firestore.collection("notification_sends").documents[SEND_KEY]
    assert stored.get("review_completed_at") is None
    assert stored.get("outcome") is None
    assert stored["outcome_resolved"] is False


def test_reconcile_ignores_only_truly_unresolved_sends():
    accepted_at = NOW - timedelta(hours=25)
    other_key = "b" * 64
    repository, firestore = _repository(
        _collections(
            {
                SEND_KEY: _send(
                    accepted_at,
                    influenced_studied_at=_epoch(accepted_at) + 600,
                    outcome="influenced_success",
                    outcome_resolved=True,
                ),
                other_key: _send(accepted_at),
            }
        )
    )

    assert repository.reconcile_ignored_sends(NOW) == 1
    sends = firestore.collection("notification_sends").documents
    assert sends[SEND_KEY]["outcome"] == "influenced_success"
    assert sends[other_key]["outcome"] == "ignored"
    assert (
        firestore.collection("notification_states").documents["user-12345"][
            "consecutive_ignored_count"
        ]
        == 1
    )
