import os

import firebase_admin
import pytest

os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
os.environ["APP_INTERNAL_AUTH_KEY"] = "test-admin-key"
os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = "{}"

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

import main


class _Snapshot:
    def __init__(self, reference):
        self.reference = reference
        self.exists = reference.document_id in reference.collection.documents

    def to_dict(self):
        if not self.exists:
            return None
        return dict(self.reference.collection.documents[self.reference.document_id])


class _DocumentReference:
    def __init__(self, collection, document_id):
        self.collection = collection
        self.document_id = document_id

    def get(self):
        return _Snapshot(self)

    def set(self, data, merge=False):
        if merge:
            existing = self.collection.documents.setdefault(self.document_id, {})
            existing.update(data)
        else:
            self.collection.documents[self.document_id] = dict(data)

    def delete(self):
        self.collection.documents.pop(self.document_id, None)


class _Query:
    def __init__(self, collection, field, value):
        self.collection = collection
        self.field = field
        self.value = value
        self.result_limit = None

    def limit(self, result_limit):
        self.result_limit = result_limit
        return self

    def stream(self):
        matches = [
            _Snapshot(_DocumentReference(self.collection, document_id))
            for document_id, data in self.collection.documents.items()
            if data.get(self.field) == self.value
        ]
        return matches[: self.result_limit]


class _Collection:
    def __init__(self, documents=None):
        self.documents = documents or {}

    def document(self, document_id):
        return _DocumentReference(self, document_id)

    def where(self, field, operator, value):
        assert operator == "=="
        return _Query(self, field, value)


class _Batch:
    def __init__(self):
        self.references = []

    def delete(self, reference):
        self.references.append(reference)

    def commit(self):
        for reference in self.references:
            reference.delete()


class _Firestore:
    def __init__(self, collections):
        self.collections = collections

    def collection(self, name):
        return self.collections.setdefault(name, _Collection())

    def batch(self):
        return _Batch()


def test_backend_deletion_retains_permanent_tombstone_and_is_idempotent(monkeypatch):
    uid = "firebase-user"
    direct_collections = (
        "users",
        "user_credits",
        "whitelisted_users",
        "ad_reward_users",
        "ai_report_users",
        "notification_states",
    )
    query_collections = (
        "ad_reward_intents",
        "admob_ssv_transactions",
        "ai_reports",
        "notification_devices",
        "notification_events",
        "notification_jobs",
        "notification_sends",
    )
    collections = {
        "account_deletions": _Collection(
            {uid: {"status": "drive_pending", "requested_at": 100}}
        )
    }
    for name in direct_collections:
        collections[name] = _Collection({uid: {"value": 1}, "other": {"value": 2}})
    for name in query_collections:
        collections[name] = _Collection(
            {
                "owned": {"uid": uid},
                "unrelated": {"uid": "other"},
            }
        )
    firestore = _Firestore(collections)
    monkeypatch.setattr(main.fb_fs, "client", lambda: firestore)

    first = main._delete_account_data_sync(uid)
    second = main._delete_account_data_sync(uid)

    assert first == {"status": "backend_deleted", "already_deleted": False}
    assert second == {"status": "backend_deleted", "already_deleted": True}
    for name in direct_collections:
        assert uid not in collections[name].documents
        assert "other" in collections[name].documents
    for name in query_collections:
        assert "owned" not in collections[name].documents
        assert "unrelated" in collections[name].documents

    tombstone = collections["account_deletions"].documents[uid]
    assert tombstone["status"] == "backend_deleted"
    assert tombstone["requested_at"] == 100
    assert "completed_at" in tombstone
    assert "expire_at" not in tombstone
