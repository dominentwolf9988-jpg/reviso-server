import os

import firebase_admin
from fastapi.testclient import TestClient

os.environ["OPENROUTER_API_KEY"] = "test-openrouter-key"
os.environ["APP_INTERNAL_AUTH_KEY"] = "test-admin-key"
os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = "{}"

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

import main
from notifications.repository import NotificationConflict

DEVICE = {
    "installation_id": "installation-123",
    "fcm_token": "fcm-token-value-with-enough-length",
    "platform": "android",
    "app_version": "1.2.3",
    "locale": "en-IN",
    "iana_timezone": "Asia/Kolkata",
    "permission_status": "authorized",
}


class _FakeService:
    def __init__(self):
        self.calls = []

    def register_device(self, uid, body):
        self.calls.append(("register", uid, body))
        return {
            "installation_id": body.installation_id,
            "enabled": True,
            "permission_status": "authorized",
        }

    def unregister_device(self, uid, installation_id):
        self.calls.append(("unregister", uid, installation_id))
        return {"installation_id": installation_id, "enabled": False, "found": True}

    def update_settings(self, uid, body):
        self.calls.append(("settings", uid, body))
        return {"reminders_enabled": body.reminders_enabled, "next_evaluation_at": None}

    def sync_state(self, uid, body):
        self.calls.append(("state", uid, body))
        return {
            "state_revision": body.state_revision,
            "idempotent": False,
            "next_evaluation_at": None,
        }

    def ingest_events(self, uid, body):
        self.calls.append(("events", uid, body))
        return {"accepted": len(body.events), "duplicates": 0}


def _authorized_client(monkeypatch, service=None):
    service = service or _FakeService()

    async def verified_uid(authorization):
        assert authorization == "Bearer verified-token"
        return "verified-firebase-uid"

    monkeypatch.setattr(main, "get_uid", verified_uid)
    monkeypatch.setattr(main, "_notification_service", service)
    return TestClient(main.app), service


def test_notification_endpoints_require_firebase_bearer_token():
    response = TestClient(main.app).post(
        "/v1/notifications/devices/register", json=DEVICE
    )

    assert response.status_code == 401


def test_register_device_derives_uid_and_never_returns_token(monkeypatch):
    client, service = _authorized_client(monkeypatch)

    response = client.post(
        "/v1/notifications/devices/register",
        headers={"Authorization": "Bearer verified-token"},
        json={**DEVICE, "uid": "attacker-controlled"},
    )

    assert response.status_code == 422

    response = client.post(
        "/v1/notifications/devices/register",
        headers={"Authorization": "Bearer verified-token"},
        json=DEVICE,
    )
    assert response.status_code == 200
    assert service.calls[0][1] == "verified-firebase-uid"
    assert "fcm_token" not in response.json()


def test_delete_and_settings_contracts(monkeypatch):
    client, service = _authorized_client(monkeypatch)
    headers = {"Authorization": "Bearer verified-token"}

    deleted = client.post(
        "/v1/notifications/devices/unregister",
        headers=headers,
        json={"installation_id": "installation-123"},
    )
    settings = client.post(
        "/v1/notifications/preferences",
        headers=headers,
        json={
            "installation_id": "installation-123",
            "reminders_enabled": True,
            "preferred_study_minute": 1140,
            "quiet_hours_start_minute": 480,
            "quiet_hours_end_minute": 1350,
            "iana_timezone": "Asia/Kolkata",
        },
    )

    assert deleted.status_code == 200
    assert settings.status_code == 200
    assert [call[0] for call in service.calls] == ["unregister", "settings"]


def test_state_contract_accepts_plan_snapshot_without_optional_aggregates(monkeypatch):
    client, service = _authorized_client(monkeypatch)

    response = client.post(
        "/v1/notifications/state",
        headers={"Authorization": "Bearer verified-token"},
        json={
            "state_revision": 42,
            "scheduling_day_key": "2026-08-31",
            "last_study_at": "2026-08-31T14:10:00Z",
            "current_streak": 6,
            "daily_goal": 10,
            "cards_studied_today": 4,
            "actionable_queue_bucket": "6_10",
            "oldest_overdue_bucket": "1_3_days",
            "client_event_at": "2026-08-31T14:10:00Z",
        },
    )

    assert response.status_code == 200
    assert service.calls[0][2].state_revision == 42


def test_batch_event_contract_and_deterministic_review_id(monkeypatch):
    client, service = _authorized_client(monkeypatch)

    response = client.post(
        "/v1/notifications/events:batch",
        headers={"Authorization": "Bearer verified-token"},
        json={
            "installation_id": "installation-123",
            "events": [
                {
                    "event_id": "review-operation-123:applied",
                    "event_type": "review_applied",
                    "operation_id": "review-operation-123",
                    "event_at": "2026-08-31T14:10:00Z",
                    "state_revision": 42,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": 1, "duplicates": 0}
    assert service.calls[0][1] == "verified-firebase-uid"

    invalid = client.post(
        "/v1/notifications/events:batch",
        headers={"Authorization": "Bearer verified-token"},
        json={
            "installation_id": "installation-123",
            "events": [
                {
                    "event_id": "different-event-id",
                    "event_type": "review_applied",
                    "operation_id": "review-operation-123",
                    "event_at": "2026-08-31T14:10:00Z",
                    "state_revision": 42,
                }
            ],
        },
    )
    assert invalid.status_code == 422


def test_invalid_timezone_and_oversized_batch_are_rejected(monkeypatch):
    client, _ = _authorized_client(monkeypatch)
    headers = {"Authorization": "Bearer verified-token"}

    invalid_timezone = client.post(
        "/v1/notifications/devices/register",
        headers=headers,
        json={**DEVICE, "iana_timezone": "Not/A_Timezone"},
    )
    events = [
        {
            "event_id": f"event-identifier-{index}",
            "event_type": "reminder_enabled",
            "event_at": "2026-08-31T14:10:00Z",
            "state_revision": 42,
        }
        for index in range(101)
    ]
    oversized = client.post(
        "/v1/notifications/events:batch",
        headers=headers,
        json={"installation_id": "installation-123", "events": events},
    )

    assert invalid_timezone.status_code == 422
    assert oversized.status_code == 422


def test_repository_conflicts_are_reported_as_http_409(monkeypatch):
    service = _FakeService()

    def conflict(uid, body):
        raise NotificationConflict("state revision is stale")

    service.sync_state = conflict
    client, _ = _authorized_client(monkeypatch, service)
    response = client.post(
        "/v1/notifications/state",
        headers={"Authorization": "Bearer verified-token"},
        json={
            "state_revision": 1,
            "scheduling_day_key": "2026-08-31",
            "current_streak": 0,
            "daily_goal": 10,
            "cards_studied_today": 0,
            "actionable_queue_bucket": "none",
            "client_event_at": "2026-08-31T14:10:00Z",
        },
    )

    assert response.status_code == 409
