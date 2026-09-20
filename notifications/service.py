from __future__ import annotations

from .models import (
    DeviceRegistrationRequest,
    NotificationEventBatchRequest,
    NotificationSettingsRequest,
    NotificationStateRequest,
)
from .repository import NotificationRepository


class NotificationService:
    """Application boundary for authenticated notification API operations."""

    def __init__(self, repository: NotificationRepository | None = None):
        self.repository = repository or NotificationRepository()

    def register_device(self, uid: str, body: DeviceRegistrationRequest) -> dict:
        return self.repository.register_device(uid, body)

    def unregister_device(self, uid: str, installation_id: str) -> dict:
        return self.repository.unregister_device(uid, installation_id)

    def update_settings(self, uid: str, body: NotificationSettingsRequest) -> dict:
        return self.repository.update_settings(uid, body)

    def sync_state(self, uid: str, body: NotificationStateRequest) -> dict:
        return self.repository.sync_state(uid, body)

    def ingest_events(self, uid: str, body: NotificationEventBatchRequest) -> dict:
        return self.repository.ingest_events(uid, body.installation_id, body.events)
