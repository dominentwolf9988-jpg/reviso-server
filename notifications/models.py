from __future__ import annotations

from datetime import date, datetime
import re
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .constants import (
    ADULT_EARLIEST_MINUTE,
    ADULT_LATEST_MINUTE,
    AudiencePolicy,
    EVENT_BATCH_MAX_SIZE,
    EventType,
    JobStatus,
    NotificationKind,
    OverdueBucket,
    PermissionStatus,
    QueueBucket,
    SendStatus,
)

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=8,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]


def validate_timezone_name(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 64:
        raise ValueError("iana_timezone must be a valid IANA timezone")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("iana_timezone must be a valid IANA timezone") from exc
    return value


def _aware_datetime(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeviceRegistrationRequest(StrictModel):
    installation_id: Identifier
    fcm_token: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=20, max_length=4096),
    ]
    platform: Literal["android"] = "android"
    app_version: ShortText
    locale: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=2,
            max_length=35,
            pattern=r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$",
        ),
    ]
    iana_timezone: str
    permission_status: PermissionStatus

    _timezone = field_validator("iana_timezone")(validate_timezone_name)


class DeviceUnregistrationRequest(StrictModel):
    installation_id: Identifier


class NotificationSettingsRequest(StrictModel):
    installation_id: Identifier
    reminders_enabled: bool
    preferred_study_minute: int = Field(ge=0, le=1439, default=19 * 60)
    quiet_hours_start_minute: int = Field(ge=0, le=1439, default=ADULT_EARLIEST_MINUTE)
    quiet_hours_end_minute: int = Field(ge=0, le=1439, default=ADULT_LATEST_MINUTE)
    iana_timezone: str

    _timezone = field_validator("iana_timezone")(validate_timezone_name)

    @model_validator(mode="after")
    def valid_delivery_window(self):
        if self.quiet_hours_start_minute >= self.quiet_hours_end_minute:
            raise ValueError("notification delivery window may not cross midnight")
        if not (
            self.quiet_hours_start_minute
            <= self.preferred_study_minute
            <= self.quiet_hours_end_minute
        ):
            raise ValueError(
                "preferred_study_minute must be inside the delivery window"
            )
        return self


class NotificationStateRequest(StrictModel):
    state_revision: int = Field(ge=0)
    scheduling_day_key: date
    last_study_at: datetime | None = None
    last_qualified_study_day: date | None = None
    current_streak: int = Field(ge=0, le=100_000)
    daily_goal: int = Field(ge=1, le=10_000)
    cards_studied_today: int = Field(ge=0, le=1_000_000)
    actionable_queue_bucket: QueueBucket
    oldest_overdue_bucket: OverdueBucket = OverdueBucket.NONE
    total_qualified_study_days: int = Field(default=0, ge=0, le=1_000_000)
    qualified_study_days_last_7: int = Field(default=0, ge=0, le=7)
    client_event_at: datetime
    app_active: bool = False
    iana_timezone: str | None = None

    _last_study_aware = field_validator("last_study_at")(_aware_datetime)
    _client_event_aware = field_validator("client_event_at")(_aware_datetime)

    @field_validator("iana_timezone")
    @classmethod
    def optional_timezone(cls, value: str | None) -> str | None:
        return validate_timezone_name(value) if value is not None else None

    @model_validator(mode="after")
    def consistent_progress(self):
        if self.cards_studied_today > 0 and self.last_study_at is None:
            raise ValueError("last_study_at is required after studying today")
        return self


MinimalEventPayload = dict[str, str | int | bool]
_EVENT_PAYLOAD_KEYS = frozenset(
    {
        "notification_id",
        "session_id",
        "scheduling_day_key",
        "iana_timezone",
    }
)
_PAYLOAD_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,159}$")
_NOTIFICATION_ID = re.compile(r"^[a-f0-9]{64}$")


class NotificationEvent(StrictModel):
    event_id: Identifier
    event_type: EventType
    operation_id: Identifier | None = None
    event_at: datetime
    state_revision: int = Field(ge=0)
    minimal_payload: MinimalEventPayload = Field(default_factory=dict)

    _event_at_aware = field_validator("event_at")(_aware_datetime)

    @model_validator(mode="after")
    def valid_minimal_event(self):
        unknown = set(self.minimal_payload) - _EVENT_PAYLOAD_KEYS
        if unknown:
            raise ValueError("minimal_payload contains unsupported fields")
        if len(str(self.minimal_payload)) > 2048:
            raise ValueError("minimal_payload is too large")
        for value in self.minimal_payload.values():
            if isinstance(value, str) and len(value) > 200:
                raise ValueError("minimal_payload value is too large")
        notification_id = self.minimal_payload.get("notification_id")
        if notification_id is not None and (
            not isinstance(notification_id, str)
            or _NOTIFICATION_ID.fullmatch(notification_id) is None
        ):
            raise ValueError("notification_id must be a server notification ID")
        session_id = self.minimal_payload.get("session_id")
        if session_id is not None and (
            not isinstance(session_id, str)
            or _PAYLOAD_IDENTIFIER.fullmatch(session_id) is None
        ):
            raise ValueError("session_id must be a valid identifier")
        day_key = self.minimal_payload.get("scheduling_day_key")
        if day_key is not None:
            try:
                date.fromisoformat(str(day_key))
            except ValueError as exc:
                raise ValueError("scheduling_day_key must be an ISO date") from exc
        timezone_name = self.minimal_payload.get("iana_timezone")
        if timezone_name is not None:
            if not isinstance(timezone_name, str):
                raise ValueError("iana_timezone must be a string")
            validate_timezone_name(timezone_name)

        expected_suffix = {
            EventType.REVIEW_APPLIED: "applied",
            EventType.REVIEW_UNDONE: "undone",
        }.get(self.event_type)
        if expected_suffix is not None:
            if self.operation_id is None:
                raise ValueError("operation_id is required for review events")
            if self.event_id != f"{self.operation_id}:{expected_suffix}":
                raise ValueError("review event_id must be derived from operation_id")
        return self


class NotificationEventBatchRequest(StrictModel):
    installation_id: Identifier
    events: list[NotificationEvent] = Field(
        min_length=1, max_length=EVENT_BATCH_MAX_SIZE
    )

    @field_validator("events")
    @classmethod
    def distinct_event_ids(cls, events: list[NotificationEvent]):
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("events must have distinct event_id values")
        return events


class DeviceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    installation_id: str
    uid: str
    fcm_token: str | None
    platform: str = "android"
    enabled: bool
    permission_status: PermissionStatus
    iana_timezone: str
    last_seen_at: datetime


class NotificationState(BaseModel):
    model_config = ConfigDict(frozen=True)

    uid: str
    state_revision: int = 0
    audience_policy: AudiencePolicy = AudiencePolicy.UNKNOWN
    reminders_enabled: bool = False
    preferred_study_minute: int = 19 * 60
    quiet_hours_start_minute: int = ADULT_EARLIEST_MINUTE
    quiet_hours_end_minute: int = ADULT_LATEST_MINUTE
    iana_timezone: str = "UTC"
    scheduling_day_key: date
    last_study_at: datetime | None = None
    last_qualified_study_day: date | None = None
    current_streak: int = 0
    daily_goal: int = 10
    cards_studied_today: int = 0
    actionable_queue_bucket: QueueBucket = QueueBucket.NONE
    oldest_overdue_bucket: OverdueBucket = OverdueBucket.NONE
    total_qualified_study_days: int = 0
    qualified_study_days_last_7: int = 0
    successful_study_minutes: tuple[int, ...] = ()
    learned_study_minute: int | None = None
    review_event_balances: dict[str, int] = Field(default_factory=dict)
    consecutive_ignored_count: int = 0
    app_active_until: datetime | None = None
    last_client_event_at: datetime
    updated_at: datetime


class SendHistory(BaseModel):
    model_config = ConfigDict(frozen=True)

    send_key: str
    notification_kind: NotificationKind
    copy_id: str
    scheduling_day_key: date
    sent_at: datetime
    send_status: SendStatus = SendStatus.FCM_ACCEPTED
    tone: str = "neutral"
    comeback_milestone: int | None = None


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    uid: str
    status: JobStatus
    evaluate_at: datetime
    deadline_at: datetime
    lease_owner: str | None = None
    lease_until: datetime | None = None
    attempt_count: int = 0
