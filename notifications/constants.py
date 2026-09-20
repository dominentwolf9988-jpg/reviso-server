from enum import StrEnum


class AudiencePolicy(StrEnum):
    TEEN = "teen"
    ADULT = "adult"
    UNKNOWN = "unknown"


class NotificationKind(StrEnum):
    NEXT_DAY_COMMITMENT = "next_day_commitment"
    HABIT_CUE = "habit_cue"
    STREAK_RESCUE = "streak_rescue"
    COMEBACK = "comeback"


class EventType(StrEnum):
    REVIEW_APPLIED = "review_applied"
    REVIEW_UNDONE = "review_undone"
    STUDY_SESSION_STARTED = "study_session_started"
    STUDY_SESSION_COMPLETED = "study_session_completed"
    APP_OPENED_FROM_NOTIFICATION = "app_opened_from_notification"
    REMINDER_ENABLED = "reminder_enabled"
    REMINDER_DISABLED = "reminder_disabled"
    TIMEZONE_CHANGED = "timezone_changed"


class PermissionStatus(StrEnum):
    AUTHORIZED = "authorized"
    DENIED = "denied"
    NOT_DETERMINED = "not_determined"
    PROVISIONAL = "provisional"


class QueueBucket(StrEnum):
    NONE = "none"
    ONE_TO_FIVE = "1_5"
    SIX_TO_TEN = "6_10"
    ELEVEN_TO_TWENTY = "11_20"
    TWENTY_ONE_PLUS = "21_plus"


class OverdueBucket(StrEnum):
    NONE = "none"
    TODAY = "today"
    ONE_TO_THREE_DAYS = "1_3_days"
    FOUR_TO_SEVEN_DAYS = "4_7_days"
    EIGHT_PLUS_DAYS = "8_plus_days"


class LifecycleSegment(StrEnum):
    ACTIVATED = "activated"
    HABIT_BUILDER = "habit_builder"
    CURRENT = "current_learner"
    STREAK_AT_RISK = "streak_at_risk"
    RECENTLY_LAPSED = "recently_lapsed"
    AT_RISK_MONTHLY = "at_risk_monthly"
    DORMANT = "dormant"


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"
    FAILED = "failed"


class SendStatus(StrEnum):
    RESERVED = "reserved"
    RETRYING = "retrying"
    FCM_ACCEPTED = "fcm_accepted"
    FAILED_PERMANENT = "failed_permanent"


class SuppressionReason(StrEnum):
    REMINDERS_DISABLED = "reminders_disabled"
    INSTALLATION_DISABLED = "installation_disabled"
    TOKEN_MISSING = "token_missing"
    UID_MISMATCH = "uid_mismatch"
    ACCOUNT_DELETION_PENDING = "account_deletion_pending"
    OUTSIDE_QUIET_HOURS = "outside_quiet_hours"
    APP_ACTIVE = "app_active"
    ALREADY_STUDIED = "already_studied"
    GOAL_ALREADY_MET = "goal_already_met"
    QUEUE_EMPTY = "queue_empty"
    STATE_STALE = "state_stale"
    DAILY_BUDGET_EXHAUSTED = "daily_budget_exhausted"
    WEEKLY_BUDGET_EXHAUSTED = "weekly_budget_exhausted"
    RESCUE_BUDGET_EXHAUSTED = "rescue_budget_exhausted"
    MINIMUM_SPACING = "minimum_spacing"
    DUPLICATE_SEND = "duplicate_send"
    TEMPLATE_COOLDOWN = "template_cooldown"
    JOB_EXPIRED = "job_expired"
    INSTALLATION_STALE = "installation_stale"
    NOT_DUE = "not_due"
    IGNORED_BACKOFF = "ignored_backoff"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    STATE_CHANGED = "state_changed"


NOTIFICATION_DEVICES = "notification_devices"
NOTIFICATION_STATES = "notification_states"
NOTIFICATION_EVENTS = "notification_events"
NOTIFICATION_JOBS = "notification_jobs"
NOTIFICATION_SENDS = "notification_sends"
NOTIFICATION_CONFIGS = "notification_configs"
NOTIFICATION_WORKER_STATUS = "notification_worker_status"
ACCOUNT_DELETIONS = "account_deletions"

SCHEDULING_DAY_BOUNDARY_HOUR = 4
TEEN_EARLIEST_MINUTE = 8 * 60
TEEN_LATEST_MINUTE = 21 * 60 + 30
ADULT_EARLIEST_MINUTE = 8 * 60
ADULT_LATEST_MINUTE = 22 * 60 + 30
STATE_FULL_FRESHNESS_SECONDS = 24 * 60 * 60
STATE_GENERIC_FRESHNESS_SECONDS = 7 * 24 * 60 * 60
DEVICE_STALE_SECONDS = 30 * 24 * 60 * 60
MINIMUM_SPACING_SECONDS = 4 * 60 * 60
MAX_TOTAL_SENDS_PER_SEVEN_DAYS = 9
MAX_RESCUES_PER_SEVEN_DAYS = 2
ROUTINE_COPY_COOLDOWN_DAYS = 7
RESCUE_COPY_COOLDOWN_DAYS = 14
EVENT_BATCH_MAX_SIZE = 100
OUTCOME_WINDOW_SECONDS = 24 * 60 * 60
# Duolingo-parity influenced window: a manual study within this long after
# an accepted send credits that template (influenced_success) even without
# a tap. OUTCOME_WINDOW_SECONDS stays the reconcile/ignore timeout only.
INFLUENCED_WINDOW_SECONDS = 2 * 60 * 60
APP_ACTIVE_GRACE_SECONDS = 5 * 60
JOB_RETENTION_SECONDS = 14 * 24 * 60 * 60
EVENT_RETENTION_SECONDS = 30 * 24 * 60 * 60
DISABLED_DEVICE_RETENTION_SECONDS = 30 * 24 * 60 * 60
SEND_RETENTION_SECONDS = 90 * 24 * 60 * 60
COMEBACK_MILESTONES = (2, 7, 21)
ROUTINE_KINDS = frozenset(
    {
        NotificationKind.NEXT_DAY_COMMITMENT,
        NotificationKind.HABIT_CUE,
        NotificationKind.COMEBACK,
    }
)

FCM_SCHEMA_VERSION = "1"
FCM_ROUTE = "global_review"
EXPERIMENT_ID = "notification_copy_v1"
