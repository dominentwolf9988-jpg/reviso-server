from datetime import date, datetime, timedelta, timezone

import pytest

from notifications.constants import (
    AudiencePolicy,
    NotificationKind,
    SendStatus,
    SuppressionReason,
)
from notifications.decision_engine import (
    DecisionContext,
    decide,
    stable_experiment_variant,
)
from notifications.models import (
    DeviceRecord,
    NotificationState,
    SendHistory,
)
from notifications.scheduling import (
    effective_routine_minute,
    learned_habit_minute,
    local_datetime_for_scheduling_day,
    scheduling_day_key,
)
from notifications.templates import (
    render_template,
    templates_for,
)

UTC = timezone.utc
NOW = datetime(2026, 8, 31, 19, 0, tzinfo=UTC)


def _state(now: datetime = NOW, **updates) -> NotificationState:
    values = {
        "uid": "firebase-user-123",
        "state_revision": 4,
        "audience_policy": AudiencePolicy.ADULT,
        "reminders_enabled": True,
        "preferred_study_minute": 19 * 60,
        "quiet_hours_start_minute": 8 * 60,
        "quiet_hours_end_minute": 22 * 60 + 30,
        "iana_timezone": "UTC",
        "scheduling_day_key": scheduling_day_key(now, "UTC"),
        "last_study_at": now - timedelta(days=1),
        "last_qualified_study_day": scheduling_day_key(now, "UTC") - timedelta(days=1),
        "current_streak": 2,
        "daily_goal": 10,
        "cards_studied_today": 0,
        "actionable_queue_bucket": "6_10",
        "oldest_overdue_bucket": "1_3_days",
        "total_qualified_study_days": 8,
        "qualified_study_days_last_7": 6,
        "consecutive_ignored_count": 0,
        "app_active_until": None,
        "last_client_event_at": now - timedelta(minutes=2),
        "updated_at": now - timedelta(minutes=2),
    }
    values.update(updates)
    return NotificationState(**values)


def _device(now: datetime = NOW, **updates) -> DeviceRecord:
    values = {
        "installation_id": "installation-123",
        "uid": "firebase-user-123",
        "fcm_token": "fcm-token-value-with-enough-length",
        "enabled": True,
        "permission_status": "authorized",
        "iana_timezone": "UTC",
        "last_seen_at": now - timedelta(minutes=1),
    }
    values.update(updates)
    return DeviceRecord(**values)


def _history(
    now: datetime,
    kind: NotificationKind,
    copy_id: str,
    *,
    sent_at: datetime,
    day: date | None = None,
    milestone: int | None = None,
    tone: str = "neutral",
) -> SendHistory:
    return SendHistory(
        send_key=f"send-{copy_id}-{sent_at.timestamp()}",
        notification_kind=kind,
        copy_id=copy_id,
        scheduling_day_key=day or scheduling_day_key(sent_at, "UTC"),
        sent_at=sent_at,
        send_status=SendStatus.FCM_ACCEPTED,
        comeback_milestone=milestone,
        tone=tone,
    )


def _decide(
    now: datetime = NOW,
    *,
    state: NotificationState | None = None,
    device: DeviceRecord | None = None,
    history=(),
    deleting: bool = False,
    deadline: datetime | None = None,
    existing=frozenset(),
):
    return decide(
        DecisionContext(
            state=state or _state(now),
            device=device if device is not None else _device(now),
            now=now,
            job_deadline=deadline or now + timedelta(hours=4),
            deletion_pending=deleting,
            history=tuple(history),
            existing_send_keys=frozenset(existing),
            experiment_salt="test",
        )
    )


def test_habit_cue_is_due_at_preferred_time_and_stable():
    first = _decide()
    second = _decide()

    assert first.should_send
    assert first.notification_kind == NotificationKind.HABIT_CUE
    assert first.send_key == second.send_key
    assert first.copy_id == second.copy_id


def test_before_preferred_time_is_not_due():
    now = NOW - timedelta(hours=1)
    result = _decide(now, state=_state(now, preferred_study_minute=19 * 60))

    assert result.reason == SuppressionReason.NOT_DUE


def test_first_qualified_day_gets_next_day_commitment():
    state = _state(
        total_qualified_study_days=1,
        current_streak=1,
        last_qualified_study_day=date(2026, 8, 30),
    )

    result = _decide(state=state)

    assert result.notification_kind == NotificationKind.NEXT_DAY_COMMITMENT


def test_streak_rescue_requires_routine_spacing_and_fresh_state():
    now = datetime(2026, 8, 31, 21, 30, tzinfo=UTC)
    routine = _history(
        now,
        NotificationKind.HABIT_CUE,
        "habit_safe_1",
        sent_at=now - timedelta(hours=4),
        day=date(2026, 8, 31),
    )
    state = _state(now, current_streak=12)

    result = _decide(now, state=state, device=_device(now), history=(routine,))

    assert result.should_send
    assert result.notification_kind == NotificationKind.STREAK_RESCUE

    stale = state.model_copy(update={"updated_at": now - timedelta(hours=25)})
    result = _decide(now, state=stale, device=_device(now), history=(routine,))
    assert result.reason == SuppressionReason.STATE_STALE


def test_late_preference_can_go_directly_to_streak_rescue():
    now = datetime(2026, 8, 31, 21, 30, tzinfo=UTC)
    state = _state(now, current_streak=5, preferred_study_minute=22 * 60)

    result = _decide(now, state=state, device=_device(now))

    assert result.notification_kind == NotificationKind.STREAK_RESCUE


@pytest.mark.parametrize(
    ("inactive_days", "milestone"),
    [(2, 2), (8, 7), (30, 21)],
)
def test_lapsed_users_follow_comeback_milestones(inactive_days, milestone):
    state = _state(
        current_streak=0,
        last_qualified_study_day=date(2026, 8, 31) - timedelta(days=inactive_days),
    )

    result = _decide(state=state)

    assert result.notification_kind == NotificationKind.COMEBACK
    assert result.comeback_milestone == milestone


def test_comeback_milestone_is_only_sent_once():
    sent = _history(
        NOW,
        NotificationKind.COMEBACK,
        "comeback_safe_1",
        sent_at=NOW - timedelta(days=2),
        day=date(2026, 8, 29),
        milestone=7,
    )
    state = _state(
        current_streak=0,
        last_qualified_study_day=date(2026, 8, 23),
    )

    result = _decide(state=state, history=(sent,))

    assert not result.should_send


@pytest.mark.parametrize(
    ("state_updates", "device_updates", "deleting", "reason"),
    [
        ({"reminders_enabled": False}, {}, False, SuppressionReason.REMINDERS_DISABLED),
        ({}, {"enabled": False}, False, SuppressionReason.INSTALLATION_DISABLED),
        ({}, {"fcm_token": None}, False, SuppressionReason.TOKEN_MISSING),
        ({}, {"uid": "different-user"}, False, SuppressionReason.UID_MISMATCH),
        ({}, {}, True, SuppressionReason.ACCOUNT_DELETION_PENDING),
        ({"actionable_queue_bucket": "none"}, {}, False, SuppressionReason.QUEUE_EMPTY),
    ],
)
def test_fail_closed_gates(state_updates, device_updates, deleting, reason):
    result = _decide(
        state=_state(**state_updates),
        device=_device(**device_updates),
        deleting=deleting,
    )

    assert result.reason == reason


def test_already_studied_and_goal_met_suppress_all_reminders():
    studied = _state(
        cards_studied_today=1,
        last_qualified_study_day=date(2026, 8, 31),
    )
    goal_met = studied.model_copy(update={"cards_studied_today": 10})

    assert _decide(state=studied).reason == SuppressionReason.ALREADY_STUDIED
    assert _decide(state=goal_met).reason == SuppressionReason.GOAL_ALREADY_MET

    event_only = _state(review_event_balances={"2026-08-31": 1})
    assert _decide(state=event_only).reason == SuppressionReason.ALREADY_STUDIED


def test_daily_weekly_and_spacing_caps_are_hard_limits():
    routine_today = _history(
        NOW,
        NotificationKind.HABIT_CUE,
        "habit_safe_1",
        sent_at=NOW - timedelta(hours=5),
        day=date(2026, 8, 31),
    )
    assert (
        _decide(history=(routine_today,)).reason
        == SuppressionReason.DAILY_BUDGET_EXHAUSTED
    )

    too_recent = routine_today.model_copy(update={"sent_at": NOW - timedelta(hours=1)})
    assert _decide(history=(too_recent,)).reason == SuppressionReason.MINIMUM_SPACING

    weekly = tuple(
        _history(
            NOW,
            NotificationKind.HABIT_CUE,
            f"copy-{index}",
            sent_at=NOW - timedelta(hours=5 + index * 8),
            day=date(2026, 8, 30 - index // 3),
        )
        for index in range(9)
    )
    assert _decide(history=weekly).reason == SuppressionReason.WEEKLY_BUDGET_EXHAUSTED


def test_rescue_has_rolling_seven_day_cap():
    now = datetime(2026, 8, 31, 21, 30, tzinfo=UTC)
    history = (
        _history(
            now,
            NotificationKind.HABIT_CUE,
            "habit_safe_1",
            sent_at=now - timedelta(hours=5),
            day=date(2026, 8, 31),
        ),
        _history(
            now,
            NotificationKind.STREAK_RESCUE,
            "rescue_safe_1",
            sent_at=now - timedelta(days=2),
        ),
        _history(
            now,
            NotificationKind.STREAK_RESCUE,
            "rescue_safe_2",
            sent_at=now - timedelta(days=5),
        ),
    )

    result = _decide(
        now,
        state=_state(now, current_streak=10),
        device=_device(now),
        history=history,
    )

    assert result.reason == SuppressionReason.RESCUE_BUDGET_EXHAUSTED


def test_ignored_backoff_moves_daily_user_to_comeback_cadence():
    state = _state(consecutive_ignored_count=3)

    assert _decide(state=state).reason == SuppressionReason.IGNORED_BACKOFF

    lapsed = state.model_copy(
        update={
            "current_streak": 0,
            "last_qualified_study_day": date(2026, 8, 29),
        }
    )
    assert _decide(state=lapsed).notification_kind == NotificationKind.COMEBACK


def test_template_cooldowns_fail_closed_when_every_copy_is_recent():
    history = tuple(
        _history(
            NOW,
            NotificationKind.HABIT_CUE,
            template.copy_id,
            sent_at=NOW - timedelta(days=1, minutes=index),
            day=date(2026, 8, 30),
        )
        for index, template in enumerate(
            templates_for(NotificationKind.HABIT_CUE, AudiencePolicy.TEEN, "fixed")
        )
    )
    state = _state(audience_policy=AudiencePolicy.TEEN)

    result = _decide(state=state, history=history)

    assert result.reason == SuppressionReason.TEMPLATE_COOLDOWN


def test_adults_have_stable_ab_assignment_and_teens_use_fixed_safe_copy():
    assert stable_experiment_variant(
        "firebase-user-123", "copy-v1", "salt"
    ) == stable_experiment_variant("firebase-user-123", "copy-v1", "salt")
    adult = _decide()
    teen = _decide(state=_state(audience_policy=AudiencePolicy.TEEN))

    assert adult.experiment_id != "none"
    assert adult.variant_id in {"control", "treatment"}
    assert teen.experiment_id == "none"
    assert teen.variant_id == "fixed"
    title, body = render_template(teen.copy_id, {})
    assert "disappeared" not in f"{title} {body}".lower()


def test_quiet_hour_boundaries_and_job_deadline():
    earliest = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)
    at_start = _state(
        earliest,
        preferred_study_minute=8 * 60,
        updated_at=earliest - timedelta(minutes=1),
    )
    assert _decide(earliest, state=at_start, device=_device(earliest)).should_send

    after_end = datetime(2026, 8, 31, 22, 31, tzinfo=UTC)
    assert (
        _decide(after_end, state=_state(after_end), device=_device(after_end)).reason
        == SuppressionReason.OUTSIDE_QUIET_HOURS
    )
    assert (
        _decide(deadline=NOW - timedelta(seconds=1)).reason
        == SuppressionReason.JOB_EXPIRED
    )


def test_generic_habit_is_allowed_for_week_old_state_but_exact_rescue_is_not():
    generic = _state(updated_at=NOW - timedelta(days=2))
    assert _decide(state=generic).notification_kind == NotificationKind.HABIT_CUE

    stale = _state(updated_at=NOW - timedelta(days=8))
    assert _decide(state=stale).reason == SuppressionReason.STATE_STALE


def test_scheduling_day_boundary_and_dst_are_timezone_aware():
    assert scheduling_day_key(datetime(2026, 8, 31, 3, 59, tzinfo=UTC), "UTC") == date(
        2026, 8, 30
    )
    assert scheduling_day_key(datetime(2026, 8, 31, 4, 0, tzinfo=UTC), "UTC") == date(
        2026, 8, 31
    )

    spring = local_datetime_for_scheduling_day(
        date(2026, 3, 7), 2 * 60 + 30, "America/New_York"
    )
    spring_local = spring.astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")
    )
    assert (spring_local.hour, spring_local.minute) == (3, 30)

    fall = local_datetime_for_scheduling_day(
        date(2026, 10, 31), 1 * 60 + 30, "America/New_York"
    )
    fall_local = fall.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    assert fall_local.fold == 0


def test_learned_time_requires_three_days_and_stays_in_delivery_window():
    assert learned_habit_minute([500, 510], 600) is None
    assert learned_habit_minute([800, 810, 820], 600) == 810
    assert learned_habit_minute([100, 110, 120], 600, 480, 1350) == 480


def test_routine_uses_learned_time_after_three_study_days():
    before_learned_time = datetime(2026, 8, 31, 18, 0, tzinfo=UTC)
    state = _state(
        before_learned_time,
        preferred_study_minute=19 * 60,
        learned_study_minute=18 * 60,
    )

    assert effective_routine_minute(state) == 18 * 60
    assert _decide(
        before_learned_time,
        state=state,
        device=_device(before_learned_time),
    ).should_send


def test_eligible_copy_ids_distinguishes_never_computed_empty_and_non_empty():
    early = _decide(state=_state(reminders_enabled=False))

    assert not early.should_send
    assert early.reason == SuppressionReason.REMINDERS_DISABLED
    assert early.eligible_copy_ids is None

    history = tuple(
        _history(
            NOW,
            NotificationKind.HABIT_CUE,
            template.copy_id,
            sent_at=NOW - timedelta(days=1, minutes=index),
            day=date(2026, 8, 30),
        )
        for index, template in enumerate(
            templates_for(NotificationKind.HABIT_CUE, AudiencePolicy.TEEN, "fixed")
        )
    )
    cooled = _decide(state=_state(audience_policy=AudiencePolicy.TEEN), history=history)

    assert cooled.reason == SuppressionReason.TEMPLATE_COOLDOWN
    assert cooled.eligible_copy_ids == ()

    success = _decide()

    assert success.should_send
    assert success.eligible_copy_ids is not None
    assert len(success.eligible_copy_ids) > 0
    assert list(success.eligible_copy_ids) == sorted(success.eligible_copy_ids)
    assert success.copy_id in success.eligible_copy_ids
