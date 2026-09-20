from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from hmac import new as hmac_new

from .constants import (
    AudiencePolicy,
    COMEBACK_MILESTONES,
    DEVICE_STALE_SECONDS,
    EXPERIMENT_ID,
    MAX_RESCUES_PER_SEVEN_DAYS,
    MAX_TOTAL_SENDS_PER_SEVEN_DAYS,
    MINIMUM_SPACING_SECONDS,
    NotificationKind,
    ROUTINE_KINDS,
    STATE_FULL_FRESHNESS_SECONDS,
    STATE_GENERIC_FRESHNESS_SECONDS,
    SendStatus,
    SuppressionReason,
)
from .models import DeviceRecord, NotificationState, SendHistory
from .scheduling import (
    as_utc,
    effective_delivery_window,
    effective_routine_minute,
    local_datetime_for_scheduling_day,
    scheduling_day_key,
    within_delivery_window,
)
from .templates import NotificationTemplate, select_stable_template, templates_for


@dataclass(frozen=True)
class DecisionContext:
    state: NotificationState
    device: DeviceRecord | None
    now: datetime
    job_deadline: datetime
    deletion_pending: bool = False
    history: tuple[SendHistory, ...] = ()
    existing_send_keys: frozenset[str] = frozenset()
    experiment_salt: str = "reviso-notification-default"


@dataclass(frozen=True)
class Decision:
    should_send: bool
    reason: SuppressionReason | None = None
    notification_kind: NotificationKind | None = None
    copy_id: str | None = None
    copy_version: int | None = None
    tone: str | None = None
    experiment_id: str = "none"
    variant_id: str = "fixed"
    send_key: str | None = None
    scheduling_day_key: str | None = None
    comeback_milestone: int | None = None
    eligible_copy_ids: tuple[str, ...] | None = None


def stable_experiment_variant(uid: str, experiment_id: str, salt: str) -> str:
    digest = hmac_new(
        salt.encode("utf-8"),
        f"{uid}:{experiment_id}".encode("utf-8"),
        "sha256",
    ).digest()
    return "control" if digest[0] < 128 else "treatment"


def deterministic_send_key(
    uid: str,
    day_key: str,
    kind: NotificationKind,
    comeback_milestone: int | None = None,
) -> str:
    source = f"v1:{uid}:{day_key}:{kind.value}:{comeback_milestone or 0}"
    return sha256(source.encode("utf-8")).hexdigest()


def _suppressed(reason: SuppressionReason) -> Decision:
    return Decision(should_send=False, reason=reason)


def _accepted_history(context: DecisionContext) -> tuple[SendHistory, ...]:
    return tuple(
        item for item in context.history if item.send_status == SendStatus.FCM_ACCEPTED
    )


def _choose_template(
    context: DecisionContext,
    kind: NotificationKind,
    audience: AudiencePolicy,
    variant: str,
) -> tuple[NotificationTemplate | None, tuple[str, ...]]:
    copy_variant = variant
    accepted = _accepted_history(context)
    if audience == AudiencePolicy.ADULT and variant == "treatment":
        if context.state.consecutive_ignored_count >= 2:
            copy_variant = "control"
        elif accepted and accepted[-1].tone == "aggressive":
            copy_variant = "control"

    now = as_utc(context.now)
    eligible = []
    for template in templates_for(kind, audience, copy_variant):
        last_use = max(
            (item.sent_at for item in accepted if item.copy_id == template.copy_id),
            default=None,
        )
        if last_use is None or now - as_utc(last_use) >= timedelta(
            days=template.cooldown_days
        ):
            eligible.append(template)
    eligible_copy_ids = tuple(sorted(template.copy_id for template in eligible))
    if not eligible:
        return None, eligible_copy_ids
    day_key = scheduling_day_key(now, context.state.iana_timezone).isoformat()
    return (
        select_stable_template(
            tuple(eligible),
            f"{context.state.uid}:{kind.value}:{day_key}:{variant}",
        ),
        eligible_copy_ids,
    )


def _routine_due(state: NotificationState, now: datetime) -> bool:
    day = scheduling_day_key(now, state.iana_timezone)
    preferred = local_datetime_for_scheduling_day(
        day, effective_routine_minute(state), state.iana_timezone
    )
    return as_utc(now) >= preferred


def _rescue_due(state: NotificationState, now: datetime) -> bool:
    day = scheduling_day_key(now, state.iana_timezone)
    _, end = effective_delivery_window(state)
    rescue = local_datetime_for_scheduling_day(
        day, max(0, end - 60), state.iana_timezone
    )
    return as_utc(now) >= rescue


def _inactivity_days(state: NotificationState, current_day) -> int | None:
    if state.last_qualified_study_day is None:
        return None
    return max(0, (current_day - state.last_qualified_study_day).days)


def _current_comeback_milestone(inactivity_days: int | None) -> int | None:
    if inactivity_days is None:
        return None
    reached = [day for day in COMEBACK_MILESTONES if inactivity_days >= day]
    return max(reached, default=None)


def decide(context: DecisionContext) -> Decision:
    now = as_utc(context.now)
    deadline = as_utc(context.job_deadline)
    state = context.state
    device = context.device

    if not state.reminders_enabled:
        return _suppressed(SuppressionReason.REMINDERS_DISABLED)
    if context.deletion_pending:
        return _suppressed(SuppressionReason.ACCOUNT_DELETION_PENDING)
    if now > deadline:
        return _suppressed(SuppressionReason.JOB_EXPIRED)
    if device is None or not device.enabled:
        return _suppressed(SuppressionReason.INSTALLATION_DISABLED)
    if not device.fcm_token:
        return _suppressed(SuppressionReason.TOKEN_MISSING)
    if device.uid != state.uid:
        return _suppressed(SuppressionReason.UID_MISMATCH)
    if now - as_utc(device.last_seen_at) > timedelta(seconds=DEVICE_STALE_SECONDS):
        return _suppressed(SuppressionReason.INSTALLATION_STALE)
    if not within_delivery_window(state, now):
        return _suppressed(SuppressionReason.OUTSIDE_QUIET_HOURS)
    if state.app_active_until is not None and now <= as_utc(state.app_active_until):
        return _suppressed(SuppressionReason.APP_ACTIVE)

    current_day = scheduling_day_key(now, state.iana_timezone)
    event_reviews_today = (
        state.review_event_balances.get(current_day.isoformat(), 0) > 0
    )
    studied_today = (
        (state.scheduling_day_key == current_day and state.cards_studied_today > 0)
        or state.last_qualified_study_day == current_day
        or event_reviews_today
    )
    if studied_today:
        if state.cards_studied_today >= state.daily_goal:
            return _suppressed(SuppressionReason.GOAL_ALREADY_MET)
        return _suppressed(SuppressionReason.ALREADY_STUDIED)
    if state.actionable_queue_bucket.value == "none":
        return _suppressed(SuppressionReason.QUEUE_EMPTY)

    accepted = _accepted_history(context)
    rolling_start = now - timedelta(days=7)
    recent = tuple(item for item in accepted if as_utc(item.sent_at) >= rolling_start)
    if len(recent) >= MAX_TOTAL_SENDS_PER_SEVEN_DAYS:
        return _suppressed(SuppressionReason.WEEKLY_BUDGET_EXHAUSTED)
    if accepted:
        most_recent = max(as_utc(item.sent_at) for item in accepted)
        if now - most_recent < timedelta(seconds=MINIMUM_SPACING_SECONDS):
            return _suppressed(SuppressionReason.MINIMUM_SPACING)

    state_age = max(timedelta(0), now - as_utc(state.updated_at))
    inactivity = _inactivity_days(state, current_day)
    routine_today = tuple(
        item
        for item in accepted
        if item.scheduling_day_key == current_day
        and item.notification_kind in ROUTINE_KINDS
    )
    rescues_today = tuple(
        item
        for item in accepted
        if item.scheduling_day_key == current_day
        and item.notification_kind == NotificationKind.STREAK_RESCUE
    )
    rescue_week = tuple(
        item
        for item in recent
        if item.notification_kind == NotificationKind.STREAK_RESCUE
    )

    candidate: NotificationKind | None = None
    milestone: int | None = None
    fallback_reason = SuppressionReason.NO_ELIGIBLE_CANDIDATE

    if state.current_streak >= 3 and _rescue_due(state, now):
        if state_age > timedelta(seconds=STATE_FULL_FRESHNESS_SECONDS):
            fallback_reason = SuppressionReason.STATE_STALE
        elif rescues_today:
            fallback_reason = SuppressionReason.DAILY_BUDGET_EXHAUSTED
        elif len(rescue_week) >= MAX_RESCUES_PER_SEVEN_DAYS:
            fallback_reason = SuppressionReason.RESCUE_BUDGET_EXHAUSTED
        else:
            routine_sent_long_enough_ago = any(
                now - as_utc(item.sent_at) >= timedelta(seconds=MINIMUM_SPACING_SECONDS)
                for item in routine_today
            )
            _, delivery_end = effective_delivery_window(state)
            selected_time_too_late = (
                effective_routine_minute(state)
                > delivery_end - MINIMUM_SPACING_SECONDS // 60
            )
            if routine_sent_long_enough_ago or (
                not routine_today and selected_time_too_late
            ):
                candidate = NotificationKind.STREAK_RESCUE

    if candidate is None and not routine_today and _routine_due(state, now):
        if state_age <= timedelta(seconds=STATE_FULL_FRESHNESS_SECONDS):
            if (
                state.total_qualified_study_days == 1
                and state.last_qualified_study_day == current_day - timedelta(days=1)
            ):
                candidate = NotificationKind.NEXT_DAY_COMMITMENT

        if candidate is None and state_age <= timedelta(
            seconds=STATE_GENERIC_FRESHNESS_SECONDS
        ):
            if inactivity is not None and inactivity <= 1:
                if state.consecutive_ignored_count >= 3:
                    fallback_reason = SuppressionReason.IGNORED_BACKOFF
                elif state.total_qualified_study_days > 0:
                    candidate = NotificationKind.HABIT_CUE

        if candidate is None:
            current_milestone = _current_comeback_milestone(inactivity)
            already_sent = {
                item.comeback_milestone
                for item in accepted
                if item.notification_kind == NotificationKind.COMEBACK
            }
            if current_milestone is not None and current_milestone not in already_sent:
                candidate = NotificationKind.COMEBACK
                milestone = current_milestone
            elif state_age > timedelta(seconds=STATE_GENERIC_FRESHNESS_SECONDS):
                fallback_reason = SuppressionReason.STATE_STALE
    elif (
        candidate is None
        and routine_today
        and fallback_reason == SuppressionReason.NO_ELIGIBLE_CANDIDATE
    ):
        fallback_reason = SuppressionReason.DAILY_BUDGET_EXHAUSTED

    if candidate is None:
        if not _routine_due(state, now) and not _rescue_due(state, now):
            return _suppressed(SuppressionReason.NOT_DUE)
        return _suppressed(fallback_reason)

    audience = state.audience_policy
    if audience == AudiencePolicy.ADULT:
        experiment_id = EXPERIMENT_ID
        variant = stable_experiment_variant(
            state.uid, experiment_id, context.experiment_salt
        )
    else:
        experiment_id = "none"
        variant = "fixed"

    template, eligible_copy_ids = _choose_template(context, candidate, audience, variant)
    if template is None:
        return Decision(
            should_send=False,
            reason=SuppressionReason.TEMPLATE_COOLDOWN,
            eligible_copy_ids=eligible_copy_ids,
        )

    day_key = current_day.isoformat()
    send_key = deterministic_send_key(state.uid, day_key, candidate, milestone)
    if send_key in context.existing_send_keys:
        return Decision(
            should_send=False,
            reason=SuppressionReason.DUPLICATE_SEND,
            eligible_copy_ids=eligible_copy_ids,
        )
    return Decision(
        should_send=True,
        notification_kind=candidate,
        copy_id=template.copy_id,
        copy_version=template.version,
        tone=template.tone,
        experiment_id=experiment_id,
        variant_id=variant,
        send_key=send_key,
        scheduling_day_key=day_key,
        comeback_milestone=milestone,
        eligible_copy_ids=eligible_copy_ids,
    )
