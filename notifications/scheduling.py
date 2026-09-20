from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

from .constants import (
    ADULT_EARLIEST_MINUTE,
    ADULT_LATEST_MINUTE,
    AudiencePolicy,
    MINIMUM_SPACING_SECONDS,
    ROUTINE_KINDS,
    SCHEDULING_DAY_BOUNDARY_HOUR,
    TEEN_EARLIEST_MINUTE,
    TEEN_LATEST_MINUTE,
)
from .models import NotificationState, SendHistory

UTC = timezone.utc


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)


def scheduling_day_key(now: datetime, iana_timezone: str) -> date:
    local = as_utc(now).astimezone(ZoneInfo(iana_timezone))
    return (local - timedelta(hours=SCHEDULING_DAY_BOUNDARY_HOUR)).date()


def local_minute(now: datetime, iana_timezone: str) -> int:
    local = as_utc(now).astimezone(ZoneInfo(iana_timezone))
    return local.hour * 60 + local.minute


def effective_delivery_window(state: NotificationState) -> tuple[int, int]:
    start = state.quiet_hours_start_minute
    end = state.quiet_hours_end_minute
    if state.audience_policy in {AudiencePolicy.TEEN, AudiencePolicy.UNKNOWN}:
        return max(start, TEEN_EARLIEST_MINUTE), min(end, TEEN_LATEST_MINUTE)
    return max(start, 0), min(end, 1439)


def within_delivery_window(state: NotificationState, now: datetime) -> bool:
    start, end = effective_delivery_window(state)
    minute = local_minute(now, state.iana_timezone)
    return start <= minute <= end


def _calendar_date_for_scheduling_day(day: date, minute: int) -> date:
    if minute < SCHEDULING_DAY_BOUNDARY_HOUR * 60:
        return day + timedelta(days=1)
    return day


def local_datetime_for_scheduling_day(
    day: date,
    minute: int,
    iana_timezone: str,
) -> datetime:
    """Resolve a wall time deterministically, advancing through DST gaps."""
    calendar_day = _calendar_date_for_scheduling_day(day, minute)
    naive = datetime.combine(
        calendar_day,
        time(hour=minute // 60, minute=minute % 60),
    )
    zone = ZoneInfo(iana_timezone)
    candidate = naive.replace(tzinfo=zone, fold=0)
    round_trip = candidate.astimezone(UTC).astimezone(zone)
    if round_trip.replace(tzinfo=None) != naive:
        candidate = round_trip
    return candidate.astimezone(UTC)


def learned_habit_minute(
    successful_local_minutes: Iterable[int],
    preferred_minute: int,
    quiet_start_minute: int = ADULT_EARLIEST_MINUTE,
    quiet_end_minute: int = ADULT_LATEST_MINUTE,
) -> int | None:
    samples = tuple(successful_local_minutes)
    if len(samples) < 3:
        return None
    if any(sample < 0 or sample > 1439 for sample in samples):
        raise ValueError("study minute must be between 0 and 1439")

    # Increasing integer weights make newer samples more influential while the
    # circular distance keeps times around midnight close to each other.
    weighted = tuple((sample, index + 1) for index, sample in enumerate(samples))

    def cost(candidate: int) -> int:
        return sum(
            weight * min(abs(candidate - sample), 1440 - abs(candidate - sample))
            for sample, weight in weighted
        )

    center = min(samples, key=lambda candidate: (cost(candidate), candidate))
    return max(quiet_start_minute, min(quiet_end_minute, center))


def effective_routine_minute(state: NotificationState) -> int:
    start, end = effective_delivery_window(state)
    selected = (
        state.learned_study_minute
        if state.learned_study_minute is not None
        else state.preferred_study_minute
    )
    return max(start, min(end, selected))


def _routine_sent_today(history: Iterable[SendHistory], current_day: date) -> bool:
    return any(
        item.scheduling_day_key == current_day
        and item.notification_kind in ROUTINE_KINDS
        for item in history
    )


def next_evaluation_at(
    state: NotificationState,
    now: datetime,
    history: Iterable[SendHistory] = (),
) -> datetime | None:
    if not state.reminders_enabled:
        return None

    now = as_utc(now)
    current_day = scheduling_day_key(now, state.iana_timezone)
    history = tuple(history)
    studied = (
        state.scheduling_day_key == current_day and state.cards_studied_today > 0
    ) or state.last_qualified_study_day == current_day
    target_day = current_day + timedelta(days=1) if studied else current_day
    routine_minute = effective_routine_minute(state)
    preferred = local_datetime_for_scheduling_day(
        target_day, routine_minute, state.iana_timezone
    )
    if studied:
        return preferred

    _, delivery_end = effective_delivery_window(state)
    rescue_minute = max(0, delivery_end - 60)
    rescue = local_datetime_for_scheduling_day(
        current_day, rescue_minute, state.iana_timezone
    )
    routine_sent = _routine_sent_today(history, current_day)
    selected_time_too_late = (
        routine_minute > delivery_end - MINIMUM_SPACING_SECONDS // 60
    )
    if (
        state.current_streak >= 3
        and now < rescue
        and (routine_sent or selected_time_too_late)
    ):
        return rescue
    if not routine_sent and now < preferred:
        return preferred
    if state.current_streak >= 3 and routine_sent and now < rescue:
        return rescue
    return local_datetime_for_scheduling_day(
        current_day + timedelta(days=1),
        routine_minute,
        state.iana_timezone,
    )


def evaluation_deadline_at(state: NotificationState, evaluate_at: datetime) -> datetime:
    day = scheduling_day_key(evaluate_at, state.iana_timezone)
    _, end = effective_delivery_window(state)
    return local_datetime_for_scheduling_day(day, end, state.iana_timezone)
