from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .constants import AudiencePolicy, NotificationKind


@dataclass(frozen=True)
class NotificationTemplate:
    copy_id: str
    kind: NotificationKind
    audiences: frozenset[AudiencePolicy]
    variants: frozenset[str]
    tone: str
    title: str
    body: str
    required_fields: frozenset[str] = frozenset()
    cooldown_days: int = 7
    version: int = 1
    title_limit: int = 65
    body_limit: int = 180


_SAFE_AUDIENCES = frozenset({AudiencePolicy.TEEN, AudiencePolicy.UNKNOWN})
_ADULT = frozenset({AudiencePolicy.ADULT})

TEMPLATES = (

    # ============================================================
    # NEXT_DAY_COMMITMENT — day 2 after first session
    # ============================================================

    # -- safe (all audiences), variants narrowed to fixed/control so
    #    treatment-assigned adults route to the new coach pool below.
    NotificationTemplate(
        "next_day_safe_1", NotificationKind.NEXT_DAY_COMMITMENT,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Ready for day two?",
        "One short review keeps your study habit moving.",
    ),
    NotificationTemplate(
        "next_day_safe_2", NotificationKind.NEXT_DAY_COMMITMENT,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Two days in a row",
        "A quick review today turns yesterday into a habit.",
    ),
    NotificationTemplate(
        "next_day_safe_3", NotificationKind.NEXT_DAY_COMMITMENT,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "You already started",
        "Finishing day two is easier than starting day one was.",
    ),
    NotificationTemplate(
        "next_day_safe_4", NotificationKind.NEXT_DAY_COMMITMENT,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Come back for round two",
        "Pick up where you left off — it only takes a minute.",
    ),
    NotificationTemplate(
        "next_day_safe_5", NotificationKind.NEXT_DAY_COMMITMENT,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Habits start on day two",
        "Day one was the hard part. Today just keeps it going.",
    ),
    NotificationTemplate(
        "next_day_safe_6", NotificationKind.NEXT_DAY_COMMITMENT,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Your queue is ready",
        "A short session today is how yesterday becomes a routine.",
    ),

    # -- new adult_coach pool (didn't exist before)
    NotificationTemplate(
        "next_day_coach_1", NotificationKind.NEXT_DAY_COMMITMENT,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Day one doesn't count alone",
        "Day two is what makes it real. Two minutes, that's it.",
    ),
    NotificationTemplate(
        "next_day_coach_2", NotificationKind.NEXT_DAY_COMMITMENT,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Don't stop at one day",
        "One session means nothing without a second. Go now.",
    ),
    NotificationTemplate(
        "next_day_coach_3", NotificationKind.NEXT_DAY_COMMITMENT,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "You showed up once",
        "Show up twice and it starts being a habit. Open the queue.",
    ),
    NotificationTemplate(
        "next_day_coach_4", NotificationKind.NEXT_DAY_COMMITMENT,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Prove yesterday wasn't a fluke",
        "One more session today is all it takes.",
    ),
    NotificationTemplate(
        "next_day_coach_5", NotificationKind.NEXT_DAY_COMMITMENT,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "This is the part people skip",
        "Day two is where most people quit. Don't let this be that day.",
    ),
    NotificationTemplate(
        "next_day_coach_6", NotificationKind.NEXT_DAY_COMMITMENT,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Lock in day two",
        "Yesterday was the easy part. Today's the one that counts.",
    ),

    # ============================================================
    # HABIT_CUE — cards ready, not studied yet
    # ============================================================

    NotificationTemplate(
        "habit_safe_1", NotificationKind.HABIT_CUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Your review is ready",
        "A short session now keeps today's progress on track.",
    ),
    NotificationTemplate(
        "habit_safe_2", NotificationKind.HABIT_CUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "A few minutes, that's all",
        "Open your queue and see how far a short session gets you.",
    ),
    NotificationTemplate(
        "habit_safe_3", NotificationKind.HABIT_CUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Still time today",
        "A quick review now keeps your day on track.",
    ),
    NotificationTemplate(
        "habit_safe_4", NotificationKind.HABIT_CUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Pick up where you left off",
        "Your cards are exactly where you left them, ready when you are.",
    ),
    NotificationTemplate(
        "habit_safe_5", NotificationKind.HABIT_CUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "One small step today",
        "Review a few cards — you can stop whenever feels right.",
    ),
    NotificationTemplate(
        "habit_safe_6", NotificationKind.HABIT_CUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Today's review is up to you",
        "No pressure — just an open queue whenever you have a minute.",
    ),

    NotificationTemplate(
        "habit_coach_1", NotificationKind.HABIT_CUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Your cards are waiting",
        "You know the deal. Two minutes. Start the review.",
    ),
    NotificationTemplate(
        "habit_coach_2", NotificationKind.HABIT_CUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Still not done today",
        "Your queue isn't going to clear itself. Two minutes.",
    ),
    NotificationTemplate(
        "habit_coach_3", NotificationKind.HABIT_CUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "{streak} days deep — don't stall now",
        "One review keeps today counted. Go.",
        frozenset({"streak"}),
    ),
    NotificationTemplate(
        "habit_coach_4", NotificationKind.HABIT_CUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Open it. Start it. Done.",
        "One card is enough to get moving. The rest is easy after that.",
    ),
    NotificationTemplate(
        "habit_coach_5", NotificationKind.HABIT_CUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "You said you'd do this daily",
        "Today's still open. One review, that's all it takes.",
    ),
    NotificationTemplate(
        "habit_coach_6", NotificationKind.HABIT_CUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Two minutes, full stop",
        "No excuses today. Open it, do one card, decide from there.",
    ),

    # ============================================================
    # STREAK_RESCUE — streak 3+, day ending (real 4 AM reset deadline)
    # ============================================================

    NotificationTemplate(
        "rescue_safe_1", NotificationKind.STREAK_RESCUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Keep your {streak}-day streak going",
        "One short review keeps today's progress moving.",
        frozenset({"streak"}), cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_safe_2", NotificationKind.STREAK_RESCUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Your study streak is still active",
        "One review before the day ends keeps it going.",
        cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_safe_3", NotificationKind.STREAK_RESCUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "There is still time to review",
        "One short review keeps your current study streak active.",
        cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_safe_4", NotificationKind.STREAK_RESCUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Keep today's streak active",
        "Review one card before the study day ends to continue it.",
        cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_safe_5", NotificationKind.STREAK_RESCUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Almost the end of the day",
        "One quick review keeps your {streak}-day streak alive.",
        frozenset({"streak"}), cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_safe_6", NotificationKind.STREAK_RESCUE,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Don't let today slip by",
        "A short review now keeps your streak exactly where it is.",
        cooldown_days=14,
    ),

    NotificationTemplate(
        "rescue_coach_1", NotificationKind.STREAK_RESCUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Your {streak}-day streak is at risk",
        "It resets at 4 AM. One review keeps it alive.",
        frozenset({"streak"}), cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_coach_2", NotificationKind.STREAK_RESCUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Your streak resets at 4 AM",
        "One review is enough to save today's streak.",
        cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_coach_3", NotificationKind.STREAK_RESCUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "There is still time for your streak",
        "One review before 4 AM keeps the current streak active.",
        cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_coach_4", NotificationKind.STREAK_RESCUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Protect today's study streak",
        "Open the queue and complete one review before the day resets.",
        cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_coach_5", NotificationKind.STREAK_RESCUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "{streak} days on the line",
        "4 AM is the cutoff. One review, streak saved.",
        frozenset({"streak"}), cooldown_days=14,
    ),
    NotificationTemplate(
        "rescue_coach_6", NotificationKind.STREAK_RESCUE,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Don't lose {streak} days over one review",
        "The clock resets at 4 AM. You have time — use it.",
        frozenset({"streak"}), cooldown_days=14,
    ),

    # ============================================================
    # COMEBACK — stopped 2 / 7 / 21 days. Deliberately low-pressure
    # even in "aggressive" tone — see file header.
    # ============================================================

    NotificationTemplate(
        "comeback_safe_1", NotificationKind.COMEBACK,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Start small again",
        "One short review is enough to restart your routine.",
    ),
    NotificationTemplate(
        "comeback_safe_2", NotificationKind.COMEBACK,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "A fresh start can be small",
        "Open a quick review whenever you're ready to return.",
    ),
    NotificationTemplate(
        "comeback_safe_3", NotificationKind.COMEBACK,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Whenever you're ready",
        "No pressure — one short review is a fine way to restart.",
    ),
    NotificationTemplate(
        "comeback_safe_4", NotificationKind.COMEBACK,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Pick up anytime",
        "Your progress is saved. A short review is all it takes to continue.",
    ),
    NotificationTemplate(
        "comeback_safe_5", NotificationKind.COMEBACK,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "A quiet nudge to return",
        "One small review, whenever it suits you.",
    ),
    NotificationTemplate(
        "comeback_safe_6", NotificationKind.COMEBACK,
        frozenset({*_SAFE_AUDIENCES, AudiencePolicy.ADULT}),
        frozenset({"fixed", "control"}), "neutral",
        "Restarting is easier than it seems",
        "One card is enough to get going again.",
    ),

    NotificationTemplate(
        "comeback_coach_1", NotificationKind.COMEBACK,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Your review queue is still here",
        "Two minutes is enough to get moving again.",
    ),
    NotificationTemplate(
        "comeback_coach_2", NotificationKind.COMEBACK,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Ready to return?",
        "Start with one review and rebuild from there.",
    ),
    NotificationTemplate(
        "comeback_coach_3", NotificationKind.COMEBACK,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Pick this back up",
        "One review is enough to restart. No catching up required.",
    ),
    NotificationTemplate(
        "comeback_coach_4", NotificationKind.COMEBACK,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Everything's saved, nothing's lost",
        "Jump back in with one quick review.",
    ),
    NotificationTemplate(
        "comeback_coach_5", NotificationKind.COMEBACK,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Restart, don't rebuild",
        "One review gets you moving again — that's the whole ask.",
    ),
    NotificationTemplate(
        "comeback_coach_6", NotificationKind.COMEBACK,
        _ADULT, frozenset({"treatment"}), "aggressive",
        "Back whenever you are",
        "One short review is all it takes to pick up again.",
    ),
)

TEMPLATES_BY_ID = {template.copy_id: template for template in TEMPLATES}


def templates_for(
    kind: NotificationKind,
    audience: AudiencePolicy,
    variant: str,
) -> tuple[NotificationTemplate, ...]:
    return tuple(
        template
        for template in TEMPLATES
        if template.kind == kind
        and audience in template.audiences
        and variant in template.variants
    )


def select_stable_template(
    templates: tuple[NotificationTemplate, ...], stable_key: str
) -> NotificationTemplate:
    if not templates:
        raise ValueError("at least one template is required")
    ordered = sorted(templates, key=lambda item: item.copy_id)
    index = int.from_bytes(sha256(stable_key.encode("utf-8")).digest()[:8], "big")
    return ordered[index % len(ordered)]


def render_template(copy_id: str, fields: dict[str, str | int]) -> tuple[str, str]:
    template = TEMPLATES_BY_ID[copy_id]
    if set(fields) != set(template.required_fields):
        raise ValueError("template interpolation fields do not match the registry")
    title = template.title.format_map(fields)
    body = template.body.format_map(fields)
    if len(title) > template.title_limit or len(body) > template.body_limit:
        raise ValueError("rendered notification exceeds its copy limit")
    return title, body