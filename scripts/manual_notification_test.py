"""Manual FCM delivery + notification decision pipeline tester (standalone).

ONE-OFF manual tool. Does not run the poll loop.

Subcommands:
  raw   - bypass decision logic, send one FCM data message directly via
          messaging.send() with the correct ``token=`` param. Firestore-free.
  full  - load real WorkerContext via NotificationRepository.worker_context(),
          run real decision_engine.decide(), and optionally execute the real
          worker send pipeline (reserve_send -> FCMSender -> mark_* ->
          finish_job) for one uid.

Safety:
  * Nothing is sent and nothing is written without --confirm-production.
  * ``raw`` without --confirm-production prints what WOULD be sent and exits.
  * ``full`` without --send prints the Decision only and makes zero writes.
  * ``full --send`` requires --confirm-production or it exits with an error.
  * ``full --send`` on a suppressed Decision exits without writing anything.
  * ``full --force-due`` bypasses ONLY the routine/rescue due-timing gate
    (``now >= 19:00 local``). All other suppressions (studied, queue,
    budgets, app_active, token, etc.) still apply.

Existing-file bug note (read-only finding, not fixed here):
  notifications/sender.py builds ``messaging.Message(fid=outbound.token, ...)``.
  The installed firebase-admin (7.5.0) signature is
  ``Message(..., token=None, ..., fid=None)`` -- ``token`` is the FCM
  registration token, ``fid`` is the Firebase Installation ID. This script's
  ``raw`` mode deliberately uses the correct ``token=`` param. ``full --send``
  still goes through the real FCMSender untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

# Make repo root importable when run as `python scripts/manual_notification_test.py`
# (sys.path[0] is scripts/, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

import firebase_admin  # noqa: E402
from firebase_admin import credentials, firestore as fb_fs  # noqa: E402
from firebase_admin import messaging  # noqa: E402

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Exact Firebase init pattern reused from notifications/worker.py
# (_initialize_firebase) -- do not invent a different pattern.
# ---------------------------------------------------------------------------
def _initialize_firebase():
    try:
        return firebase_admin.get_app()
    except ValueError:
        service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
        if service_account:
            return firebase_admin.initialize_app(
                credentials.Certificate(json.loads(service_account))
            )
        return firebase_admin.initialize_app()


def _load_constants():
    """Return (NotificationKind, FCM_SCHEMA_VERSION, FCM_ROUTE)."""
    from notifications.constants import FCM_ROUTE, FCM_SCHEMA_VERSION, NotificationKind

    return NotificationKind, FCM_SCHEMA_VERSION, FCM_ROUTE


def _truncate_token(token: str) -> str:
    if not token:
        return "<empty>"
    if len(token) <= 16:
        return f"{token[:4]}... (len={len(token)})"
    return f"{token[:12]}...{token[-4:]} (len={len(token)})"


def _utc_now_iso_z(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _manual_notification_id() -> str:
    # 64 lowercase hex chars (sha256 hex digest of a random test string).
    raw = f"manual_test:{uuid4().hex}:{time.time_ns()}:{secrets.token_hex(16)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_day_key(now: datetime, tz_name: str = "UTC") -> str:
    """Today's scheduling_day_key via the real function if importable."""
    try:
        from notifications.scheduling import scheduling_day_key

        return scheduling_day_key(now, tz_name).isoformat()
    except Exception:
        return now.astimezone(UTC).date().isoformat()


# ---------------------------------------------------------------------------
# raw: direct FCM send, no Firestore writes, no decision logic.
# ---------------------------------------------------------------------------
def cmd_raw(args) -> int:
    NotificationKind, FCM_SCHEMA_VERSION, FCM_ROUTE = _load_constants()

    try:
        kind_enum = NotificationKind(args.kind)
    except ValueError:
        print(f"ERROR: --kind must be one of {[e.value for e in NotificationKind]}")
        return 2

    load_dotenv()
    app = _initialize_firebase()
    print(f"[raw] firebase app: {app.name}")

    # Resolve token: explicit --token wins; otherwise query devices for uid.
    token = args.token
    token_source = "explicit --token"
    device_tz = "UTC"
    if not token:
        # Devices are keyed by installation_id, NOT by uid, so query by uid field
        # (same as NotificationRepository.worker_context does).
        from notifications.constants import NOTIFICATION_DEVICES

        db = fb_fs.client(app=app)
        print(f"[raw] looking up {NOTIFICATION_DEVICES} where uid == {args.uid!r} ...")
        try:
            snaps = list(
                db.collection(NOTIFICATION_DEVICES)
                .where("uid", "==", args.uid)
                .limit(100)
                .stream()
            )
        except Exception as exc:
            print(f"[raw] ERROR querying devices: {exc!r}")
            return 1
        print(f"[raw] found {len(snaps)} device doc(s) for uid.")
        candidates = []
        for s in snaps:
            d = s.to_dict() or {}
            tok = d.get("fcm_token")
            print(
                f"  - {NOTIFICATION_DEVICES}/{s.id}: enabled={d.get('enabled')} "
                f"permission={d.get('permission_status')} "
                f"tz={d.get('iana_timezone')} "
                f"token={_truncate_token(tok) if tok else '<none>'}"
            )
            if tok:
                candidates.append((s.id, d))
        if not candidates:
            print(
                "[raw] ERROR: no device with fcm_token found for uid. "
                "Re-run with --token <FCM_TOKEN>."
            )
            return 1
        # Prefer enabled + authorized, most recently seen (mirrors worker_context).
        def _rank(item):
            _doc_id, d = item
            enabled = 1 if d.get("enabled") else 0
            auth = 1 if d.get("permission_status") == "authorized" else 0
            return (enabled, auth, int(d.get("last_seen_at") or 0))

        best_id, best = max(candidates, key=_rank)
        token = best.get("fcm_token")
        device_tz = str(best.get("iana_timezone") or "UTC")
        token_source = f"{NOTIFICATION_DEVICES}/{best_id} (queried, best candidate)"
        print(f"[raw] using token from {token_source}: {_truncate_token(token)}")
    else:
        print(f"[raw] using {token_source}: {_truncate_token(token)}")

    now = datetime.now(UTC)
    day_key = _resolve_day_key(now, device_tz)
    notification_id = _manual_notification_id()
    sent_at = _utc_now_iso_z(now)

    data = {
        "schema_version": FCM_SCHEMA_VERSION,  # "1"
        "notification_id": notification_id,
        "notification_kind": kind_enum.value,
        "route": FCM_ROUTE,  # "global_review"
        "scheduling_day_key": day_key,
        "copy_id": "manual_test",
        "experiment_id": "manual_test",
        "variant_id": "manual_test",
        "sent_at": sent_at,
    }
    title = args.title or "Manual test notification"
    body = args.body or "This is a manual FCM delivery test. Tap should open the app."

    print("")
    print("[raw] exact data payload to send:")
    print(json.dumps(data, indent=2, sort_keys=True))
    print(f"[raw] title={title!r} body={body!r}")
    print(f"[raw] token_source={token_source} token={_truncate_token(token)}")
    print("[raw] SDK param: token= (correct; sender.py currently uses fid= -- see header)")

    if not args.confirm_production:
        print("")
        print(
            "[raw] DRY-RUN: would call messaging.send() with the above payload. "
            "Pass --confirm-production to actually send. "
            "No FCM call made, zero Firestore writes."
        )
        return 0

    # Real send: direct messaging.send(), NOT via FCMSender (bypasses validation).
    message = messaging.Message(
        token=token,
        notification=messaging.Notification(title=title, body=body),
        data=data,
        android=messaging.AndroidConfig(
            priority="high",
            ttl=timedelta(hours=1),
            collapse_key=f"{kind_enum.value}:{day_key}",
            notification=messaging.AndroidNotification(visibility="private"),
        ),
    )
    print("[raw] calling messaging.send() ...")
    try:
        message_id = messaging.send(message, app=app)
    except Exception as exc:
        print(f"[raw] FCM send FAILED: {exc!r}")
        import traceback

        traceback.print_exc()
        return 1
    print(f"[raw] FCM send OK, message ID: {message_id}")
    print("[raw] Firestore writes in this mode: none (FCM-only).")
    return 0


# ---------------------------------------------------------------------------
# full: real decide() + optionally the real worker send pipeline.
# ---------------------------------------------------------------------------
def _print_decision(decision) -> None:
    print("  should_send:", decision.should_send)
    print(
        "  reason:",
        decision.reason.value if decision.reason is not None else None,
    )
    print(
        "  notification_kind:",
        decision.notification_kind.value if decision.notification_kind else None,
    )
    print("  copy_id:", decision.copy_id)
    print("  copy_version:", decision.copy_version)
    print("  tone:", decision.tone)
    print("  experiment_id:", decision.experiment_id)
    print("  variant_id:", decision.variant_id)
    print("  send_key:", decision.send_key)
    print("  scheduling_day_key:", decision.scheduling_day_key)
    print("  comeback_milestone:", decision.comeback_milestone)
    print(
        "  eligible_copy_ids:",
        list(decision.eligible_copy_ids)
        if decision.eligible_copy_ids is not None
        else None,
    )


def cmd_full(args) -> int:
    if args.send and not args.confirm_production:
        print(
            "ERROR: --send requires --confirm-production. "
            "Re-run with --confirm-production to actually write/send, "
            "or drop --send for a read-only decision print."
        )
        return 2
    if getattr(args, "decide_at", None) and args.send:
        print(
            "ERROR: --decide-at is dry-run only and cannot be combined with --send. "
            "Drop --send to print the Decision at the synthetic time (zero writes)."
        )
        return 2

    load_dotenv()
    app = _initialize_firebase()
    print(f"[full] firebase app: {app.name}")

    from firebase_admin import firestore as _fb_fs  # noqa: F401 (ensures init)

    from notifications.constants import JobStatus, SuppressionReason
    from notifications.decision_engine import DecisionContext, decide
    from notifications.repository import NotificationRepository
    from notifications.scheduling import as_utc, evaluation_deadline_at

    repo = NotificationRepository(fb_fs.client(app=app))
    now = repo.now()
    print(f"[full] now (utc): {_utc_now_iso_z(now)}")

    experiment_salt = os.getenv("NOTIFICATION_EXPERIMENT_SALT", "").strip()
    salt_source = "env NOTIFICATION_EXPERIMENT_SALT"
    if not experiment_salt:
        experiment_salt = "reviso-notification-default"
        salt_source = "default (env missing/empty)"
    print(f"[full] experiment_salt source: {salt_source}")

    # Exact repository method -- no hand-constructed fake context.
    context = repo.worker_context(args.uid, now)
    if context is None:
        print(f"[full] no notification state for uid={args.uid!r} (context is None).")
        print("Decision: cannot decide (would be STATE_CHANGED suppression in worker).")
        print("Firestore writes: none.")
        return 0

    state, device = context.state, context.device
    print(f"[full] state: revision={state.state_revision} audience={state.audience_policy.value} "
          f"reminders={state.reminders_enabled} tz={state.iana_timezone} "
          f"streak={state.current_streak} studied_today={state.cards_studied_today} "
          f"queue={state.actionable_queue_bucket.value}")
    if device is None:
        print("[full] device: <none eligible>")
    else:
        print(f"[full] device: installation={device.installation_id} "
              f"enabled={device.enabled} perm={device.permission_status.value} "
              f"token={_truncate_token(device.fcm_token or '')} "
              f"last_seen={device.last_seen_at.isoformat()}")
    print(f"[full] deletion_pending={context.deletion_pending} "
          f"history_items={len(context.history)} existing_send_keys={len(context.existing_send_keys)}")

    # Per plan: dry-run deadline comes from the real evaluation_deadline_at().
    # --skip-quiet-hours shifts ONLY the inputs fed to decide(): decide() takes a
    # single `now` that inseparably drives quiet-hours AND spacing/budget/state-age
    # checks, so the synthetic time affects all of them (see warning below).
    if getattr(args, "skip_quiet_hours", False) and getattr(
        args, "force_due", False
    ):
        print("ERROR: --skip-quiet-hours and --force-due are mutually exclusive.")
        return 2
    if getattr(args, "decide_at", None) and (
        getattr(args, "skip_quiet_hours", False) or getattr(args, "force_due", False)
    ):
        print("ERROR: --decide-at is mutually exclusive with --skip-quiet-hours and --force-due.")
        return 2

    decide_now = now
    decide_deadline = evaluation_deadline_at(state, now)
    if getattr(args, "decide_at", None):
        try:
            custom_now = datetime.fromisoformat(args.decide_at)
        except ValueError:
            print(
                "ERROR: --decide-at must be ISO-8601 with an explicit offset, "
                "e.g. 2026-09-10T13:00:00+05:30."
            )
            return 2
        if custom_now.tzinfo is None:
            print(
                "ERROR: --decide-at must include a UTC offset, "
                "e.g. 2026-09-10T13:00:00+05:30 (naive datetimes rejected)."
            )
            return 2

        print("")
        print("==== CUSTOM DECIDE TIME FOR MANUAL TESTING — NOT REAL WALL-CLOCK TIME ====")
        print(f"[full] real wall-clock now: {_utc_now_iso_z(now)}")
        print(f"[full] custom decide time:  {custom_now.isoformat()}")
        print("[full] WARNING: decide() takes a single `now` for everything, so "
              "state_age / minimum_spacing / weekly-rescue budgets / existing_send_keys "
              "cutoff / app_active_until / device-staleness are ALSO evaluated relative "
              "to the CUSTOM time above, not real wall-clock time. Interpret with that caveat.")
        decide_now = custom_now
        decide_deadline = evaluation_deadline_at(state, decide_now)
    if getattr(args, "force_due", False):
        from notifications.scheduling import (
            effective_routine_minute,
            local_datetime_for_scheduling_day,
            scheduling_day_key,
        )

        print("")
        print("==== ROUTINE-DUE BYPASSED FOR MANUAL TESTING — OTHER GATES INTACT ====")
        routine_minute = effective_routine_minute(state)
        sched_day = scheduling_day_key(now, state.iana_timezone)
        routine_at = local_datetime_for_scheduling_day(
            sched_day, routine_minute, state.iana_timezone
        )
        forced_now = max(as_utc(now), as_utc(routine_at) + timedelta(seconds=60))
        print(f"[full] routine_minute={routine_minute} "
              f"routine_at={routine_at.isoformat()} "
              f"real now={_utc_now_iso_z(now)}")
        print(f"[full] decide now forced to {forced_now.isoformat()} "
              f"(scheduling_day={sched_day.isoformat()}). "
              "Studied/queue/budget/app_active/token checks unchanged.")
        decide_now = forced_now
        decide_deadline = evaluation_deadline_at(state, decide_now)
    if getattr(args, "skip_quiet_hours", False):
        from notifications.scheduling import (
            effective_delivery_window,
            local_datetime_for_scheduling_day,
            scheduling_day_key,
        )

        print("")
        print("==== QUIET HOURS BYPASSED FOR MANUAL TESTING — NOT A REAL DELIVERY WINDOW ====")
        start, end = effective_delivery_window(state)
        syn_minute = min(start + 5, end)
        # Use TOMORROW's scheduling day so the synthetic time sits after this
        # evening's app_active_until grace (today start+5 is always before it and
        # would sticky-fail app_active). Still derived from real functions.
        from datetime import timedelta as _td

        sched_day = scheduling_day_key(now, state.iana_timezone) + _td(days=1)
        synthetic_now = local_datetime_for_scheduling_day(
            sched_day, syn_minute, state.iana_timezone
        )
        print(f"[full] effective_delivery_window(audience={state.audience_policy.value}, "
              f"tz={state.iana_timezone}): start={start} end={end} (minutes)")
        print(f"[full] real wall-clock now:       {_utc_now_iso_z(now)}")
        print(f"[full] synthetic in-window now:   {synthetic_now.isoformat()} "
              f"(scheduling_day={sched_day.isoformat()} minute={syn_minute})")
        print("[full] WARNING: decide() takes a single `now` for everything, so "
              "state_age / minimum_spacing / weekly-rescue budgets / existing_send_keys "
              "cutoff / app_active_until / device-staleness are ALSO evaluated relative "
              "to the SIMULATED time above, not real wall-clock time. Interpret with that caveat.")
        decide_now = synthetic_now
        decide_deadline = evaluation_deadline_at(state, synthetic_now)
    print(f"[full] job_deadline for decide() (evaluation_deadline_at): {decide_deadline.isoformat()}")
    if (
        getattr(args, "skip_quiet_hours", False)
        or getattr(args, "force_due", False)
        or getattr(args, "decide_at", None)
    ):
        print(f"[full] decide inputs: now={decide_now.isoformat()} deadline={decide_deadline.isoformat()}")

    decision = decide(
        DecisionContext(
            state=state,
            device=device,
            now=decide_now,
            job_deadline=decide_deadline,
            deletion_pending=context.deletion_pending,
            history=context.history,
            existing_send_keys=context.existing_send_keys,
            experiment_salt=experiment_salt,
        )
    )
    print("[full] Decision:")
    _print_decision(decision)

    if not decision.should_send:
        print(f"[full] SUPPRESSED ({decision.reason.value if decision.reason else 'unknown'}). "
              "No writes made (suppressions never force-send; use `raw` to bypass).")
        print("Firestore writes: none.")
        return 0

    if not args.send:
        print(f"[full] DRY-RUN: would send kind={decision.notification_kind.value} "
              f"send_key={decision.send_key}. Pass --send --confirm-production to execute. "
              "No writes made.")
        print("Firestore writes: none.")
        return 0

    # --send + eligible + confirmed: execute the exact worker _process() path.
    print("[full] --send confirmed: executing real worker pipeline (reserve -> send -> mark -> finish).")
    worker_id = f"manual-{socket.gethostname()}-{uuid4().hex[:8]}"
    print(f"[full] manual worker_id: {worker_id}")
    artifacts: list[str] = []

    from notifications.constants import NOTIFICATION_JOBS, NOTIFICATION_SENDS, NOTIFICATION_STATES
    from notifications.models import JobRecord, SendHistory
    from notifications.sender import (
        FCMSender,
        OutboundNotification,
        PermanentTokenError,
        TransientSendError,
    )
    from notifications.templates import TEMPLATES_BY_ID, render_template

    if device is None or not device.fcm_token:
        print("[full] device/token missing at send time; worker would suppress TOKEN_MISSING. No writes.")
        return 0

    template = TEMPLATES_BY_ID[decision.copy_id]
    fields = {"streak": state.current_streak} if "streak" in template.required_fields else {}
    title, body = render_template(decision.copy_id, fields)
    print(f"[full] copy: {decision.copy_id} title={title!r} body={body!r}")

    send_data = {
        "send_key": decision.send_key,
        "scheduling_day_key": decision.scheduling_day_key,
        "notification_kind": decision.notification_kind.value,
        "copy_id": decision.copy_id,
        "copy_version": decision.copy_version,
        "experiment_id": decision.experiment_id,
        "variant_id": decision.variant_id,
        "audience_policy": state.audience_policy.value,
        "tone": decision.tone,
        "comeback_milestone": decision.comeback_milestone,
        "eligible_copy_ids": (
            list(decision.eligible_copy_ids)
            if decision.eligible_copy_ids is not None
            else None
        ),
    }

    # reserve_send requires a PROCESSING job owned by us, so create one isolated
    # manual job doc (tracked for later deletion) instead of stealing a real job.
    def _epoch(dt: datetime) -> int:
        return int(as_utc(dt).timestamp())

    manual_job_id = f"manual-{uuid4().hex[:16]}"
    lease_seconds = 120
    max_attempts = 5
    db = repo.db
    # Flag only changes decide() inputs above. The send pipeline below runs on real
    # wall-clock time. can_send_now / FCMSender TTL require deadline >= real now, but
    # decide_deadline (synthetic day end) can already be past when testing after-hours,
    # so extend the manual job deadline to stay future-dated (logged clearly).
    job_deadline_for_job = decide_deadline
    if getattr(args, "skip_quiet_hours", False):
        floor = as_utc(now) + timedelta(hours=2)
        if as_utc(job_deadline_for_job) < floor:
            print(f"[full] bypass: decide_deadline {job_deadline_for_job.isoformat()} is past "
                  f"relative to real now; extending manual job deadline to {floor.isoformat()} "
                  "so reserve/can_send_now/TTL can pass (decide() already ran on synthetic time).")
            job_deadline_for_job = floor
    job_doc = {
        "job_id": manual_job_id,
        "uid": args.uid,
        "status": JobStatus.PROCESSING.value,
        "evaluate_at": _epoch(now),
        "deadline_at": _epoch(job_deadline_for_job),
        "lease_owner": worker_id,
        "lease_until": _epoch(now) + lease_seconds,
        "attempt_count": 1,
        "last_error_code": None,
        "manual_test": True,
        "created_at": _epoch(now),
        "updated_at": _epoch(now),
    }
    db.collection(NOTIFICATION_JOBS).document(manual_job_id).set(job_doc)
    artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (created manual PROCESSING job)")
    job = JobRecord(
        job_id=manual_job_id,
        uid=args.uid,
        status=JobStatus.PROCESSING,
        evaluate_at=now,
        deadline_at=job_deadline_for_job,
        lease_owner=worker_id,
        lease_until=now + timedelta(seconds=lease_seconds),
        attempt_count=1,
    )

    reservation = repo.reserve_send(job, worker_id, state, device, send_data)
    print(f"[full] reserve_send -> {reservation!r}")
    artifacts.append(f"{NOTIFICATION_SENDS}/{decision.send_key} (reserve_send touched)")
    if reservation != "reserved":
        reason = {
            "deleted": SuppressionReason.ACCOUNT_DELETION_PENDING,
            "duplicate": SuppressionReason.DUPLICATE_SEND,
            "device_changed": SuppressionReason.INSTALLATION_DISABLED,
        }.get(reservation, SuppressionReason.STATE_CHANGED)
        repo.finish_job(job, worker_id, JobStatus.SUPPRESSED, reason.value, state, context.history)
        artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (finish_job SUPPRESSED {reason.value})")
        artifacts.append(f"{NOTIFICATION_STATES}/{args.uid} (finish_job may update next_evaluation_at)")
        print(f"[full] reservation failed -> finished SUPPRESSED {reason.value}")
        _print_artifacts(artifacts)
        return 0

    if not repo.can_send_now(job, worker_id, state, device):
        repo.mark_send_permanent_failure(decision.send_key, "pre_send_guard")
        repo.finish_job(job, worker_id, JobStatus.SUPPRESSED,
                        SuppressionReason.STATE_CHANGED.value, state, context.history)
        artifacts.append(f"{NOTIFICATION_SENDS}/{decision.send_key} (mark_send_permanent_failure pre_send_guard)")
        artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (finish_job SUPPRESSED state_changed)")
        print("[full] pre-send guard failed -> SUPPRESSED state_changed")
        _print_artifacts(artifacts)
        return 0

    sender = FCMSender(firebase_app=app)
    try:
        fcm_id = sender.send(
            OutboundNotification(
                token=device.fcm_token,
                notification_id=decision.send_key,
                notification_kind=decision.notification_kind,
                scheduling_day_key=decision.scheduling_day_key,
                copy_id=decision.copy_id,
                experiment_id=decision.experiment_id,
                variant_id=decision.variant_id,
                title=title,
                body=body,
                sent_at=now,
                deadline_at=job.deadline_at,
            )
        )
    except PermanentTokenError as exc:
        print(f"[full] permanent token failure: {exc.code}")
        repo.mark_send_permanent_failure(decision.send_key, exc.code)
        repo.disable_invalid_device(args.uid, device, exc.code)
        repo.finish_job(job, worker_id, JobStatus.FAILED, exc.code, state, context.history)
        artifacts.append(f"{NOTIFICATION_SENDS}/{decision.send_key} (mark_send_permanent_failure {exc.code})")
        artifacts.append(f"notification_devices/{device.installation_id} (disabled, token removed)")
        artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (finish_job FAILED {exc.code})")
        _print_artifacts(artifacts)
        return 1
    except TransientSendError as exc:
        print(f"[full] transient failure: {exc.code}")
        _retry_or_fail(repo, job, worker_id, state, context.history,
                       decision.send_key, exc.code, max_attempts)
        artifacts.append(f"{NOTIFICATION_SENDS}/{decision.send_key} (mark_send_retrying {exc.code} or permanent)")
        artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (retry_job or finish_job FAILED)")
        _print_artifacts(artifacts)
        return 1
    except Exception as exc:  # noqa: BLE001 -- mirror worker's catch-all
        print(f"[full] unexpected send failure: {exc!r}")
        _retry_or_fail(repo, job, worker_id, state, context.history,
                       decision.send_key, "unexpected_error", max_attempts)
        artifacts.append(f"{NOTIFICATION_SENDS}/{decision.send_key} (mark_send_retrying unexpected_error or permanent)")
        artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (retry_job or finish_job FAILED)")
        _print_artifacts(artifacts)
        return 1

    accepted_at = repo.now()
    print(f"[full] FCM accepted: {fcm_id}")
    repo.mark_send_accepted(decision.send_key, accepted_at)
    accepted = SendHistory(
        send_key=decision.send_key,
        notification_kind=decision.notification_kind,
        copy_id=decision.copy_id,
        scheduling_day_key=decision.scheduling_day_key,
        sent_at=accepted_at,
        send_status=repo_send_accepted_status(),
        tone=decision.tone,
        comeback_milestone=decision.comeback_milestone,
    )
    repo.finish_job(job, worker_id, JobStatus.COMPLETED, None, state,
                    (*context.history, accepted))
    artifacts.append(f"{NOTIFICATION_SENDS}/{decision.send_key} (mark_send_accepted)")
    artifacts.append(f"{NOTIFICATION_JOBS}/{manual_job_id} (finish_job COMPLETED)")
    artifacts.append(f"{NOTIFICATION_STATES}/{args.uid} (finish_job may update next_evaluation_at + schedule follow-up job)")
    artifacts.append("(follow-up) query notification_jobs where uid==uid to find any auto-scheduled next job")
    print("[full] COMPLETED.")
    _print_artifacts(artifacts)
    return 0


def repo_send_accepted_status():
    from notifications.constants import SendStatus

    return SendStatus.FCM_ACCEPTED


def _retry_or_fail(repo, job, worker_id, state, history, send_key, error_code, max_attempts) -> None:
    """Mirror NotificationWorker._retry_or_fail."""
    from datetime import timedelta as _td

    from notifications.constants import JobStatus

    now = repo.now()
    repo.mark_send_retrying(send_key, error_code)
    backoff_seconds = min(15 * 60, 30 * (2 ** max(0, job.attempt_count - 1)))
    retry_at = now + _td(seconds=backoff_seconds)
    if job.attempt_count >= max_attempts or retry_at >= job.deadline_at:
        repo.mark_send_permanent_failure(send_key, "retry_exhausted")
        repo.finish_job(job, worker_id, JobStatus.FAILED, "retry_exhausted", state, history)
        print(f"[full] retry exhausted -> FAILED (backoff={backoff_seconds}s)")
        return
    repo.retry_job(job, worker_id, error_code, retry_at)
    print(f"[full] requeued for retry at {retry_at.isoformat()} (backoff={backoff_seconds}s)")


def _print_artifacts(artifacts: list[str]) -> None:
    print("")
    print("Firestore artifacts created/modified this run (delete manually to clean up):")
    if not artifacts:
        print("  (none)")
    for a in artifacts:
        print(f"  - {a}")


def build_parser() -> argparse.ArgumentParser:
    from notifications.constants import NotificationKind as _Kind

    parser = argparse.ArgumentParser(
        description="Manual FCM delivery + notification decision pipeline tester."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_raw = sub.add_parser("raw", help="Direct FCM send, bypass decision logic (FCM-only).")
    p_raw.add_argument("--uid", required=True, help="Firebase UID (used for token lookup if --token omitted).")
    p_raw.add_argument("--token", default=None, help="FCM registration token. If omitted, query notification_devices for uid.")
    p_raw.add_argument("--kind", default="habit_cue",
                       choices=[e.value for e in _Kind],
                       help="Must be a NotificationKind value (client allowlist).")
    p_raw.add_argument("--title", default=None)
    p_raw.add_argument("--body", default=None)
    p_raw.add_argument("--confirm-production", action="store_true",
                       help="Required to actually call FCM. Without it, dry-run only.")
    p_raw.set_defaults(func=cmd_raw)

    p_full = sub.add_parser("full", help="Real decide() + optional real send pipeline for one uid.")
    p_full.add_argument("--uid", required=True)
    p_full.add_argument("--send", action="store_true",
                        help="Without it: print Decision only, zero writes.")
    p_full.add_argument("--confirm-production", action="store_true",
                        help="Required when --send is passed.")
    p_full.add_argument("--skip-quiet-hours", action="store_true",
                         help="Bypass quiet-hours/deadline gate for decide() via a synthetic "
                              "in-window time derived from the real delivery window. "
                              "Dry-run safe without --send (zero writes).")
    p_full.add_argument("--force-due", action="store_true",
                         help="Bypass ONLY the routine-due timing gate "
                              "(now >= routine local time, default 19:00) by "
                              "advancing decide() time to routine+60s on the "
                              "same scheduling day. Keeps studied/queue/budget/"
                              "app_active/token checks intact. "
                              "Dry-run safe without --send (zero writes).")
    p_full.add_argument("--decide-at", default=None, metavar="ISO_DATETIME",
                         help="Custom decide() clock as ISO-8601 with explicit offset, "
                              "e.g. 2026-09-10T13:00:00+05:30. Dry-run only "
                              "(cannot be combined with --send, --skip-quiet-hours, "
                              "or --force-due). Zero writes.")
    p_full.set_defaults(func=cmd_full)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
