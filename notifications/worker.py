from __future__ import annotations

import json
import logging
import os
import socket
import time
from datetime import timedelta
from uuid import uuid4

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore as fb_fs

from .constants import JobStatus, SendStatus, SuppressionReason
from .decision_engine import DecisionContext, decide
from .models import SendHistory
from .repository import NotificationRepository
from .sender import (
    FCMSender,
    OutboundNotification,
    PermanentTokenError,
    TransientSendError,
)
from .templates import TEMPLATES_BY_ID, render_template

logger = logging.getLogger("reviso.notifications.worker")


class NotificationWorker:
    def __init__(
        self,
        repository: NotificationRepository,
        sender: FCMSender,
        worker_id: str,
        batch_size: int = 25,
        lease_seconds: int = 120,
        experiment_salt: str = "reviso-notification-default",
        max_attempts: int = 5,
    ):
        self.repository = repository
        self.sender = sender
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.lease_seconds = lease_seconds
        self.experiment_salt = experiment_salt
        self.max_attempts = max_attempts
        self._last_cleanup_at = None

    def run_once(self) -> int:
        now = self.repository.now()
        if self._last_cleanup_at is None or now - self._last_cleanup_at >= timedelta(
            hours=1
        ):
            self.repository.purge_expired_records(now)
            self._last_cleanup_at = now
        self.repository.reconcile_ignored_sends(now)
        self.repository.scan_due_states(now)
        jobs = self.repository.claim_due_jobs(
            self.worker_id,
            now,
            self.batch_size,
            self.lease_seconds,
        )
        for job in jobs:
            self._process(job)
        self.repository.record_worker_poll(
            self.worker_id, self.repository.now(), len(jobs)
        )
        return len(jobs)

    def _process(self, job) -> None:
        now = self.repository.now()
        context = self.repository.worker_context(job.uid, now)
        if context is None:
            self.repository.finish_job(
                job,
                self.worker_id,
                JobStatus.SUPPRESSED,
                SuppressionReason.STATE_CHANGED.value,
                None,
            )
            return

        decision = decide(
            DecisionContext(
                state=context.state,
                device=context.device,
                now=now,
                job_deadline=job.deadline_at,
                deletion_pending=context.deletion_pending,
                history=context.history,
                existing_send_keys=context.existing_send_keys,
                experiment_salt=self.experiment_salt,
            )
        )
        if not decision.should_send:
            logger.info(
                "notification_suppressed reason=%s audience=%s",
                decision.reason.value if decision.reason else "unknown",
                context.state.audience_policy.value,
            )
            status = (
                JobStatus.EXPIRED
                if decision.reason == SuppressionReason.JOB_EXPIRED
                else JobStatus.SUPPRESSED
            )
            self.repository.finish_job(
                job,
                self.worker_id,
                status,
                decision.reason.value if decision.reason else None,
                context.state,
                context.history,
            )
            return

        device = context.device
        if device is None or device.fcm_token is None:
            self.repository.finish_job(
                job,
                self.worker_id,
                JobStatus.SUPPRESSED,
                SuppressionReason.TOKEN_MISSING.value,
                context.state,
                context.history,
            )
            return

        template = TEMPLATES_BY_ID[decision.copy_id]
        fields = (
            {"streak": context.state.current_streak}
            if "streak" in template.required_fields
            else {}
        )
        title, body = render_template(decision.copy_id, fields)
        send_data = {
            "send_key": decision.send_key,
            "scheduling_day_key": decision.scheduling_day_key,
            "notification_kind": decision.notification_kind.value,
            "copy_id": decision.copy_id,
            "copy_version": decision.copy_version,
            "experiment_id": decision.experiment_id,
            "variant_id": decision.variant_id,
            "audience_policy": context.state.audience_policy.value,
            "tone": decision.tone,
            "comeback_milestone": decision.comeback_milestone,
            "eligible_copy_ids": (
                list(decision.eligible_copy_ids)
                if decision.eligible_copy_ids is not None
                else None
            ),
        }
        reservation = self.repository.reserve_send(
            job, self.worker_id, context.state, device, send_data
        )
        if reservation != "reserved":
            reason = {
                "deleted": SuppressionReason.ACCOUNT_DELETION_PENDING,
                "duplicate": SuppressionReason.DUPLICATE_SEND,
                "device_changed": SuppressionReason.INSTALLATION_DISABLED,
            }.get(reservation, SuppressionReason.STATE_CHANGED)
            self.repository.finish_job(
                job,
                self.worker_id,
                JobStatus.SUPPRESSED,
                reason.value,
                context.state,
                context.history,
            )
            return

        # This is intentionally the last datastore read before the network call.
        if not self.repository.can_send_now(job, self.worker_id, context.state, device):
            self.repository.mark_send_permanent_failure(
                decision.send_key, "pre_send_guard"
            )
            self.repository.finish_job(
                job,
                self.worker_id,
                JobStatus.SUPPRESSED,
                SuppressionReason.STATE_CHANGED.value,
                context.state,
                context.history,
            )
            return

        try:
            self.sender.send(
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
            logger.warning(
                "notification_permanent_failure code=%s audience=%s",
                exc.code,
                context.state.audience_policy.value,
            )
            self.repository.mark_send_permanent_failure(decision.send_key, exc.code)
            self.repository.disable_invalid_device(job.uid, device, exc.code)
            self.repository.finish_job(
                job,
                self.worker_id,
                JobStatus.FAILED,
                exc.code,
                context.state,
                context.history,
            )
            return
        except TransientSendError as exc:
            logger.warning(
                "notification_transient_failure code=%s attempt=%d",
                exc.code,
                job.attempt_count,
            )
            self._retry_or_fail(job, context, decision.send_key, exc.code)
            return
        except Exception:
            logger.exception("Unexpected FCM send failure job_id=%s", job.job_id)
            self._retry_or_fail(job, context, decision.send_key, "unexpected_error")
            return

        accepted_at = self.repository.now()
        logger.info(
            "notification_fcm_accepted kind=%s audience=%s variant=%s",
            decision.notification_kind.value,
            context.state.audience_policy.value,
            decision.variant_id,
        )
        self.repository.mark_send_accepted(decision.send_key, accepted_at)
        accepted = SendHistory(
            send_key=decision.send_key,
            notification_kind=decision.notification_kind,
            copy_id=decision.copy_id,
            scheduling_day_key=decision.scheduling_day_key,
            sent_at=accepted_at,
            send_status=SendStatus.FCM_ACCEPTED,
            tone=decision.tone,
            comeback_milestone=decision.comeback_milestone,
        )
        self.repository.finish_job(
            job,
            self.worker_id,
            JobStatus.COMPLETED,
            None,
            context.state,
            (*context.history, accepted),
        )

    def _retry_or_fail(self, job, context, send_key: str, error_code: str) -> None:
        now = self.repository.now()
        self.repository.mark_send_retrying(send_key, error_code)
        backoff_seconds = min(15 * 60, 30 * (2 ** max(0, job.attempt_count - 1)))
        retry_at = now + timedelta(seconds=backoff_seconds)
        if job.attempt_count >= self.max_attempts or retry_at >= job.deadline_at:
            self.repository.mark_send_permanent_failure(send_key, "retry_exhausted")
            self.repository.finish_job(
                job,
                self.worker_id,
                JobStatus.FAILED,
                "retry_exhausted",
                context.state,
                context.history,
            )
            return
        self.repository.retry_job(job, self.worker_id, error_code, retry_at)


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


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = _initialize_firebase()
    repository = NotificationRepository(fb_fs.client(app=app))
    worker_id = f"{socket.gethostname()}-{uuid4().hex[:10]}"
    experiment_salt = os.getenv("NOTIFICATION_EXPERIMENT_SALT", "").strip()
    if not experiment_salt or experiment_salt.startswith("replace-"):
        raise RuntimeError("NOTIFICATION_EXPERIMENT_SALT must be configured")
    worker = NotificationWorker(
        repository=repository,
        sender=FCMSender(firebase_app=app),
        worker_id=worker_id,
        batch_size=int(os.getenv("NOTIFICATION_WORKER_BATCH_SIZE", "25")),
        lease_seconds=int(os.getenv("NOTIFICATION_JOB_LEASE_SECONDS", "120")),
        experiment_salt=experiment_salt,
    )
    poll_seconds = max(5, int(os.getenv("NOTIFICATION_WORKER_POLL_SECONDS", "60")))
    logger.info("Notification worker started poll_seconds=%d", poll_seconds)
    while True:
        try:
            claimed = worker.run_once()
            if claimed:
                logger.info("Notification worker processed jobs=%d", claimed)
        except Exception:
            logger.exception("Notification worker poll failed")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
