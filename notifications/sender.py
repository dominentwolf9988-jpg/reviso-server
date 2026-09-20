from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from firebase_admin import exceptions as firebase_exceptions
from firebase_admin import messaging

from .constants import FCM_ROUTE, FCM_SCHEMA_VERSION, NotificationKind
from .scheduling import as_utc


class PermanentTokenError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class TransientSendError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OutboundNotification:
    token: str
    notification_id: str
    notification_kind: NotificationKind
    scheduling_day_key: str
    copy_id: str
    experiment_id: str
    variant_id: str
    title: str
    body: str
    sent_at: datetime
    deadline_at: datetime


class FCMSender:
    def __init__(
        self,
        send_function: Callable[..., str] | None = None,
        firebase_app=None,
    ):
        self._send = send_function or messaging.send
        self._firebase_app = firebase_app

    def send(self, outbound: OutboundNotification) -> str:
        sent_at = as_utc(outbound.sent_at)
        deadline = as_utc(outbound.deadline_at)
        ttl = deadline - sent_at
        if ttl.total_seconds() <= 0:
            raise TransientSendError("job_expired")
        # Client currently sends FCM registration tokens via
        # FirebaseMessaging.getToken(). During the FCM migration period
        # `token` also accepts FIDs, while `fid` does not accept
        # registration tokens. Use `token` until the app migrates to
        # FirebaseInstallations.getId(). See
        # https://firebase.google.com/docs/cloud-messaging/send/admin-sdk
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            message = messaging.Message(
                token=outbound.token,
                notification=messaging.Notification(
                    title=outbound.title,
                    body=outbound.body,
                ),
                data={
                    "schema_version": FCM_SCHEMA_VERSION,
                    "notification_id": outbound.notification_id,
                    "notification_kind": outbound.notification_kind.value,
                    "route": FCM_ROUTE,
                    "scheduling_day_key": outbound.scheduling_day_key,
                    "copy_id": outbound.copy_id,
                    "experiment_id": outbound.experiment_id,
                    "variant_id": outbound.variant_id,
                    "sent_at": sent_at.isoformat().replace("+00:00", "Z"),
                },
                android=messaging.AndroidConfig(
                    priority="high",
                    ttl=ttl,
                    collapse_key=(
                        f"{outbound.notification_kind.value}:"
                        f"{outbound.scheduling_day_key}"
                    ),
                    notification=messaging.AndroidNotification(
                        visibility="private"
                    ),
                ),
            )
        try:
            return self._send(message, app=self._firebase_app)
        except messaging.UnregisteredError as exc:
            raise PermanentTokenError("unregistered") from exc
        except messaging.SenderIdMismatchError as exc:
            raise PermanentTokenError("sender_id_mismatch") from exc
        except firebase_exceptions.InvalidArgumentError as exc:
            # The payload is constructed from a closed schema, so an invalid
            # argument from FCM at this point is treated as an invalid token.
            raise PermanentTokenError("invalid_token") from exc
        except (
            messaging.QuotaExceededError,
            messaging.ThirdPartyAuthError,
            firebase_exceptions.UnavailableError,
            firebase_exceptions.InternalError,
            firebase_exceptions.DeadlineExceededError,
            firebase_exceptions.ResourceExhaustedError,
        ) as exc:
            raise TransientSendError(type(exc).__name__) from exc
        except firebase_exceptions.FirebaseError as exc:
            raise TransientSendError(type(exc).__name__) from exc
