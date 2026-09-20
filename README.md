# Reviso Server

FastAPI backend for Reviso. It provides Firebase-authenticated AI, credits,
AdMob verification, account deletion, and server-driven Firebase Cloud
Messaging (FCM) notification APIs.

## Setup

Create a Python 3.11 environment and install development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and provide the required values. Do not commit
`.env`.

Run the web API:

```powershell
uvicorn main:app --reload
```

Run the tests:

```powershell
python -m pytest
```

## Notification API

All notification endpoints require a Firebase bearer token.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/notifications/devices/register` | POST | Register or refresh an opted-in Android installation and FCM token |
| `/v1/notifications/devices/unregister` | POST | Disable an installation and remove its token |
| `/v1/notifications/preferences` | POST | Synchronize reminder state, timezone, and automatic delivery policy |
| `/v1/notifications/state` | POST | Synchronize privacy-minimized study aggregates |
| `/v1/notifications/events:batch` | POST | Ingest retry-safe notification outcome events |

The dedicated worker claims leased Firestore jobs, re-evaluates current state,
sends eligible FCM messages, records controlled suppression reasons, retries
transient failures, and removes invalid tokens.

Routine reminders use 7 PM local time until three qualified study days are
available. The server then learns a weighted routine time from the latest 14
study days and keeps delivery inside the applicable age-safe window. The API
retains its timing fields for compatibility with older app versions.

## Notification Configuration

Generate a stable 32-byte experiment salt:

```powershell
$bytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$salt = ([BitConverter]::ToString($bytes) -replace '-', '').ToLowerInvariant()
$rng.Dispose()
$salt
```

Set the output as `NOTIFICATION_EXPERIMENT_SALT`. Do not change it after
deployment because it controls deterministic adult experiment assignments.

Optional worker settings:

```text
NOTIFICATION_WORKER_POLL_SECONDS=60
NOTIFICATION_WORKER_BATCH_SIZE=25
NOTIFICATION_JOB_LEASE_SECONDS=120
```

## Railway Deployment

Deploy two services from this repository and branch:

| Service | Start command | Public domain |
|---|---|---|
| Web API | `uvicorn main:app --host 0.0.0.0 --port $PORT` | Required |
| Notification worker | `python -m notifications.worker` | Not required |

The worker requires `FIREBASE_SERVICE_ACCOUNT_JSON` and
`NOTIFICATION_EXPERIMENT_SALT`. Start with one worker replica and an
always-restart policy. A successful worker updates
`notification_worker_status/current.last_success_at` after each poll.

Deploy `firestore.rules` and `firestore.indexes.json` from the companion Reviso
Flutter repository before starting the worker:

```powershell
firebase deploy --only "firestore:rules,firestore:indexes" --project "reviso-edu-flashcard"
```

Wait for every Firestore index to report **Enabled** before testing delivery.

## Backward Compatibility

The notification endpoints are additive. Existing API routes and contracts are
retained, and older app versions do not register with the notification system,
so they remain opted out.
