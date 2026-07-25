# AI Editing Guide

This document is the handoff for future AI agents and maintainers. Read it before
modifying the application.

## User intent

The application must operate independently of ChatGPT after launch:

- Human signs in and completes CAPTCHA the day before.
- User supplies date, continuous start/end time, and second player.
- Application keeps the authenticated session alive.
- At exactly 7:00 AM Pacific, the application uses direct HTTP GET/POST calls.
- There must be no graphical clicking in the critical path.
- Speed matters, but duplicate reservations and CAPTCHA bypasses are unacceptable.

`src/start.mjs` is the primary user entry point. It sequentially runs manual
login, interactive configuration, read-only preflight, explicit `ARM`
confirmation, and the long-running live API process. Do not hard-code a
reservation date or second player into the application.

`Pickleball Reservation GUI.ps1` is the primary Windows interface launched by
the Desktop shortcut. It writes the same private `config.json` and invokes the
same Node login, preflight, and live API scripts. Keep all booking behavior in
the Node modules; the GUI should remain an orchestration/configuration layer.

While a live process is armed, the GUI must not launch login or a second
preflight because both would contend for the persistent Chrome profile. The
**Check Armed Session** button writes `logs/gui-healthcheck.request.json`; the
existing live Node process consumes it, validates the authenticated balance
endpoint, and emits one `__ARMED_HEALTH__` JSON line. Keep this health check
read-only and within the existing process. Failures emit
`__ARMED_HEALTH_ERROR__`, warn the user, and leave the process armed; transient
network or Cloudflare failures must not disarm an otherwise valid live run.

## Current facility

- Facility: iTennis/iPickle Arcadia
- Facility ID: `549`
- Facility slug: `ipicklearcadia`
- Time zone: `America/Los_Angeles`
- Observed booking window: 10 days
- Observed release time: 7:00 AM
- Observed time step: 1,800 seconds
- Observed minimum party: 2 users

All of these except the slug and ID should be validated from the current
`BookBox` React props during preflight.

## Authentication model

`src/login.mjs` launches ordinary Google Chrome directly. It does not use
Playwright during login. The dedicated profile lives at:

```text
.playbypoint-profile/
```

The user manually solves any CAPTCHA and signs in. `src/api-book.mjs` later
opens that profile with Playwright only to obtain:

- Existing authenticated cookies
- Current `BookBox` React props
- Rails CSRF token
- Same-origin browser-page `fetch()` for the direct API calls

Do not replace the page-based API client in `src/api-book.mjs` with
`context.request`. PlayByPoint's Cloudflare layer returned HTTP 403 to
Playwright's separate request client even with shared cookies. Same-origin
`fetch()` inside the authenticated Chrome page uses Chrome's normal network
context and passed the authenticated preflight. This is still direct GET/POST;
there are no booking-page clicks in the critical path.

Never extract, print, commit, or transmit cookies, passwords, CSRF tokens, card
details, or full user objects.

## Discovered PlayByPoint endpoints

These endpoints were identified from PlayByPoint's served application bundle and
the authenticated booking flow:

```text
GET  /api/facilities/{facility_id}/court_types?kind=reservation
GET  /api/facilities/{facility_id}/available_hours
GET  /api/facilities/{facility_id}/available_courts
GET  /api/courts/{court_id}/price
POST /api/courts/{court_id}/booking_player

GET  /api/guest_users
GET  /api/users/{user_id}/following
GET  /api/find_user
GET  /api/users/{user_id}/balance/{facility_id}
```

The critical path intentionally uses only:

```text
GET available_courts -> POST booking_player
```

Static metadata, player resolution, balance, and CSRF are fetched before release.

## Availability request

The application sends:

```text
GET /api/facilities/549/available_courts
```

Query parameters:

```json
{
  "date": "<target local midnight as Unix seconds>",
  "surface": "<surface value returned by court_types>",
  "start_hour": "<seconds from local midnight>",
  "hour_end": "<seconds from local midnight>",
  "kind": "reservation"
}
```

For 8:00–10:00 PM:

```text
start_hour = 72000
hour_end   = 79200
```

This requests one continuous two-hour interval. Do not turn it into four
independent 30-minute bookings.

## Booking request

Endpoint:

```text
POST /api/courts/{court_id}/booking_player
Content-Type: application/json
X-CSRF-Token: <current token>
X-Requested-With: XMLHttpRequest
```

Payload shape observed in the site bundle:

```json
{
  "reservation": {
    "date": "YYYY-MM-DD",
    "hour_start": 72000,
    "hour_end": 79200,
    "reservation_type": 2,
    "public_game": false,
    "min_ntrp": 1,
    "max_ntrp": 7,
    "kind": "reservation",
    "ntrp_verified": false
  },
  "payment": {
    "method": "card|prepaid|cash|...",
    "payment_intent_id": "",
    "card_details": {},
    "coupon": { "code": "" },
    "booking_package_purchase_id": null,
    "moment": "now|later"
  },
  "user_ids": ["<owner id>", "<second player id>"],
  "user_excluded_ids": [],
  "user_ids_guest_names": {
    "player0": { "name": null },
    "player1": { "name": null }
  },
  "reservation_fees": [],
  "users_fees": [],
  "auto_fill_courts": false,
  "free_fare_players": [],
  "guest_pass_users": [],
  "booking_package_applies_to_user_ids": []
}
```

Success is accepted only when the response contains `location.id` or
`location.slug`. Never infer success solely from HTTP 200.

## Player resolution

The app supports the owner plus exactly one additional player:

1. Exact match against `/api/guest_users`
2. Exact match against `/api/users/{owner}/following`
3. Exact match against `/api/find_user`

Do not select the first fuzzy match. Ambiguous or missing names must fail
preflight and ask the user for the exact display name.

## Timing and keepalive

The application starts immediately when launched. It does not wait until prewarm
before opening the authenticated context.

- Keepalive: authenticated balance GET every configured interval
- Prewarm: reload booking page and refresh CSRF shortly before release
- Release: configured local release epoch plus `releaseOffsetMs`
- Polling: GET availability at `retryIntervalMs` until `retryWindowMs`
- POST: one booking request after selecting a court

Do not send parallel booking POSTs. The endpoint has no observed idempotency key.

`timeFlexibilityMinutes` supports only `0` or `30`. At `30`, each polling round
checks exact, 30 minutes earlier, and 30 minutes later in parallel, then selects
the first available result in that priority order. Every candidate retains the
original continuous duration. The final reservation payload uses the selected
candidate's `hour_start` and `hour_end`, and there is still exactly one POST.

For a confirmed live booking, `releaseToConfirmationMs` is measured from the
official facility release time (7:00:00.000 AM), not from the configurable
request offset. `bookingResponseMs` measures only the final POST round trip.
`releaseWasOpenAtStart` distinguishes a release-timed run from a manual run
started after the target date had already opened; the GUI does not present the
former metric as booking latency in that case.
The Node process emits one `__BOOKING_SUCCESS__` JSON line for the Windows GUI;
the GUI buffers stdout by complete lines, consumes that marker, and shows the
large green confirmation window.
Keep this marker machine-readable and never include secrets in it.

## Safety requirements

- Never bypass or automate CAPTCHA.
- Never disable Cloudflare or browser security controls.
- Never log secrets or payment data.
- Never run a live POST during tests.
- `--preflight` and `--dry-run` must never send `booking_player`.
- Preserve the direct API path; UI automation is fallback-only.
- Preserve the one-POST rule unless the server exposes a documented idempotency
  mechanism.
- If a POST response is ambiguous, stop and inspect reservations; do not retry.

## Validation checklist after edits

```powershell
npm install
npm test
node --check src/start.mjs
node --check src/api-book.mjs
node --check src/login.mjs
node --check src/configure.mjs
npm run preflight
```

For preflight, the user must first complete `npm run login` and `npm run
configure`. Preflight is read-only and is the required authenticated integration
test.

## Known limitations

- Payment requiring new card entry or 3-D Secure is not handled.
- The API is private and can change without notice.
- The current release supports one owner plus one other player.
- System sleep, network loss, forced logout, CAPTCHA re-challenge, or account
  policy can still prevent booking.
- Server clock calibration is not yet implemented; Windows clock should be kept
  synchronized.
