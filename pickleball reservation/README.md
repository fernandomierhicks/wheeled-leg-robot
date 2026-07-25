# Pickleball Reservation

A standalone PlayByPoint reservation application for competitive 7:00 AM court
releases. The time-critical path uses PlayByPoint's authenticated JSON API—not
mouse control, screen coordinates, or booking-page clicks.

The only graphical step is the one-time human login/CAPTCHA in ordinary Chrome.

## How it works

1. Run the app the day before the reservation release.
2. Sign in manually through a dedicated Chrome profile.
3. Configure the target date, continuous start/end time, and other player.
4. Run a read-only authenticated preflight.
5. Start the application and leave it running.
6. The app sends a lightweight keepalive every five minutes.
7. Shortly before release, it refreshes the session and CSRF token.
8. At 7:00 AM it calls `available_courts`, chooses a court, and immediately
   calls `booking_player`.

The critical release path is normally two network round trips:

```text
GET available_courts
        |
        v
choose preferred/any court
        |
        v
POST booking_player
        |
        v
verify reservation ID
```

## Requirements

- Windows with Google Chrome
- Node.js 20 or newer
- A PlayByPoint account
- The computer awake, online, and logged in at release time

## Install

Open PowerShell in this folder:

```powershell
npm install
```

## Recommended day-before workflow

Double-click the Desktop shortcut named **Pickleball Reservation**. The shortcut
opens the Windows GUI; Chrome does not open until you press **Login / Refresh
Session**.

The GUI contains:

- Reservation date picker
- Start and end time selectors
- Editable additional-player dropdown with saved common players
- Court preference selector
- Time flexibility selector: exact only or 30 minutes earlier/later
- **Login / Refresh Session**
- **Save + Test API**
- **Arm Reservation**
- Live status output, **Check Armed Session**, and **Stop Armed Process**
- A large green confirmation window after a successful booking

Use the buttons in order. **Save + Test API** is read-only. **Arm Reservation**
shows the complete live reservation and requires a confirmation before the
booking process starts.

While armed, **Save + Test API** and login are disabled because the live process
owns the dedicated Chrome profile. Use **Check Armed Session** instead. It asks
the already-running process to call an authenticated account endpoint and
reports **ARMED SESSION OK** with the check time. It does not launch another
browser or create a reservation. A temporary health-check failure displays a
warning but does not disarm the live process; automatic keepalive retries.

After a successful live booking, the status and JSON run log include both the
booking API response time and the total milliseconds from the official 7:00 AM
release to confirmation. The confirmation window displays both measurements.

When **Allow 30 minutes earlier or later** is selected, the app checks the
requested continuous interval, the same-duration interval 30 minutes earlier,
and the same-duration interval 30 minutes later. It prefers them in that order
and sends only one booking POST. For example, a requested 8:00-10:00 PM block
tries 8:00-10:00, 7:30-9:30, then 8:30-10:30; it never splits the reservation
into separate time slots.

The command-line wizard remains available:

```powershell
npm start
```

This launches one guided setup:

1. Opens normal Chrome using the private `.playbypoint-profile/`.
2. You complete any CAPTCHA and sign in manually.
3. You enter the reservation date, start/end time, second player, and optional
   court preferences.
4. The app performs a read-only authenticated API preflight.
5. It asks you to type `ARM`.
6. It remains active overnight, performs keepalives, and submits at release.

Passwords are never stored by this application. The private Chrome profile
contains session cookies and is excluded from Git.

No reservation date or player is hard-coded. They are collected every time the
day-before wizard runs.

## Individual setup commands

The wizard above runs these automatically. They are also available separately
for troubleshooting.

### Login only

```powershell
npm run login
```

Complete the CAPTCHA/login, reach the PlayByPoint Home page, then close Chrome.

### Configure only

```powershell
npm run configure
```

Configuration asks for:

- Reservation date in `YYYY-MM-DD`
- Start and end time in 24-hour `HH:MM` format
- Exact PlayByPoint name of the other player, or `Guest Player`
- Optional court preferences; enter `any` to accept any available court

It writes the private, Git-ignored `config.json`.

### Read-only preflight

Run this after login and configuration:

```powershell
npm run preflight
```

Preflight verifies:

- The saved session is authenticated
- The facility and pickleball surface resolve
- The other player resolves to exactly one user
- The requested duration complies with the facility rule
- Balance/payment metadata is readable
- The availability endpoint accepts the requested interval

Preflight never sends the reservation POST. It is safe even when the target date
has not opened yet.

### Arm using an existing configuration

This skips the wizard and immediately uses the existing `config.json`:

```powershell
npm run arm
```

Normally use the Windows GUI. Use `npm start` only when you prefer the terminal
wizard.

Do not run two live copies for the same reservation. The PlayByPoint endpoint
does not expose an idempotency key, so parallel booking POSTs could create
duplicate reservations.

## Other commands

```powershell
npm run dry-run        # Availability GET only; never POSTs a reservation
npm run arm            # Arm an already-authenticated/configured reservation
npm run book:ui        # Legacy DOM-driven fallback; not used by npm start
npm test               # Unit tests
```

`register-task.ps1` remains available as an optional Windows Task Scheduler
alternative, but the recommended workflow is to launch `npm start` the day
before so keepalives and authentication checks happen before release.

## Configuration notes

- `bookingWindowDays`: currently 10 for iTennis/iPickle Arcadia.
- `releaseTime`: currently `07:00:00` Pacific.
- `releaseOffsetMs`: defaults to 75 ms after release.
- `retryIntervalMs`: availability retry interval when inventory propagates late.
- `retryWindowMs`: total availability polling window.
- `keepAliveIntervalSeconds`: authenticated session check interval.
- `courtPreferences`: ordered list; any unlisted returned court is still accepted.
- `headless`: keep `false` unless headed mode has been proven reliable.

## Security and operational limits

- The app does not bypass CAPTCHA, Cloudflare, rate limits, or account controls.
- CAPTCHA is completed manually during login.
- Keepalive requests are intentionally infrequent.
- Only one reservation POST is sent after a court is selected.
- Payment flows requiring new card entry or 3-D Secure cannot be completed by
  the direct path; preflight should expose related account/configuration issues.
- PlayByPoint can change its private API without notice.

## For future AI/code maintenance

Read [AI_EDITING.md](AI_EDITING.md) before changing endpoints, payloads, release
timing, authentication, retries, or payment behavior.
