# PlayByPoint Pickleball Booker

This tool opens a persistent, signed-in Chrome profile before 7:00 AM, waits with
millisecond precision, then uses PlayByPoint's own authenticated JSON API. Chrome
supplies the login cookies and CSRF token; court discovery and booking are direct
requests rather than UI clicks.

At release it makes one availability request for the complete configured time
range, selects the first preferred available court, and posts the reservation.
The older DOM-clicking implementation remains available as `npm run book:ui`.

It stores no password. Chrome keeps the PlayByPoint session in the local,
git-ignored `.playbypoint-profile` directory.

## One-time setup

Open PowerShell in this directory:

```powershell
npm install
Copy-Item config.example.json config.json
notepad config.json
npm run login
```

Sign in in the normal Chrome window and complete any CAPTCHA manually. Once the
Home page appears, close that Chrome window. The setup uses ordinary Chrome
without browser automation so the login itself is not treated as an automated
attempt.

Edit these fields in `config.json`:

- `timePreferences`: fastest acceptable time first.
- `startTime` and `endTime`: the exact continuous reservation block in 24-hour
  `HH:MM` form. The configured target is `20:00`–`22:00`.
- `courtPreferences`: fastest acceptable court first.
- `additionalPlayers`: exact PlayByPoint display name for every player besides
  the account owner. Use `Guest Player` when you do not want to name a specific
  PlayByPoint user.
- `partySize`: must equal one plus the number of additional players.
- `autoSubmit`: leave `false` until a dry run succeeds.

## Test safely

Choose a date that is already released and still has availability:

```powershell
npm run book -- --dry-run --date 2026-07-29
```

Dry-run mode calls availability but never sends the reservation POST.

## Run live

For a target reservation date:

```powershell
npm run book -- --live --date 2026-08-02
```

Start it before the configured prewarm time. It will wait until the correct
release day and 7:00 AM automatically. Live mode clicks **Book**, so only use it
after verifying the config with a dry run.

## Schedule it in Windows

The facility currently exposes dates ten days ahead. The helper subtracts
`bookingWindowDays` from the target reservation date and creates a one-time task
at 6:58 AM. The app then waits internally and prewarms at the configured time:

```powershell
.\register-task.ps1 -TargetDate 2026-08-02 -DryRun
.\register-task.ps1 -TargetDate 2026-08-02
```

The second command replaces the dry-run task with a live task. The task is
allowed to wake the PC and run on battery. Chrome is headed by default, so the
Windows user must still be logged in. Set `headless` to `true` only after a
successful headed dry run.

## Logs and failure evidence

Each attempt writes a JSON result under `logs/`. Failures also save a screenshot.
Both the logs and private browser profile are ignored by Git.

## Important behavior

- The release offset defaults to 75 ms after 7:00:00 to avoid arriving before the
  server opens inventory. Tune `releaseOffsetMs` only after observing real runs.
- Images, fonts, and media are blocked during booking by default so availability
  and checkout controls load with less network competition.
- The script retries availability every 250 ms for six seconds if the new date
  is not visible immediately.
- API mode requests the full `startTime`–`endTime` interval and tries courts in
  configured order.
- It never attempts CAPTCHA handling or rate-limit bypassing.
- PlayByPoint or the facility can change the page. Always perform a dry run after
  a site update or a long period without use.
