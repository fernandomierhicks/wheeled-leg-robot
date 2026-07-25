import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";
import {
  buildTimeCandidates,
  secondsFromClock
} from "./api-utils.mjs";
import { loadConfig, parseArgs, projectRoot } from "./config.mjs";
import {
  formatIsoDate,
  parseIsoDate,
  releaseEpochForTarget,
  targetDateForRun,
  waitUntil,
  zonedEpoch
} from "./time.mjs";

const ORIGIN = "https://app.playbypoint.com";

function stamp(message) {
  console.log(`[${new Date().toISOString()}] ${message}`);
}

async function responseJson(response, description) {
  const body = await response.text();
  let parsed;
  try {
    parsed = body ? JSON.parse(body) : {};
  } catch {
    parsed = { raw: body.slice(0, 1000) };
  }
  if (!response.ok()) {
    if (
      response.status() === 403 &&
      typeof parsed.raw === "string" &&
      (parsed.raw.includes("challenges.cloudflare.com") ||
        parsed.raw.includes("<title>Just a moment...</title>"))
    ) {
      throw new Error(
        "Cloudflare browser verification is required. Click Login / Refresh " +
          "Session, complete any browser check, close Chrome, then run " +
          "Save + Test API again."
      );
    }
    throw new Error(
      `${description} failed (${response.status()}): ${JSON.stringify(parsed)}`
    );
  }
  return parsed;
}

function createPageApiClient(page) {
  async function send(method, rawUrl, options = {}) {
    const url = new URL(rawUrl);
    for (const [key, value] of Object.entries(options.params ?? {})) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    }

    const result = await page.evaluate(
      async ({ method, url, headers, data }) => {
        const response = await fetch(url, {
          method,
          credentials: "same-origin",
          cache: "no-store",
          headers: {
            ...headers,
            ...(data === undefined ? {} : { "Content-Type": "application/json" })
          },
          ...(data === undefined ? {} : { body: JSON.stringify(data) })
        });
        return {
          ok: response.ok,
          status: response.status,
          body: await response.text()
        };
      },
      {
        method,
        url: url.toString(),
        headers: options.headers ?? {},
        data: options.data
      }
    );

    return {
      ok: () => result.ok,
      status: () => result.status,
      text: async () => result.body
    };
  }

  return {
    get: (url, options) => send("GET", url, options),
    post: (url, options) => send("POST", url, options)
  };
}

function initialPaymentMethod(props, balance) {
  if ((balance > 0 && props.allow_prepaid) || props.allow_razor_pay) {
    return "prepaid";
  }
  if (props.userDefaultPaymentMethod) return props.userDefaultPaymentMethod;
  if (props.allow_pay_online) return "card";
  if (props.allow_prepaid) return "prepaid";
  if (props.allow_accounts_receivable) return "accounts_receivable";
  if (props.allow_qr_payments) return "qr";
  if (props.allow_ewallet_payments) return "ewallet";
  if (props.allow_reservation_without_payment) return "cash";
  return "card";
}

function initialPaymentMoment(props) {
  if (
    props.allow_pay_online ||
    props.allow_prepaid ||
    props.allow_accounts_receivable ||
    props.allow_qr_payments ||
    props.allow_ewallet_payments
  ) {
    return "now";
  }
  return props.allow_reservation_without_payment ? "later" : undefined;
}

function chooseCourt(courts, preferences) {
  for (const preferred of preferences) {
    const match = courts.find(
      (court) => String(court.name).toLowerCase() === preferred.toLowerCase()
    );
    if (match) return match;
  }
  return courts[0];
}

function selectSport(courtTypes, sportName) {
  const wanted = sportName.toLowerCase();
  const exact = courtTypes.find((item) =>
    [item.surface_name, item.surface]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase() === wanted)
  );
  if (exact) return exact;
  const pickleball = courtTypes.find((item) =>
    [item.surface_name, item.surface]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes("pickleball"))
  );
  if (pickleball) return pickleball;
  throw new Error(`Court type "${sportName}" was not found.`);
}

function exactName(items, name) {
  const wanted = String(name).trim().toLowerCase();
  const matches = items.filter(
    (item) => String(item.name).trim().toLowerCase() === wanted
  );
  if (matches.length > 1) {
    throw new Error(`Player "${name}" matched more than one account.`);
  }
  return matches[0];
}

async function writeRunLog(data) {
  const logsPath = path.join(projectRoot, "logs");
  await fs.mkdir(logsPath, { recursive: true });
  const fileName = `api-${new Date().toISOString().replaceAll(":", "-")}.json`;
  await fs.writeFile(
    path.join(logsPath, fileName),
    `${JSON.stringify(data, null, 2)}\n`,
    "utf8"
  );
}

async function loadSession(page, bookingUrl, facilityId) {
  await page.goto(bookingUrl, { waitUntil: "domcontentloaded" });
  const root = page.locator('[data-react-class="BookBox"]');
  await root.waitFor({ state: "attached", timeout: 15_000 });
  const props = JSON.parse(await root.getAttribute("data-react-props"));
  if (!props.current_user?.id) {
    throw new Error("Saved PlayByPoint login is missing. Run `npm run login`.");
  }
  if (Number(props.facility_id) !== Number(facilityId)) {
    throw new Error(
      `Facility mismatch: config=${facilityId}, page=${props.facility_id}.`
    );
  }
  const csrfToken = await page
    .locator('meta[name="csrf-token"]')
    .getAttribute("content");
  return {
    props,
    headers: {
      Accept: "application/json",
      "X-Requested-With": "XMLHttpRequest",
      ...(csrfToken ? { "X-CSRF-Token": csrfToken } : {})
    }
  };
}

async function getBalance(request, headers, userId, facilityId) {
  const balance = await responseJson(
    await request.get(
      `${ORIGIN}/api/users/${userId}/balance/${facilityId}`,
      { headers }
    ),
    "User balance"
  );
  if (
    !balance ||
    typeof balance !== "object" ||
    !Object.prototype.hasOwnProperty.call(balance, "balance")
  ) {
    throw new Error(
      "The authenticated session check did not return account data. " +
        "Run Login / Refresh Session again."
    );
  }
  return balance;
}

async function consumeHealthCheck(filePath) {
  try {
    const source = await fs.readFile(filePath, "utf8");
    await fs.unlink(filePath).catch(() => {});
    return JSON.parse(source.replace(/^\uFEFF/, ""));
  } catch (error) {
    if (error.code === "ENOENT") return null;
    await fs.unlink(filePath).catch(() => {});
    throw new Error(`Could not read the armed-session health request: ${error.message}`);
  }
}

async function resolvePlayer({
  request,
  headers,
  props,
  facilityId,
  ratingProvider,
  playerName
}) {
  const guestData = await responseJson(
    await request.get(`${ORIGIN}/api/guest_users`, {
      headers,
      params: { facility_id: facilityId, approval: "true" }
    }),
    "Guest users"
  );
  const guest = exactName(guestData.guests ?? [], playerName);
  if (guest) return guest;

  const followingData = await responseJson(
    await request.get(`${ORIGIN}/api/users/${props.current_user.id}/following`, {
      headers,
      params: {
        include_affiliation: "true",
        current_facility: facilityId,
        ...(ratingProvider ? { rating_provider: ratingProvider } : {})
      }
    }),
    "Following list"
  );
  const following = exactName(followingData.following ?? [], playerName);
  if (following) return following;

  const searchData = await responseJson(
    await request.get(`${ORIGIN}/api/find_user`, {
      headers,
      params: {
        q: playerName,
        is_teacher: "false",
        limit: 15,
        with_social_info: "true",
        include_affiliation: "true",
        current_facility: facilityId,
        ...(ratingProvider ? { rating_provider: ratingProvider } : {})
      }
    }),
    "Player search"
  );
  const remote = exactName(searchData.users ?? [], playerName);
  if (remote) return remote;
  throw new Error(
    `Player "${playerName}" was not found. Use the exact PlayByPoint display name or Guest Player.`
  );
}

function excludedPlayerIds(props, player) {
  if (player.is_guest) {
    return props.force_payment_for_guests ? [] : [`${player.id}_1`];
  }
  const rules = props.force_payment_by_affiliation_on_surface ?? {};
  return rules[player.affiliation] ? [] : [`${player.id}_1`];
}

async function fetchCourts({
  request,
  headers,
  facilityId,
  targetTimestamp,
  surface,
  startHour,
  endHour
}) {
  return responseJson(
    await request.get(`${ORIGIN}/api/facilities/${facilityId}/available_courts`, {
      headers,
      params: {
        date: targetTimestamp,
        surface,
        start_hour: startHour,
        hour_end: endHour,
        kind: "reservation"
      }
    }),
    "Available courts"
  );
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(
    "Usage: node src/api-book.mjs [--config PATH] [--date YYYY-MM-DD] [--preflight|--dry-run|--live]"
  );
  process.exit(0);
}

const config = await loadConfig(args.configPath);
const targetDate = parseIsoDate(
  args.targetDate ??
    config.targetDate ??
    formatIsoDate(
      targetDateForRun(Date.now(), config.timeZone, config.bookingWindowDays)
    )
);
const targetDateText = formatIsoDate(targetDate);
const officialReleaseEpoch = releaseEpochForTarget(
  targetDate,
  config.timeZone,
  config.bookingWindowDays,
  config.releaseTime
);
const releaseEpoch = officialReleaseEpoch + config.releaseOffsetMs;
const prewarmEpoch = releaseEpoch - config.prewarmSeconds * 1000;
const dryRun = args.dryRun ?? !config.autoSubmit;
const facilityId = Number(config.facilityId);
const startHour = secondsFromClock(config.startTime);
const endHour = secondsFromClock(config.endTime);
const timeCandidates = buildTimeCandidates(
  startHour,
  endHour,
  config.timeFlexibilityMinutes
);
const durationHours = (endHour - startHour) / 3600;
const targetTimestamp = Math.floor(
  zonedEpoch(
    { ...targetDate, hour: 0, minute: 0, second: 0 },
    config.timeZone
  ) / 1000
);
const bookingUrl = `${ORIGIN}/book/${config.facilitySlug}?skip_waivers=true`;
const profilePath = path.join(projectRoot, ".playbypoint-profile");
const healthCheckPath = path.join(
  projectRoot,
  "logs",
  "gui-healthcheck.request.json"
);
const playerName = config.additionalPlayers[0];
const runStartedAt = Date.now();
const releaseWasOpenAtStart = runStartedAt >= officialReleaseEpoch;

if (!facilityId) throw new Error("config.facilityId is required.");
if (!(endHour > startHour)) throw new Error("endTime must be after startTime.");
if (config.partySize !== 2 || config.additionalPlayers.length !== 1) {
  throw new Error(
    "This release supports the account owner plus exactly one additional player."
  );
}

stamp(
  `${args.preflight ? "PREFLIGHT" : dryRun ? "API DRY RUN" : "API LIVE"}: ` +
    `${targetDateText} ${config.startTime}-${config.endTime}, with ${playerName}; ` +
    `time mode=${config.timeFlexibilityMinutes === 30 ? "plus/minus 30 minutes" : "exact"}.`
);

const context = await chromium.launchPersistentContext(profilePath, {
  channel: "chrome",
  headless: config.headless,
  viewport: null
});
if (config.blockHeavyAssets) {
  await context.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type === "image" || type === "font" || type === "media") {
      await route.abort();
    } else {
      await route.continue();
    }
  });
}
const pages = context.pages();
const page = pages[0] ?? (await context.newPage());
const run = {
  mode: "api",
  startedAt: new Date(runStartedAt).toISOString(),
  releaseWasOpenAtStart,
  targetDate: targetDateText,
  startTime: config.startTime,
  endTime: config.endTime,
  player: playerName,
  dryRun,
  preflight: args.preflight,
  success: false
};

try {
  let session = await loadSession(page, bookingUrl, facilityId);
  const request = createPageApiClient(page);
  let { props, headers } = session;
  const maxHours =
    config.partySize <= 2
      ? Number(props.amount_of_max_consecutive_hours)
      : Number(props.amount_of_max_consecutive_hours_doubles);
  if (maxHours && durationHours > maxHours) {
    throw new Error(
      `Facility rule allows at most ${maxHours} consecutive hours for ${config.partySize} players.`
    );
  }

  const courtTypes = await responseJson(
    await request.get(`${ORIGIN}/api/facilities/${facilityId}/court_types`, {
      headers,
      params: { kind: "reservation" }
    }),
    "Court types"
  );
  const sport = selectSport(courtTypes, config.sportName);
  const player = await resolvePlayer({
    request,
    headers,
    props,
    facilityId,
    ratingProvider: sport.rating_provider,
    playerName
  });
  let balanceData = await getBalance(
    request,
    headers,
    props.current_user.id,
    facilityId
  );
  let paymentMethod = initialPaymentMethod(
    props,
    Number(balanceData.balance ?? 0)
  );
  let paymentMoment = initialPaymentMoment(props);

  stamp(
    `Authenticated and resolved ${player.name}; surface=${sport.surface}; payment=${paymentMethod}.`
  );

  if (args.preflight) {
    const availability = await Promise.all(
      timeCandidates.map(async (candidate) => ({
        candidate,
        courts: await fetchCourts({
          request,
          headers,
          facilityId,
          targetTimestamp,
          surface: sport.surface,
          startHour: candidate.startHour,
          endHour: candidate.endHour
        })
      }))
    );
    run.availabilityByTime = Object.fromEntries(
      availability.map(({ candidate, courts }) => [
        `${candidate.startTime}-${candidate.endTime}`,
        courts.length
      ])
    );
    run.availableCourtCount = availability.reduce(
      (sum, { courts }) => sum + courts.length,
      0
    );
    run.success = true;
    run.completedAt = new Date().toISOString();
    await writeRunLog(run);
    stamp(
      `PREFLIGHT PASSED. Availability by time: ` +
        `${JSON.stringify(run.availabilityByTime)}; no reservation POST was sent.`
    );
    process.exitCode = 0;
  } else {
    const keepAliveMs = Math.max(
      60_000,
      Number(config.keepAliveIntervalSeconds ?? 300) * 1000
    );
    let nextKeepAliveAt = Date.now() + keepAliveMs;
    while (Date.now() < prewarmEpoch) {
      const nextCheck = Math.min(
        Date.now() + 1000,
        nextKeepAliveAt,
        prewarmEpoch
      );
      await waitUntil(nextCheck);
      if (Date.now() >= prewarmEpoch) break;

      const healthRequest = await consumeHealthCheck(healthCheckPath);
      const automaticKeepAliveDue = Date.now() >= nextKeepAliveAt;
      if (healthRequest || automaticKeepAliveDue) {
        try {
          balanceData = await getBalance(
            request,
            headers,
            props.current_user.id,
            facilityId
          );
          nextKeepAliveAt = Date.now() + keepAliveMs;
          if (automaticKeepAliveDue) {
            stamp(
              `Session keepalive passed; release in ${Math.ceil((releaseEpoch - Date.now()) / 60_000)} minute(s).`
            );
          }
          if (healthRequest) {
            console.log(
              `__ARMED_HEALTH__${JSON.stringify({
                requestedAt: healthRequest.requestedAt ?? null,
                checkedAt: new Date().toISOString(),
                releaseInMs: Math.max(0, officialReleaseEpoch - Date.now())
              })}`
            );
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          nextKeepAliveAt =
            Date.now() + Math.min(60_000, keepAliveMs);
          if (automaticKeepAliveDue) {
            stamp(`SESSION KEEPALIVE WARNING: ${message} Retrying later.`);
          }
          if (healthRequest) {
            console.log(
              `__ARMED_HEALTH_ERROR__${JSON.stringify({
                requestedAt: healthRequest.requestedAt ?? null,
                checkedAt: new Date().toISOString(),
                message
              })}`
            );
          }
        }
      }
    }

    if (Date.now() < releaseEpoch) {
      session = await loadSession(page, bookingUrl, facilityId);
      props = session.props;
      headers = session.headers;
      balanceData = await getBalance(
        request,
        headers,
        props.current_user.id,
        facilityId
      );
      paymentMethod = initialPaymentMethod(
        props,
        Number(balanceData.balance ?? 0)
      );
      paymentMoment = initialPaymentMoment(props);
      stamp("Prewarm refreshed the session and CSRF token.");
      await waitUntil(releaseEpoch);
    }

    const deadline = Date.now() + config.retryWindowMs;
    let courts = [];
    let selectedTime = timeCandidates[0];
    let availabilityAttempts = 0;
    while (Date.now() <= deadline) {
      availabilityAttempts += 1;
      const availability = await Promise.all(
        timeCandidates.map(async (candidate) => ({
          candidate,
          courts: await fetchCourts({
            request,
            headers,
            facilityId,
            targetTimestamp,
            surface: sport.surface,
            startHour: candidate.startHour,
            endHour: candidate.endHour
          })
        }))
      );
      const available = availability.find(
        ({ courts: candidateCourts }) => candidateCourts.length
      );
      if (available) {
        selectedTime = available.candidate;
        courts = available.courts;
        break;
      }
      await new Promise((resolve) =>
        setTimeout(resolve, config.retryIntervalMs)
      );
    }
    if (!courts.length) {
      throw new Error(
        `No court was available for any full continuous block: ` +
          `${timeCandidates
            .map(({ startTime, endTime }) => `${startTime}-${endTime}`)
            .join(", ")}.`
      );
    }

    const court = chooseCourt(courts, config.courtPreferences);
    run.court = court.name;
    run.courtId = court.id;
    run.bookedStartTime = selectedTime.startTime;
    run.bookedEndTime = selectedTime.endTime;
    run.availabilityAttempts = availabilityAttempts;
    stamp(
      `Selected ${court.name}, ${selectedTime.startTime}-${selectedTime.endTime}, ` +
        `after ${availabilityAttempts} availability round(s).`
    );

    if (dryRun) {
      run.success = true;
      run.completedAt = new Date().toISOString();
      await writeRunLog(run);
      stamp("API dry run complete. No reservation POST was sent.");
    } else {
      const payment = {
        method: paymentMethod,
        payment_intent_id: "",
        card_details: {},
        coupon: { code: "" },
        booking_package_purchase_id: null,
        ...(paymentMoment ? { moment: paymentMoment } : {})
      };
      const payload = {
        reservation: {
          date: targetDateText,
          hour_start: selectedTime.startHour,
          hour_end: selectedTime.endHour,
          reservation_type: config.partySize,
          public_game: false,
          min_ntrp: 1,
          max_ntrp: 7,
          kind: "reservation",
          ntrp_verified: false
        },
        payment,
        user_ids: [props.current_user.id, player.id],
        user_excluded_ids: excludedPlayerIds(props, player),
        user_ids_guest_names: {
          player0: { name: null },
          player1: { name: null }
        },
        reservation_fees: [],
        users_fees: [],
        auto_fill_courts: props.autoAssingCourtsRule,
        free_fare_players: [],
        guest_pass_users: [],
        booking_package_applies_to_user_ids: []
      };

      const startedPostAt = Date.now();
      const result = await responseJson(
        await request.post(
          `${ORIGIN}/api/courts/${court.id}/booking_player`,
          { headers, data: payload }
        ),
        "Booking"
      );
      const confirmedAt = Date.now();
      run.bookingResponseMs = confirmedAt - startedPostAt;
      run.releaseToConfirmationMs = confirmedAt - officialReleaseEpoch;
      run.completedAt = new Date(confirmedAt).toISOString();
      run.location = result.location ?? null;
      run.success = Boolean(result.location?.id || result.location?.slug);
      if (!run.success) {
        throw new Error(
          `Booking API returned without a reservation location: ${JSON.stringify(result)}`
        );
      }
      await writeRunLog(run);
      const confirmationTiming = releaseWasOpenAtStart
        ? `reservations were already open when this run started; ` +
          `booking API: ${run.bookingResponseMs} ms`
        : `${run.releaseToConfirmationMs} ms after release; ` +
          `booking API: ${run.bookingResponseMs} ms`;
      stamp(
        `BOOKING CONFIRMED (${confirmationTiming}): reservation ` +
          `${result.location.slug ?? result.location.id}.`
      );
      console.log(
        `__BOOKING_SUCCESS__${JSON.stringify({
          court: court.name,
          date: targetDateText,
          startTime: selectedTime.startTime,
          endTime: selectedTime.endTime,
          player: player.name,
          releaseWasOpenAtStart,
          releaseToConfirmationMs: run.releaseToConfirmationMs,
          bookingResponseMs: run.bookingResponseMs,
          reservation: result.location.slug ?? result.location.id
        })}`
      );
    }
  }
} catch (error) {
  run.error = error instanceof Error ? error.stack : String(error);
  await writeRunLog(run).catch(() => {});
  console.error(error);
  process.exitCode = 1;
} finally {
  await context.close();
}
