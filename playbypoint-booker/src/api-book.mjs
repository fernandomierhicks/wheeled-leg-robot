import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";
import { secondsFromClock } from "./api-utils.mjs";
import { loadConfig, parseArgs, projectRoot } from "./config.mjs";
import {
  formatIsoDate,
  parseIsoDate,
  releaseEpochForTarget,
  targetDateForRun,
  waitUntil,
  zonedEpoch
} from "./time.mjs";

function stamp(message) {
  console.log(`[${new Date().toISOString()}] ${message}`);
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

async function responseJson(response, description) {
  const body = await response.text();
  let parsed;
  try {
    parsed = body ? JSON.parse(body) : {};
  } catch {
    parsed = { raw: body.slice(0, 1000) };
  }
  if (!response.ok()) {
    throw new Error(
      `${description} failed (${response.status()}): ${JSON.stringify(parsed)}`
    );
  }
  return parsed;
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

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(
    "Usage: node src/api-book.mjs [--config PATH] [--date YYYY-MM-DD] [--dry-run|--live]"
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
const releaseEpoch =
  releaseEpochForTarget(
    targetDate,
    config.timeZone,
    config.bookingWindowDays,
    config.releaseTime
  ) + config.releaseOffsetMs;
const prewarmEpoch = releaseEpoch - config.prewarmSeconds * 1000;
const dryRun = args.dryRun ?? !config.autoSubmit;
const facilityId = config.facilityId;
const startHour = secondsFromClock(config.startTime);
const endHour = secondsFromClock(config.endTime);
const durationHours = (endHour - startHour) / 3600;
const targetTimestamp = Math.floor(
  zonedEpoch(
    { ...targetDate, hour: 0, minute: 0, second: 0 },
    config.timeZone
  ) / 1000
);
const bookingUrl = `https://app.playbypoint.com/book/${config.facilitySlug}?skip_waivers=true`;
const profilePath = path.join(projectRoot, ".playbypoint-profile");

if (!facilityId) throw new Error("config.facilityId is required for API mode.");
if (!(endHour > startHour)) throw new Error("endTime must be after startTime.");

stamp(
  `${dryRun ? "API DRY RUN" : "API LIVE"}: ${targetDateText} ${config.startTime}-${config.endTime}.`
);

if (Date.now() < prewarmEpoch) {
  stamp(`Waiting until prewarm at ${new Date(prewarmEpoch).toISOString()}.`);
  await waitUntil(prewarmEpoch);
}

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
  startedAt: new Date().toISOString(),
  targetDate: targetDateText,
  startTime: config.startTime,
  endTime: config.endTime,
  dryRun,
  success: false
};

try {
  await page.goto(bookingUrl, { waitUntil: "domcontentloaded" });
  const root = page.locator('[data-react-class="BookBox"]');
  await root.waitFor({ state: "attached", timeout: 10_000 });
  const props = JSON.parse(await root.getAttribute("data-react-props"));
  if (!props.current_user?.id) {
    throw new Error("Saved PlayByPoint login is missing. Run `npm run login`.");
  }
  if (Number(props.facility_id) !== Number(facilityId)) {
    throw new Error(
      `Facility mismatch: config=${facilityId}, page=${props.facility_id}.`
    );
  }

  const maxHours =
    config.partySize <= 2
      ? Number(props.amount_of_max_consecutive_hours)
      : Number(props.amount_of_max_consecutive_hours_doubles);
  if (maxHours && durationHours > maxHours) {
    throw new Error(
      `Facility rule allows at most ${maxHours} consecutive hours for ${config.partySize} players.`
    );
  }

  const csrfToken = await page
    .locator('meta[name="csrf-token"]')
    .getAttribute("content");
  const headers = {
    Accept: "application/json",
    "X-Requested-With": "XMLHttpRequest",
    ...(csrfToken ? { "X-CSRF-Token": csrfToken } : {})
  };
  const request = context.request;

  const [courtTypes, guestData, balanceData] = await Promise.all([
    responseJson(
      await request.get(
        `https://app.playbypoint.com/api/facilities/${facilityId}/court_types`,
        { headers, params: { kind: "reservation" } }
      ),
      "Court types"
    ),
    responseJson(
      await request.get("https://app.playbypoint.com/api/guest_users", {
        headers,
        params: { facility_id: facilityId, approval: "true" }
      }),
      "Guest users"
    ),
    responseJson(
      await request.get(
        `https://app.playbypoint.com/api/users/${props.current_user.id}/balance/${facilityId}`,
        { headers }
      ),
      "User balance"
    )
  ]);

  const sport = selectSport(courtTypes, config.sportName);
  const guest = (guestData.guests ?? []).find(
    (item) =>
      String(item.name).toLowerCase() ===
      String(config.additionalPlayers[0]).toLowerCase()
  );
  if (!guest) {
    throw new Error(
      `Guest player "${config.additionalPlayers[0]}" was not returned by the API.`
    );
  }
  if (config.additionalPlayers.length !== 1 || config.partySize !== 2) {
    throw new Error(
      "API mode currently supports the account owner plus one Guest Player."
    );
  }

  const paymentMethod = initialPaymentMethod(
    props,
    Number(balanceData.balance ?? 0)
  );
  const paymentMoment = initialPaymentMoment(props);
  stamp(
    `Pre-warmed: user ${props.current_user.id}, surface ${sport.surface}, payment ${paymentMethod}.`
  );

  if (Date.now() < releaseEpoch) {
    await waitUntil(releaseEpoch);
  }

  const deadline = Date.now() + config.retryWindowMs;
  let courts = [];
  let availabilityAttempts = 0;
  while (Date.now() <= deadline) {
    availabilityAttempts += 1;
    const response = await request.get(
      `https://app.playbypoint.com/api/facilities/${facilityId}/available_courts`,
      {
        headers,
        params: {
          date: targetTimestamp,
          surface: sport.surface,
          start_hour: startHour,
          hour_end: endHour,
          kind: "reservation"
        }
      }
    );
    courts = await responseJson(response, "Available courts");
    if (courts.length) break;
    await new Promise((resolve) => setTimeout(resolve, config.retryIntervalMs));
  }
  if (!courts.length) {
    throw new Error(
      `No court was available for the full ${config.startTime}-${config.endTime} block.`
    );
  }

  const court = chooseCourt(courts, config.courtPreferences);
  run.court = court.name;
  run.courtId = court.id;
  run.availabilityAttempts = availabilityAttempts;
  stamp(`API selected ${court.name} after ${availabilityAttempts} availability request(s).`);

  if (dryRun) {
    run.success = true;
    run.readyAt = new Date().toISOString();
    await writeRunLog(run);
    stamp("API dry run complete. No booking request was sent.");
  } else {
    const guestExcluded = props.force_payment_for_guests
      ? []
      : [`${guest.id}_1`];
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
        hour_start: startHour,
        hour_end: endHour,
        reservation_type: config.partySize,
        public_game: false,
        min_ntrp: 1,
        max_ntrp: 7,
        kind: "reservation",
        ntrp_verified: false
      },
      payment,
      user_ids: [props.current_user.id, guest.id],
      user_excluded_ids: guestExcluded,
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
        `https://app.playbypoint.com/api/courts/${court.id}/booking_player`,
        { headers, data: payload }
      ),
      "Booking"
    );
    run.bookingResponseMs = Date.now() - startedPostAt;
    run.completedAt = new Date().toISOString();
    run.location = result.location ?? null;
    run.success = Boolean(result.location?.id || result.location?.slug);
    if (!run.success) {
      throw new Error(
        `Booking API returned without a reservation location: ${JSON.stringify(result)}`
      );
    }
    await writeRunLog(run);
    stamp(
      `BOOKING CONFIRMED in ${run.bookingResponseMs} ms: reservation ${result.location.slug ?? result.location.id}.`
    );
  }
} catch (error) {
  run.error = error instanceof Error ? error.stack : String(error);
  await writeRunLog(run).catch(() => {});
  console.error(error);
  process.exitCode = 1;
} finally {
  await context.close();
}
