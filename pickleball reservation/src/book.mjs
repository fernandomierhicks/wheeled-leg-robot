import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";
import { loadConfig, parseArgs, projectRoot } from "./config.mjs";
import {
  dateButtonName,
  formatIsoDate,
  parseIsoDate,
  releaseEpochForTarget,
  targetDateForRun,
  waitUntil
} from "./time.mjs";

function usage() {
  console.log(`
Usage:
  npm run book -- --dry-run [--date YYYY-MM-DD]
  npm run book -- --live [--date YYYY-MM-DD]
  node src/book.mjs [--config PATH] [--date YYYY-MM-DD] [--dry-run|--live]
`);
}

function stamp(message) {
  console.log(`[${new Date().toISOString()}] ${message}`);
}

async function clickExactlyOne(locator, description) {
  const count = await locator.count();
  if (count !== 1) {
    throw new Error(`${description}: expected one matching control, found ${count}.`);
  }
  await locator.click();
}

async function acceptNoticeIfPresent(page) {
  const accept = page.getByRole("button", { name: "Accept", exact: true });
  if ((await accept.count()) === 1 && (await accept.isVisible())) {
    await accept.click();
  }
}

async function ensureSignedIn(page) {
  const signIn = page.getByRole("button", { name: "Sign in", exact: true });
  if ((await signIn.count()) === 1 && (await signIn.isVisible())) {
    throw new Error("The saved session has expired. Run `npm run login` again.");
  }
}

async function chooseSport(page) {
  const button = page.getByRole("button", {
    name: "Pickleball only",
    exact: true
  });
  await button.waitFor({ state: "visible", timeout: 10_000 });
  await button.click();
}

async function reloadForRelease(page, bookingUrl) {
  await page.goto(bookingUrl, { waitUntil: "domcontentloaded" });
  await ensureSignedIn(page);
  await acceptNoticeIfPresent(page);
  await chooseSport(page);
}

async function selectReleasedDate(page, bookingUrl, label, config) {
  const deadline = Date.now() + config.retryWindowMs;
  let attempt = 0;

  while (Date.now() <= deadline) {
    attempt += 1;
    const dateButton = page.getByRole("button", { name: label, exact: true });
    if (
      (await dateButton.count()) === 1 &&
      (await dateButton.isVisible()) &&
      (await dateButton.isEnabled())
    ) {
      await dateButton.click();
      stamp(`Selected released date ${label} on attempt ${attempt}.`);
      return;
    }

    await page.waitForTimeout(config.retryIntervalMs);
    await reloadForRelease(page, bookingUrl);
  }
  throw new Error(
    `Target date button "${label}" did not appear within ${config.retryWindowMs} ms.`
  );
}

async function chooseFirstAvailableTime(page, preferences) {
  for (const name of preferences) {
    const button = page.getByRole("button", { name, exact: true });
    if (
      (await button.count()) === 1 &&
      (await button.isVisible()) &&
      (await button.isEnabled())
    ) {
      await button.click();
      stamp(`Selected time ${name}.`);
      return name;
    }
  }
  throw new Error(`None of the preferred times is available: ${preferences.join(", ")}`);
}

async function chooseFirstAvailableCourt(page, preferences) {
  for (const name of preferences) {
    const button = page.getByRole("button", { name, exact: true });
    if (
      (await button.count()) === 1 &&
      (await button.isVisible()) &&
      (await button.isEnabled())
    ) {
      await button.click();
      stamp(`Selected ${name}.`);
      return name;
    }
  }
  throw new Error(`None of the preferred courts is available: ${preferences.join(", ")}`);
}

async function addPlayer(page, playerName) {
  await clickExactlyOne(
    page.getByRole("button", { name: "Add Users", exact: false }),
    "Add Users"
  );

  const search = page.getByPlaceholder("Search for Users", { exact: true });
  await search.waitFor({ state: "visible", timeout: 5_000 });
  await search.fill(playerName);

  const result = page.locator(".flex_spcbtw").filter({ hasText: playerName });
  await result.first().waitFor({ state: "visible", timeout: 5_000 });
  const exactMatches = result.filter({
    has: page.getByText(playerName, { exact: true })
  });
  const count = await exactMatches.count();
  if (count !== 1) {
    throw new Error(
      `Player "${playerName}" was not uniquely found (matched ${count} cards).`
    );
  }
  await clickExactlyOne(
    exactMatches.getByRole("button", { name: "Add", exact: true }),
    `Add ${playerName}`
  );
  stamp(`Added player ${playerName}.`);
}

async function selectPlayers(page, config) {
  const partySize = page.getByRole("button", {
    name: String(config.partySize),
    exact: true
  });
  if ((await partySize.count()) === 1 && (await partySize.isEnabled())) {
    await partySize.click();
  }

  for (const playerName of config.additionalPlayers) {
    await addPlayer(page, playerName);
  }
}

async function next(page) {
  await clickExactlyOne(
    page.getByRole("button", { name: "Next", exact: false }),
    "Next"
  );
}

async function writeRunLog(data) {
  const logsPath = path.join(projectRoot, "logs");
  await fs.mkdir(logsPath, { recursive: true });
  const fileName = `${new Date().toISOString().replaceAll(":", "-")}.json`;
  await fs.writeFile(
    path.join(logsPath, fileName),
    `${JSON.stringify(data, null, 2)}\n`,
    "utf8"
  );
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  usage();
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
const bookingUrl = `https://app.playbypoint.com/book/${config.facilitySlug}`;
const dateLabel = dateButtonName(targetDate, config.timeZone);
const profilePath = path.join(projectRoot, ".playbypoint-profile");

stamp(
  `${dryRun ? "DRY RUN" : "LIVE"} for ${targetDateText}; release is ${new Date(releaseEpoch).toISOString()}.`
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
page.setDefaultTimeout(4_000);

const run = {
  startedAt: new Date().toISOString(),
  targetDate: targetDateText,
  dryRun,
  success: false
};

try {
  await reloadForRelease(page, bookingUrl);
  stamp("Booking page is pre-warmed and signed in.");

  if (Date.now() < releaseEpoch) {
    let lastReportedSecond = -1;
    await waitUntil(releaseEpoch, (remaining) => {
      const second = Math.ceil(remaining / 1000);
      if (second !== lastReportedSecond) {
        lastReportedSecond = second;
        stamp(`Release in ${second}s.`);
      }
    });
  }

  stamp("Release reached; refreshing availability.");
  await reloadForRelease(page, bookingUrl);
  await selectReleasedDate(page, bookingUrl, dateLabel, config);
  run.time = await chooseFirstAvailableTime(page, config.timePreferences);
  run.court = await chooseFirstAvailableCourt(page, config.courtPreferences);
  await next(page);
  await selectPlayers(page, config);
  await next(page);

  const book = page.getByRole("button", { name: "Book", exact: true });
  await book.waitFor({ state: "visible", timeout: 5_000 });
  if (!(await book.isEnabled())) throw new Error("The final Book button is disabled.");

  if (dryRun) {
    run.success = true;
    run.readyAt = new Date().toISOString();
    stamp("DRY RUN complete: checkout is ready. The Book button was NOT clicked.");
    await writeRunLog(run);
    const holdMs = Math.max(0, config.keepOpenSecondsAfterDryRun * 1000);
    if (holdMs) {
      stamp(`Keeping the checkout window open for ${config.keepOpenSecondsAfterDryRun}s.`);
      await page.waitForTimeout(holdMs);
    }
  } else {
    stamp("Clicking the final Book button.");
    await book.click();
    await page.waitForTimeout(1200);

    const pageText = await page.locator("body").innerText();
    const url = page.url();
    const success =
      /reservation|booking/i.test(url) ||
      /booked|confirmed|reservation (?:was )?successful/i.test(pageText);
    if (!success) {
      throw new Error(
        `The site did not show a recognizable success confirmation. Current URL: ${url}`
      );
    }
    run.success = true;
    run.completedAt = new Date().toISOString();
    run.confirmationUrl = url;
    await writeRunLog(run);
    stamp(`BOOKING CONFIRMED: ${url}`);
    await page.waitForTimeout(10_000);
  }
} catch (error) {
  run.error = error instanceof Error ? error.stack : String(error);
  await writeRunLog(run).catch(() => {});
  await page.screenshot({
    path: path.join(projectRoot, "logs", `failure-${Date.now()}.png`),
    fullPage: true
  }).catch(() => {});
  console.error(error);
  process.exitCode = 1;
  await page.waitForTimeout(config.headless ? 0 : 30_000).catch(() => {});
} finally {
  await context.close();
}
