import fs from "node:fs/promises";
import path from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { projectRoot } from "./config.mjs";
import { parseIsoDate } from "./time.mjs";
import { secondsFromClock } from "./api-utils.mjs";

const templatePath = path.join(projectRoot, "config.example.json");
const configPath = path.join(projectRoot, "config.json");
const config = JSON.parse(await fs.readFile(templatePath, "utf8"));
const rl = readline.createInterface({ input, output });

async function ask(prompt, defaultValue) {
  const suffix =
    defaultValue === undefined || defaultValue === null
      ? ""
      : ` [${defaultValue}]`;
  const value = (await rl.question(`${prompt}${suffix}: `)).trim();
  return value || defaultValue;
}

try {
  config.targetDate = await ask(
    "Reservation date (YYYY-MM-DD)",
    config.targetDate
  );
  parseIsoDate(config.targetDate);

  config.startTime = await ask(
    "Start time in 24-hour format (HH:MM)",
    config.startTime
  );
  config.endTime = await ask(
    "End time in 24-hour format (HH:MM)",
    config.endTime
  );
  const start = secondsFromClock(config.startTime);
  const end = secondsFromClock(config.endTime);
  if (end <= start) throw new Error("End time must be after start time.");

  const player = await ask(
    "Other player's exact PlayByPoint name, or Guest Player",
    config.additionalPlayers[0]
  );
  config.partySize = 2;
  config.additionalPlayers = [player];

  const courtMode = String(
    await ask("Court preference (any, or comma-separated exact names)", "any")
  );
  if (courtMode.toLowerCase() !== "any") {
    const preferred = courtMode
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const remaining = config.courtPreferences.filter(
      (court) => !preferred.includes(court)
    );
    config.courtPreferences = [...preferred, ...remaining];
  }

  await fs.writeFile(
    configPath,
    `${JSON.stringify(config, null, 2)}\n`,
    "utf8"
  );
  console.log("");
  console.log(`Saved ${configPath}`);
  console.log(
    `Reservation: ${config.targetDate}, ${config.startTime}-${config.endTime}, with ${player}`
  );
  console.log("Next: npm run preflight");
} finally {
  rl.close();
}
