import fs from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");

export function parseArgs(argv) {
  const args = {
    configPath: path.join(ROOT, "config.json"),
    dryRun: undefined,
    targetDate: undefined,
    preflight: false
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--config") args.configPath = path.resolve(argv[++index]);
    else if (value === "--date") args.targetDate = argv[++index];
    else if (value === "--live") args.dryRun = false;
    else if (value === "--dry-run") args.dryRun = true;
    else if (value === "--preflight") {
      args.preflight = true;
      args.dryRun = true;
    }
    else if (value === "--help" || value === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${value}`);
  }
  return args;
}

export async function loadConfig(configPath) {
  let source;
  try {
    source = await fs.readFile(configPath, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error(
        `Missing ${configPath}. Copy config.example.json to config.json and edit it first.`
      );
    }
    throw error;
  }

  // Windows PowerShell 5.1 writes a UTF-8 BOM for `-Encoding UTF8`.
  // Accept one defensively even though the GUI now saves BOM-free UTF-8.
  const config = JSON.parse(source.replace(/^\uFEFF/, ""));
  const required = [
    "facilitySlug",
    "facilityId",
    "sportName",
    "timeZone",
    "bookingWindowDays",
    "releaseTime",
    "startTime",
    "endTime",
    "timePreferences",
    "courtPreferences",
    "partySize",
    "additionalPlayers"
  ];
  for (const key of required) {
    if (config[key] === undefined) throw new Error(`Config is missing "${key}".`);
  }
  if (!Array.isArray(config.timePreferences) || !config.timePreferences.length) {
    throw new Error("timePreferences must contain at least one time.");
  }
  if (!Array.isArray(config.courtPreferences) || !config.courtPreferences.length) {
    throw new Error("courtPreferences must contain at least one court.");
  }
  if (config.additionalPlayers.length !== config.partySize - 1) {
    throw new Error(
      `partySize ${config.partySize} requires exactly ${config.partySize - 1} additionalPlayers.`
    );
  }
  config.timeFlexibilityMinutes = Number(config.timeFlexibilityMinutes ?? 0);
  if (![0, 30].includes(config.timeFlexibilityMinutes)) {
    throw new Error("timeFlexibilityMinutes must be 0 or 30.");
  }
  return config;
}

export const projectRoot = ROOT;
