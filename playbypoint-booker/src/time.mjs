const datePartsFormatterCache = new Map();

function formatterFor(timeZone) {
  if (!datePartsFormatterCache.has(timeZone)) {
    datePartsFormatterCache.set(
      timeZone,
      new Intl.DateTimeFormat("en-CA", {
        timeZone,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hourCycle: "h23"
      })
    );
  }
  return datePartsFormatterCache.get(timeZone);
}

export function zonedParts(epochMs, timeZone) {
  return Object.fromEntries(
    formatterFor(timeZone)
      .formatToParts(new Date(epochMs))
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, Number(part.value)])
  );
}

export function zonedEpoch(parts, timeZone) {
  const desiredUtc = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour ?? 0,
    parts.minute ?? 0,
    parts.second ?? 0,
    parts.millisecond ?? 0
  );
  let estimate = desiredUtc;

  for (let pass = 0; pass < 3; pass += 1) {
    const seen = zonedParts(estimate, timeZone);
    const seenUtc = Date.UTC(
      seen.year,
      seen.month - 1,
      seen.day,
      seen.hour,
      seen.minute,
      seen.second,
      parts.millisecond ?? 0
    );
    estimate += desiredUtc - seenUtc;
  }
  return estimate;
}

export function parseIsoDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error(`Invalid date "${value}". Expected YYYY-MM-DD.`);
  }
  const [year, month, day] = value.split("-").map(Number);
  return { year, month, day };
}

export function formatIsoDate(parts) {
  return [
    String(parts.year).padStart(4, "0"),
    String(parts.month).padStart(2, "0"),
    String(parts.day).padStart(2, "0")
  ].join("-");
}

export function addDays(dateParts, days) {
  const value = new Date(
    Date.UTC(dateParts.year, dateParts.month - 1, dateParts.day + days)
  );
  return {
    year: value.getUTCFullYear(),
    month: value.getUTCMonth() + 1,
    day: value.getUTCDate()
  };
}

export function targetDateForRun(nowMs, timeZone, bookingWindowDays) {
  const today = zonedParts(nowMs, timeZone);
  return addDays(today, bookingWindowDays);
}

export function releaseEpochForTarget(
  targetDate,
  timeZone,
  bookingWindowDays,
  releaseTime
) {
  const releaseDate = addDays(targetDate, -bookingWindowDays);
  const match = /^(\d{2}):(\d{2}):(\d{2})$/.exec(releaseTime);
  if (!match) {
    throw new Error(`Invalid releaseTime "${releaseTime}". Expected HH:MM:SS.`);
  }
  const [, hour, minute, second] = match.map(Number);
  return zonedEpoch(
    { ...releaseDate, hour, minute, second, millisecond: 0 },
    timeZone
  );
}

export function dateButtonName(targetDate, timeZone) {
  const midday = zonedEpoch(
    { ...targetDate, hour: 12, minute: 0, second: 0 },
    timeZone
  );
  const weekday = new Intl.DateTimeFormat("en-US", {
    timeZone,
    weekday: "short"
  }).format(new Date(midday));
  return `${weekday} ${String(targetDate.day).padStart(2, "0")}`;
}

export async function waitUntil(epochMs, onLongWait = () => {}) {
  while (true) {
    const remaining = epochMs - Date.now();
    if (remaining <= 0) return;
    if (remaining > 1000) {
      onLongWait(remaining);
      await new Promise((resolve) =>
        setTimeout(resolve, Math.min(remaining - 500, 30_000))
      );
      continue;
    }
    if (remaining > 15) {
      await new Promise((resolve) =>
        setTimeout(resolve, Math.max(1, remaining - 8))
      );
      continue;
    }
    // Avoid a scheduler-sized overshoot during the final few milliseconds.
    while (Date.now() < epochMs) {
      // Intentionally empty.
    }
    return;
  }
}
