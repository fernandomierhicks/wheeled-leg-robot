export function secondsFromClock(value) {
  const match = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(value);
  if (!match) throw new Error(`Invalid clock time "${value}". Expected HH:MM.`);
  return Number(match[1]) * 3600 + Number(match[2]) * 60;
}

export function clockFromSeconds(value) {
  if (!Number.isInteger(value) || value < 0 || value >= 86_400) {
    throw new Error(`Invalid seconds from midnight: ${value}.`);
  }
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

export function buildTimeCandidates(
  startHour,
  endHour,
  flexibilityMinutes = 0
) {
  if (
    !Number.isInteger(startHour) ||
    !Number.isInteger(endHour) ||
    startHour < 0 ||
    endHour > 86_400 ||
    endHour <= startHour
  ) {
    throw new Error("Invalid continuous reservation interval.");
  }
  const flexibility = Number(flexibilityMinutes);
  if (![0, 30].includes(flexibility)) {
    throw new Error("timeFlexibilityMinutes must be 0 or 30.");
  }

  const step = flexibility * 60;
  const offsets = step ? [0, -step, step] : [0];
  return offsets
    .map((offset) => ({
      startHour: startHour + offset,
      endHour: endHour + offset
    }))
    .filter(
      (candidate) =>
        candidate.startHour >= 0 && candidate.endHour <= 86_400
    )
    .map((candidate) => ({
      ...candidate,
      startTime: clockFromSeconds(candidate.startHour),
      endTime:
        candidate.endHour === 86_400
          ? "24:00"
          : clockFromSeconds(candidate.endHour)
    }));
}
