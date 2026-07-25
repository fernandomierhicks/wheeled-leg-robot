export function secondsFromClock(value) {
  const match = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(value);
  if (!match) throw new Error(`Invalid clock time "${value}". Expected HH:MM.`);
  return Number(match[1]) * 3600 + Number(match[2]) * 60;
}
