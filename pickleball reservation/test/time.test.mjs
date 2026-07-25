import assert from "node:assert/strict";
import test from "node:test";
import {
  addDays,
  dateButtonName,
  formatIsoDate,
  releaseEpochForTarget,
  targetDateForRun,
  zonedParts
} from "../src/time.mjs";

const zone = "America/Los_Angeles";

test("adds days across a month boundary", () => {
  assert.deepEqual(addDays({ year: 2026, month: 7, day: 23 }, 10), {
    year: 2026,
    month: 8,
    day: 2
  });
});

test("computes the target date from the facility booking window", () => {
  const now = Date.parse("2026-07-23T20:00:00Z");
  assert.equal(formatIsoDate(targetDateForRun(now, zone, 10)), "2026-08-02");
});

test("computes a 7 AM Pacific release across daylight saving time", () => {
  const epoch = releaseEpochForTarget(
    { year: 2026, month: 8, day: 2 },
    zone,
    10,
    "07:00:00"
  );
  assert.equal(new Date(epoch).toISOString(), "2026-07-23T14:00:00.000Z");
  assert.deepEqual(zonedParts(epoch, zone), {
    month: 7,
    day: 23,
    year: 2026,
    hour: 7,
    minute: 0,
    second: 0
  });
});

test("matches PlayByPoint's accessible date button label", () => {
  assert.equal(
    dateButtonName({ year: 2026, month: 7, day: 29 }, zone),
    "Wed 29"
  );
});
