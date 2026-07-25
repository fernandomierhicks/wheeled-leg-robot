import assert from "node:assert/strict";
import test from "node:test";
import {
  buildTimeCandidates,
  secondsFromClock
} from "../src/api-utils.mjs";

test("converts the requested evening range to API seconds", () => {
  assert.equal(secondsFromClock("20:00"), 72_000);
  assert.equal(secondsFromClock("22:00"), 79_200);
});

test("rejects malformed clock times", () => {
  assert.throws(() => secondsFromClock("8pm"), /Expected HH:MM/);
  assert.throws(() => secondsFromClock("24:00"), /Expected HH:MM/);
});

test("keeps an exact reservation as one continuous interval", () => {
  assert.deepEqual(buildTimeCandidates(72_000, 79_200, 0), [
    {
      startHour: 72_000,
      endHour: 79_200,
      startTime: "20:00",
      endTime: "22:00"
    }
  ]);
});

test("orders exact, 30 minutes earlier, then 30 minutes later", () => {
  assert.deepEqual(
    buildTimeCandidates(72_000, 79_200, 30).map(
      ({ startTime, endTime }) => `${startTime}-${endTime}`
    ),
    ["20:00-22:00", "19:30-21:30", "20:30-22:30"]
  );
});

test("does not create a flexible interval across a day boundary", () => {
  assert.deepEqual(
    buildTimeCandidates(0, 7_200, 30).map(({ startTime }) => startTime),
    ["00:00", "00:30"]
  );
});
