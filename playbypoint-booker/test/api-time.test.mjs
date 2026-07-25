import assert from "node:assert/strict";
import test from "node:test";
import { secondsFromClock } from "../src/api-utils.mjs";

test("converts the requested evening range to API seconds", () => {
  assert.equal(secondsFromClock("20:00"), 72_000);
  assert.equal(secondsFromClock("22:00"), 79_200);
});

test("rejects malformed clock times", () => {
  assert.throws(() => secondsFromClock("8pm"), /Expected HH:MM/);
  assert.throws(() => secondsFromClock("24:00"), /Expected HH:MM/);
});
