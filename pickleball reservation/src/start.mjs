import { spawn } from "node:child_process";
import path from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { projectRoot } from "./config.mjs";

function runNode(script, args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(projectRoot, script), ...args], {
      cwd: projectRoot,
      stdio: "inherit",
      windowsHide: false
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (signal) {
        reject(new Error(`${script} stopped by signal ${signal}.`));
      } else if (code !== 0) {
        reject(new Error(`${script} exited with code ${code}.`));
      } else {
        resolve();
      }
    });
  });
}

console.log("");
console.log("======================================================");
console.log("  PlayByPoint Pickleball Reservation — Day-Before Setup");
console.log("======================================================");
console.log("");
console.log("This wizard will:");
console.log("  1. Open normal Chrome for manual login/CAPTCHA");
console.log("  2. Ask for date, time, player, and court preferences");
console.log("  3. Run a read-only API preflight");
console.log("  4. Ask before arming the live reservation");
console.log("  5. Keep the session active until release");
console.log("");

await runNode("src/login.mjs");
await runNode("src/configure.mjs");
await runNode("src/api-book.mjs", ["--preflight"]);

const rl = readline.createInterface({ input, output });
let armed = false;
try {
  console.log("");
  console.log("Preflight passed. No reservation has been created.");
  armed = (
    await rl.question(
      'Type ARM to keep the app running and submit at the configured release time: '
    )
  )
    .trim()
    .toUpperCase() === "ARM";
  if (!armed) {
    console.log("Not armed. You can restart this wizard whenever you are ready.");
  }
} finally {
  rl.close();
}

if (armed) {
  console.log("");
  console.log("LIVE RESERVATION ARMED.");
  console.log("Keep this terminal open and keep the computer awake and online.");
  console.log("");
  await runNode("src/api-book.mjs", ["--live"]);
}
