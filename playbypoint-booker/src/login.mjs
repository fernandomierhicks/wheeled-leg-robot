import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { projectRoot } from "./config.mjs";

const candidates = [
  path.join(
    process.env.ProgramFiles ?? "C:\\Program Files",
    "Google",
    "Chrome",
    "Application",
    "chrome.exe"
  ),
  path.join(
    process.env["ProgramFiles(x86)"] ?? "C:\\Program Files (x86)",
    "Google",
    "Chrome",
    "Application",
    "chrome.exe"
  ),
  path.join(
    process.env.LOCALAPPDATA ?? "",
    "Google",
    "Chrome",
    "Application",
    "chrome.exe"
  )
];
const chromePath = candidates.find((candidate) => fs.existsSync(candidate));
if (!chromePath) throw new Error("Google Chrome was not found.");

const profilePath = path.join(projectRoot, ".playbypoint-profile");
const url = "https://app.playbypoint.com/home";

console.log("");
console.log("Opening a normal, non-automated Chrome window.");
console.log("Complete any CAPTCHA and sign in manually.");
console.log("After the PlayByPoint Home page appears, close that Chrome window.");
console.log("The saved login will be reused by the scheduled API booking.");

const child = spawn(
  chromePath,
  [
    `--user-data-dir=${profilePath}`,
    "--new-window",
    "--no-first-run",
    url
  ],
  { stdio: "inherit", windowsHide: false }
);

await new Promise((resolve, reject) => {
  child.once("error", reject);
  child.once("exit", (code) => {
    if (code && code !== 0) {
      reject(new Error(`Chrome exited with code ${code}.`));
    } else {
      resolve();
    }
  });
});

console.log("Chrome closed. The private PlayByPoint profile was saved.");
