-- WLRHUD / telem.lua -- telemetry adapter for the wheeled-leg robot.
--
-- Everything the HUD reads goes through here, for one reason: on the radio a
-- sensor can be MISSING (firmware does not send it yet), PRESENT BUT STALE
-- (link dropped, last value is a lie), or LIVE. getValue() collapses all three
-- to a number -- it returns 0 both for a sensor that does not exist and for a
-- link that is down. A HUD that cannot tell those apart will happily show
-- "pitch 0.0, all good" while the robot is face down on the floor.
--
-- So: every read returns value, status  where status is one of
--   M.LIVE     fresh, trust it
--   M.STALE    sensor exists, link is down -- value is the last one seen
--   M.MISSING  no such sensor on this radio yet
-- and the HUD renders MISSING/STALE visibly differently from LIVE.

local M = {}

M.LIVE, M.STALE, M.MISSING = 1, 2, 3

-- Sensor names, in one table so re-pointing the HUD at renamed firmware
-- sensors is a single edit here.
--
-- native: created automatically by EdgeTX from the standard CRSF frames
--   listed in tx15-robot-integration-plan.md section 3. These appear as soon
--   as the receiver is bound and the Teensy emits the frames.
-- custom: the robot-specific fields. These do NOT exist until the CRSF
--   telemetry emitter lands in firmware (plan section 6, item 3). Until then
--   the HUD shows them as MISSING, which is the intended behaviour -- it is
--   honest about what the radio can and cannot see.
M.SENSOR = {
  -- native CRSF
  pitch    = "Ptch",   -- ATTITUDE 0x1E, degrees
  roll     = "Roll",   -- ATTITUDE 0x1E, degrees
  yaw      = "Yaw",    -- ATTITUDE 0x1E, degrees
  pack_v   = "RxBt",   -- BATTERY_SENSOR 0x08, volts
  pack_a   = "Curr",   -- BATTERY_SENSOR 0x08, amps
  pack_pct = "Bat%",   -- BATTERY_SENSOR 0x08, percent
  lq       = "RQly",   -- LINK_STATISTICS 0x14, percent
  rssi     = "1RSS",   -- LINK_STATISTICS 0x14, dBm
  tpwr     = "TPWR",   -- LINK_STATISTICS 0x14, mW

  -- custom frame, pending firmware
  state    = "Stat",   -- robot_state, RobotStateEnum
  fault    = "Flt",    -- fault_code, FAULT_*
  alpha    = "Alph",   -- gain_sched_alpha, 0..1
  hip_l    = "HipL",   -- hip_l_torque_nm
  hip_r    = "HipR",   -- hip_r_torque_nm
  wheel    = "WVel",   -- wheel_vel_avg_ms
  jump     = "Jump",   -- jump_state phase
  standup  = "SUp",    -- standup_state phase
  health   = "Hlth",   -- health_flags bitfield
  glitch   = "Glch",   -- vel_glitch_count
  esp32    = "E32",    -- esp32_link_ok
  profile  = "Prof",   -- active_profile
}

-- Field-id cache. getValue() by numeric id is faster than by name, and this
-- runs inside a widget that must not starve the Lua VM. Re-probed on a slow
-- timer because sensors appear mid-session, the first time telemetry arrives.
local ids = {}
local probed_at = 0
local PROBE_MS = 3000

local function probe()
  for key, name in pairs(M.SENSOR) do
    if ids[key] == nil then
      local info = getFieldInfo(name)
      ids[key] = info and info.id or false
    end
  end
end

-- Link liveness. getRSSI() is 0 when no telemetry has arrived recently, which
-- is exactly the "the last value is a lie" condition.
local function linkUp()
  if getRSSI == nil then return false end
  local r = getRSSI()
  return r ~= nil and r > 0
end

function M.tick(now)
  if now - probed_at > PROBE_MS then
    probed_at = now
    probe()
  end
  M.link = linkUp()
end

-- Read one mapped sensor. Returns value, status.
function M.get(key, default)
  local id = ids[key]
  if id == nil then
    probe()
    id = ids[key]
  end
  if not id then
    return default, M.MISSING
  end
  local v = getValue(id)
  if v == nil then
    return default, M.MISSING
  end
  return v, (M.link and M.LIVE or M.STALE)
end

-- Convenience: value only, default when not LIVE. Use where a stale number is
-- worse than no number -- attitude, for example.
function M.live(key, default)
  local v, st = M.get(key, default)
  if st ~= M.LIVE then return default end
  return v
end

function M.present(key)
  local _, st = M.get(key)
  return st ~= M.MISSING
end

-- Any custom sensor at all? Used to show the "waiting for robot telemetry"
-- placeholder instead of a HUD full of dashes.
function M.robotTelemetry()
  return M.present("state") or M.present("alpha") or M.present("wheel")
end

function M.bit(flags, mask)
  -- Lua 5.1/5.2 on the radio: no bitwise operators, and bit32 is not always
  -- present. Integer arithmetic is portable across every EdgeTX build.
  if flags == nil or flags <= 0 then return false end
  return math.floor(flags / mask) % 2 == 1
end

return M
