-- WLRHUD / telem.lua -- telemetry adapter for the wheeled-leg robot.
--
-- Everything the HUD reads goes through here, for one reason: on the radio a
-- reading can be MISSING (firmware does not send it yet), PRESENT BUT STALE
-- (link dropped, last value is a lie), or LIVE. getValue() collapses all three
-- to a number -- it returns 0 both for a sensor that does not exist and for a
-- link that is down. A HUD that cannot tell those apart will happily show
-- "pitch 0.0, all good" while the robot is face down on the floor.
--
-- So: every read returns value, status  where status is one of
--   M.LIVE     fresh, trust it
--   M.STALE    exists, but the link is down -- value is the last one seen
--   M.MISSING  no such reading on this radio yet
-- and the HUD renders MISSING/STALE visibly differently from LIVE.
--
-- TWO SOURCES, deliberately:
--
--  1. Native CRSF sensors. EdgeTX builds these itself from the standard frames
--     the Teensy emits (ATTITUDE, BATTERY_SENSOR, FLIGHT_MODE) plus the link
--     stats the receiver adds. Guaranteed to work: ExpressLRS relays all three.
--
--  2. A private frame, type 0x24, carrying the robot-specific numerics.
--     EdgeTX pushes any frame type it does not natively decode to the Lua
--     queue (crossfire.cpp, `default:` -> pushTelemetryDataToQueues), which is
--     what crossfireTelemetryPop() below drains.
--
--     Whether ExpressLRS relays an arbitrary private type over the air is NOT
--     yet verified on hardware. If it does not, everything from source 2 shows
--     as MISSING and the HUD still works on source 1 -- state, fault, attitude
--     and pack all keep running. That is why state and fault are carried in
--     the FLIGHT_MODE text as well as in the custom frame.

local M = {}

M.LIVE, M.STALE, M.MISSING = 1, 2, 3

-- Native sensor names, in one table so re-pointing the HUD at renamed firmware
-- sensors is a single edit.
M.SENSOR = {
  pitch    = "Ptch",   -- ATTITUDE 0x1E
  roll     = "Roll",   -- ATTITUDE 0x1E
  yaw      = "Yaw",    -- ATTITUDE 0x1E
  pack_v   = "RxBt",   -- BATTERY_SENSOR 0x08
  pack_a   = "Curr",   -- BATTERY_SENSOR 0x08  (NOT instrumented yet: reads 0)
  pack_pct = "Bat%",   -- BATTERY_SENSOR 0x08
  lq       = "RQly",   -- LINK_STATISTICS 0x14, from the receiver
  rssi     = "1RSS",   -- LINK_STATISTICS 0x14
  tpwr     = "TPWR",   -- LINK_STATISTICS 0x14
  mode     = "FM",     -- FLIGHT_MODE 0x21, text: state name or !FAULT
}

-- ATTITUDE arrives in radians and EdgeTX creates the sensor with unit RADIANS,
-- so getValue() hands back radians. The HUD works in degrees.
--
-- Leave that sensor's unit alone in Telemetry setup. If you switch it to
-- degrees, EdgeTX converts and this constant double-counts.
local RAD_TO_DEG = 57.29578

-- Custom frame 0x24, laid out in firmware/robot_teensy/teensy/src/crsf_protocol.h.
-- Keep the two in step; the byte offsets below are that struct.
local WLR_FRAME_ID = 0x24
local WLR_FRAME_LEN = 16

-- Field-id cache. getValue() by numeric id is faster than by name, and this
-- runs inside a widget that must not starve the Lua VM. Re-probed on a slow
-- timer because sensors appear mid-session, the first time telemetry arrives.
local ids = {}
local probed_at = 0
local PROBE_MS = 3000

-- Decoded custom-frame fields, plus when they last arrived.
local wlr = {}
local wlr_ms = nil
local WLR_STALE_MS = 1000

-- Look up any sensor we do not yet have an id for. `false` means "looked and
-- it was not there", and it MUST be retried: the widget almost always loads
-- before the first telemetry frame arrives, so on a cold start every sensor
-- is absent. Caching that absence permanently would leave the HUD blank for
-- the whole session with the link up and working.
local function probe()
  for key, name in pairs(M.SENSOR) do
    if not ids[key] then
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

local function be16(d, i)
  local v = d[i] * 256 + d[i + 1]
  if v >= 32768 then v = v - 65536 end
  return v
end

-- Drain the Lua telemetry queue. Anything that is not our frame is discarded:
-- popping is what keeps the queue from filling, so this must run every tick
-- even when the HUD is not the visible screen.
local function drain(now)
  if crossfireTelemetryPop == nil then return end
  for _ = 1, 8 do                      -- bounded: never spin on a busy link
    local cmd, data = crossfireTelemetryPop()
    if cmd == nil then return end
    if cmd == WLR_FRAME_ID and data ~= nil and #data >= WLR_FRAME_LEN then
      wlr.state    = data[1]
      wlr.fault    = data[2]
      wlr.jump     = data[3]
      wlr.standup  = data[4]
      wlr.alpha    = data[5] / 200.0
      wlr.profile  = data[6]
      wlr.health   = data[7] * 256 + data[8]
      wlr.hip_l    = be16(data, 9) / 100.0
      wlr.hip_r    = be16(data, 11) / 100.0
      wlr.wheel    = be16(data, 13) / 100.0
      wlr.esp32    = data[15]
      wlr.glitch   = data[16]
      wlr_ms = now
    end
  end
end

function M.tick(now)
  if now - probed_at > PROBE_MS then
    probed_at = now
    probe()
  end
  M.link = linkUp()
  M.now = now
  drain(now)
end

-- Fall back to the FLIGHT_MODE text when the custom frame is absent. The
-- firmware sends the state name, or '!' plus a short fault name in ESTOP, so
-- state and fault survive even if the private frame never relays.
local function modeText()
  local id = ids.mode
  if not id then
    probe()
    id = ids.mode
  end
  if not id then return nil end
  local v = getValue(id)
  if type(v) ~= "string" or v == "" then return nil end
  return v
end

function M.faultFromMode()
  local t = modeText()
  if t == nil then return nil end
  return (string.sub(t, 1, 1) == "!") and string.sub(t, 2) or nil
end

function M.stateNameFromMode()
  local t = modeText()
  if t == nil or string.sub(t, 1, 1) == "!" then return nil end
  return t
end

-- Keys that only ever come from the custom frame, so a missing frame reports
-- MISSING rather than silently falling through to a non-existent sensor.
local WLR_KEYS = {
  state = true, fault = true, jump = true, standup = true, alpha = true,
  profile = true, health = true, hip_l = true, hip_r = true, wheel = true,
  esp32 = true, glitch = true,
}

-- Read one value. Returns value, status.
function M.get(key, default)
  -- Custom-frame fields first.
  local v = wlr[key]
  if v ~= nil then
    if wlr_ms == nil then return default, M.MISSING end
    local fresh = M.link and (M.now == nil or (M.now - wlr_ms) < WLR_STALE_MS)
    return v, (fresh and M.LIVE or M.STALE)
  end
  if WLR_KEYS[key] then
    -- Known custom field that has simply never arrived.
    return default, M.MISSING
  end

  local id = ids[key]
  if not id then
    probe()
    id = ids[key]
  end
  if not id then return default, M.MISSING end

  local raw = getValue(id)
  if raw == nil then return default, M.MISSING end
  if key == "pitch" or key == "roll" or key == "yaw" then
    raw = raw * RAD_TO_DEG
  end
  return raw, (M.link and M.LIVE or M.STALE)
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

-- Any robot telemetry at all? The custom frame is the rich source, but the
-- FLIGHT_MODE text alone is enough to say the robot is talking to us.
function M.robotTelemetry()
  return wlr_ms ~= nil or modeText() ~= nil
end

function M.haveCustomFrame()
  return wlr_ms ~= nil
end

function M.bit(flags, mask)
  -- Lua 5.1/5.2 on the radio: no bitwise operators, and bit32 is not always
  -- present. Integer arithmetic is portable across every EdgeTX build.
  if flags == nil or flags <= 0 then return false end
  return math.floor(flags / mask) % 2 == 1
end

return M
