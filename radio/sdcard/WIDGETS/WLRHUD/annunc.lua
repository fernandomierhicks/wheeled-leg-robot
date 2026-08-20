-- WLRHUD / annunc.lua -- audible + haptic annunciator.
--
-- This is an INSTRUMENT, not a control. It never writes anything to the robot
-- and never touches a channel. Arm, disarm, ESTOP and the rescue combo stay on
-- physical channels and firmware interlocks, because a Lua script can be
-- exited, starved or crash, and none of those may be able to disarm a robot.
-- (tx15-robot-integration-plan.md, section 2, "Never put safety on the Lua path".)
--
-- What it does buy: when the robot is balancing you are looking at IT, not at
-- a screen. Sixteen fault codes currently mean walking back to the laptop to
-- find out which one fired. Here they are spoken, tiered by the recovery they
-- need, so you know whether to reach for the robot or the power switch before
-- you have crossed the room.

local SND = "/SOUNDS/en/WLR/"

local M = {}

-- Tier -> how loud. Matches fault_severity() in shared/comm_protocol.h,
-- carried into robotdef.lua by the generator.
local TIER = {
  SOFT       = { tone = "t_soft",  haptic = 0,  speak = true },
  REPOSITION = { tone = "t_repos", haptic = 30, speak = true },
  GUI_FIX    = { tone = nil,       haptic = 0,  speak = true },
  REBOOT     = { tone = "t_siren", haptic = 60, speak = true },
}

local function play(name, volume)
  if name then playFile(SND .. name .. ".wav", volume) end
end

local function haptic(ms)
  if ms > 0 and playHaptic ~= nil then playHaptic(ms, 0) end
end

function M.new(def)
  local a = {
    def = def,
    state = nil,          -- last announced robot state
    fault = nil,          -- last announced fault code
    linked = nil,         -- last announced telemetry-link state
    esp32 = nil,          -- last announced esp32_link_ok
    -- Rate limits. Each of these has a "next time this may speak" stamp so a
    -- value dithering across its threshold cannot turn into a machine gun.
    nextPitch = 0,
    nextHipHot = 0,
    nextGlitch = 0,
    nextBatt = 0,
    hipHotSince = nil,
    glitchLast = nil,
    enabled = true,
  }
  return a
end

-- Predictive pitch warning. A rising tone as pitch crosses the barrier and
-- climbs toward the watchdog trip -- a stall-warning analogue. The point is to
-- get an audible cue BEFORE the ESTOP, in the window where you can still catch
-- the robot.
local PITCH_WARN_DEG = 18.0   -- start warning here
local PITCH_TRIP_DEG = 32.0   -- roughly where pitch_wd_fwd/bwd land

local function pitchWarn(a, now, pitch)
  local mag = math.abs(pitch or 0)
  if mag < PITCH_WARN_DEG then
    a.nextPitch = 0        -- re-arm the moment it comes back inside
    return
  end
  if now < a.nextPitch then return end
  local span = PITCH_TRIP_DEG - PITCH_WARN_DEG
  local frac = math.min(1.0, (mag - PITCH_WARN_DEG) / span)
  -- Closer to the trip: higher, faster, more urgent.
  local f0 = 700 + math.floor(frac * 700)
  playTone(f0, 90, 0, 0, math.floor(200 + frac * 600))
  a.nextPitch = now + math.floor(450 - frac * 300)
end

-- Hip thermal. The measured table has 3.0 Nm at about 9.2 W and a ~64 C rise,
-- and 4.07 Nm holds settling near 143 C -- past the Class B limit if
-- sustained. The winding time constant is about 30 s, so a crouch-hold that
-- has been over 3.0 Nm for 20 s is worth a word before it becomes a smell.
local HIP_HOT_NM = 3.0
local HIP_HOT_S = 20.0

local function hipThermal(a, now, tl, tr)
  local peak = math.max(math.abs(tl or 0), math.abs(tr or 0))
  if peak < HIP_HOT_NM then
    a.hipHotSince = nil
    return
  end
  a.hipHotSince = a.hipHotSince or now
  if (now - a.hipHotSince) < HIP_HOT_S * 1000 then return end
  if now < a.nextHipHot then return end
  play("t_repos")
  play("w_hiphot")
  a.nextHipHot = now + 15000
  a.hipHotSince = now       -- restart the dwell, so it repeats every 15 s of hold
end

-- Wheel glitch rate. A bench log showed this climbing to 14.6% of samples
-- before a spurious runaway trip. Hearing it build gives you a chance to stop
-- before the fall instead of explaining the fall afterwards.
local function glitchWarn(a, now, count)
  if count == nil then return end
  if a.glitchLast == nil then a.glitchLast = count return end
  local delta = count - a.glitchLast
  a.glitchLast = count
  if delta <= 0 then return end
  if now < a.nextGlitch then return end
  play("t_stale")
  play("w_glitch")
  a.nextGlitch = now + 8000
end

local function battWarn(a, now, volts)
  if volts == nil or volts < 1.0 then return end   -- 0 V means "no sensor"
  if now < a.nextBatt then return end
  -- 6S pack. 24.0 V is the fully-charged working assumption for this robot,
  -- not the 22.2 V LiPo textbook nominal, so the thresholds sit accordingly.
  if volts <= 19.8 then
    play("t_siren"); play("w_batcrt"); haptic(60)
    a.nextBatt = now + 20000
  elseif volts <= 21.0 then
    play("t_repos"); play("w_batlow")
    a.nextBatt = now + 60000
  end
end

-- The main tick. `t` is the telem module, `now` is getTime()*10 in ms.
function M.tick(a, t, now)
  if not a.enabled then return end

  local linked = t.link and t.robotTelemetry()
  if a.linked == nil then
    a.linked = linked          -- first pass: adopt, do not announce
  elseif linked ~= a.linked then
    a.linked = linked
    play(linked and "t_ok" or "t_stale")
    play(linked and "o_conn" or "o_lost")
    if not linked then haptic(20) end
  end
  if not linked then return end

  local state = t.live("state", nil)
  local fault = t.live("fault", nil)

  -- State transitions get the canonical callout, matching the LED/Neopixel
  -- colour table so audio and light always agree about what the robot is doing.
  if state ~= nil then
    state = math.floor(state + 0.5)
    if a.state == nil then
      a.state = state
    elseif state ~= a.state then
      a.state = state
      play(a.def.state(state).wav)
    end
  end

  -- Faults. Announced once per distinct code, tiered by recovery.
  if fault ~= nil then
    fault = math.floor(fault + 0.5)
    if fault ~= a.fault then
      a.fault = fault
      if fault ~= 0 then
        local f = a.def.fault(fault)
        local tier = TIER[f.tier] or TIER.REBOOT
        play(tier.tone)
        haptic(tier.haptic)
        if tier.speak then play(f.wav, 5) end
      end
    end
  end

  -- "GUI went quiet" and "robot went quiet" are different problems and need to
  -- be distinguishable without a laptop.
  local esp = t.live("esp32", nil)
  if esp ~= nil then
    local ok = esp > 0.5
    if a.esp32 == nil then
      a.esp32 = ok
    elseif ok ~= a.esp32 then
      a.esp32 = ok
      if not ok then play("t_stale"); play("w_esp") end
    end
  end

  pitchWarn(a, now, t.live("pitch", nil))
  hipThermal(a, now, t.live("hip_l", nil), t.live("hip_r", nil))
  glitchWarn(a, now, t.live("glitch", nil))
  battWarn(a, now, t.live("pack_v", nil))
end

return M
