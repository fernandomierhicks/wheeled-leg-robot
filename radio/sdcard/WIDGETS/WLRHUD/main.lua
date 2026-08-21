-- WLRHUD -- wheeled-leg robot heads-up display for the RadioMaster TX15.
--
-- Read-only instrument. It draws robot state, attitude, leg gain-schedule,
-- torques and link health, and speaks faults out loud. It writes NOTHING to
-- the robot: arm, disarm, ESTOP and the rescue combo live on physical
-- channels and firmware interlocks, where a crashed or exited Lua script
-- cannot reach them.
--
-- Layout adapts to the widget zone. Full screen on a TX15 is 480x320; drop it
-- into a small zone and it degrades to a state chip plus attitude readout.
-- Double-tap (or ENTER -> Full screen) to expand.
--
-- Requires EdgeTX 2.11+ for the LVGL Lua API. On anything older `lvgl` is nil
-- and the widget says so rather than failing silently.

local DIR = "/WIDGETS/WLRHUD/"

local def    = assert(loadScript(DIR .. "robotdef.lua"))()
local telem  = assert(loadScript(DIR .. "telem.lua"))()
local annunc = assert(loadScript(DIR .. "annunc.lua"))()

-- Palette. Mirrors THEMES/RoboBlue/theme.yml so the widget looks native under
-- that theme, but it is self-contained -- the HUD is legible under any theme.
local C = {
  bg      = lcd.RGB(0x070A0F),
  panel   = lcd.RGB(0x121821),
  track   = lcd.RGB(0x0D121A),
  border  = lcd.RGB(0x39434F),
  text    = lcd.RGB(0xE2ECF7),
  dim     = lcd.RGB(0x8FA3B8),
  accent  = lcd.RGB(0x00A8FF),
  cyan    = lcd.RGB(0x00E5FF),
  ok      = lcd.RGB(0x2EE68A),
  amber   = lcd.RGB(0xFFB400),
  warn    = lcd.RGB(0xFF3B30),
  off     = lcd.RGB(0x4A5A6B),
}

-- Scale factors for readings that drive a bar. These are display ranges, not
-- limits -- a bar that pins just means "at the top of what this HUD plots".
local FS = {
  wheel   = 2.0,    -- m/s, radio_vel_max tops out at 2.0
  hip_nm  = 8.0,    -- N.m, jump_torque_max ceiling
  pitch   = 40.0,   -- degrees either side
  roll    = 30.0,   -- degrees either side
  pack_lo = 18.0,   -- 6S empty, for the pack gauge
  pack_hi = 25.2,   -- 6S full
}

local options = {
  { "Sounds", BOOL, 1 },
  { "Accent", COLOR, COLOR_THEME_SECONDARY2 },
}

-- ---------------------------------------------------------------------------
-- formatting helpers
-- ---------------------------------------------------------------------------

local DASH = "--"

local function num(key, pattern, scale)
  return function()
    local v, st = telem.get(key)
    if st == telem.MISSING or v == nil then return DASH end
    return string.format(pattern, v * (scale or 1))
  end
end

-- Colour a reading by whether it can be trusted: live is bright, stale is
-- amber, missing is grey. You should never have to wonder whether the number
-- in front of you is current.
local function valColour(key, live)
  return function()
    local _, st = telem.get(key)
    if st == telem.MISSING then return C.off end
    if st == telem.STALE then return C.amber end
    return live or C.text
  end
end

local function frac(key, full, offset)
  return function()
    local v, st = telem.get(key)
    if st == telem.MISSING or v == nil then return 0 end
    local f = (math.abs(v) - (offset or 0)) / full
    if f < 0 then return 0 end
    if f > 1 then return 1 end
    return f
  end
end

local function stateOf()
  local v, st = telem.get("state")
  if st == telem.MISSING or v == nil then return nil end
  return math.floor(v + 0.5)
end

local function faultOf()
  local v, st = telem.get("fault")
  if st == telem.MISSING or v == nil then return 0 end
  return math.floor(v + 0.5)
end

local function inFault()
  if faultOf() ~= 0 then return true end
  return telem.faultFromMode() ~= nil
end

local function noRobot()
  return not telem.robotTelemetry()
end

-- State and fault reach the radio twice: as numbers in the private 0x24 frame,
-- and as text in FLIGHT_MODE. The text is the one guaranteed to relay, so it
-- is the fallback whenever the private frame is absent -- which is the
-- expected state until that relay is confirmed on hardware.
local function stateLabel()
  local s = stateOf()
  if s ~= nil then return def.state(s).short end
  local name = telem.stateNameFromMode()
  if name ~= nil then return name end
  local f = telem.faultFromMode()
  if f ~= nil then return "ESTOP" end
  return "NO LINK"
end

local function stateColour()
  local s = stateOf()
  if inFault() then return C.warn end
  if s ~= nil then return lcd.RGB(def.state(s).colour) end
  local name = telem.stateNameFromMode()
  if name ~= nil then
    for id, entry in pairs(def.states) do
      if entry.name == name then return lcd.RGB(entry.colour) end
    end
    return C.text
  end
  return C.off
end

-- Fault name for display: the numeric code when we have it, otherwise the
-- short name the firmware put in FLIGHT_MODE.
local function faultLabel()
  local c = faultOf()
  if c ~= 0 then return def.fault(c).name end
  return telem.faultFromMode() or "NONE"
end

-- ---------------------------------------------------------------------------
-- layout
-- ---------------------------------------------------------------------------

local PAD = 6
local HDR = 34

local function layout(w, h)
  local L = { w = w, h = h, pad = PAD }
  L.full = (w >= 400 and h >= 240)
  L.mid = (not L.full) and (w >= 220 and h >= 120)

  if L.full then
    L.hdr = HDR
    local body = h - HDR - PAD * 4
    L.rowA = { y = HDR + PAD, h = math.floor(body * 0.44) }
    L.rowB = { y = L.rowA.y + L.rowA.h + PAD, h = math.floor(body * 0.37) }
    L.rowC = { y = L.rowB.y + L.rowB.h + PAD,
               h = h - (L.rowB.y + L.rowB.h + PAD) - PAD }
    L.colA = { x = PAD, w = math.floor(w * 0.30) }
    L.colB = { x = PAD * 2 + L.colA.w, w = math.floor(w * 0.40) }
    L.colC = { x = L.colB.x + L.colB.w + PAD }
    L.colC.w = w - L.colC.x - PAD
  elseif L.mid then
    L.hdr = 26
    L.rowA = { y = L.hdr + 4, h = h - L.hdr - 8 }
  end
  return L
end

-- ---------------------------------------------------------------------------
-- pieces. Each returns a build-table fragment, kept small on purpose: very
-- large or deeply nested lvgl.build tables can work from .lua and fail when
-- compiled to .luac.
-- ---------------------------------------------------------------------------

local function stateChip(x, y, w, h)
  return {
    { type = "rectangle", x = x, y = y, w = w, h = h, rounded = math.floor(h / 2),
      thickness = 2, filled = false,
      color = stateColour },
    { type = "label", x = x, y = y + math.floor((h - 16) / 2), w = w,
      align = CENTER, font = BOLD,
      text = stateLabel,
      color = stateColour },
  }
end

-- Attitude ball. Pitch and roll are THE state variables for a balancing
-- robot, so they get the largest single element and a horizon that moves the
-- way the robot does.
local function attitude(p, cx, cy, r)
  p:circle({ x = cx, y = cy, radius = r, filled = true, color = C.track })
  p:circle({ x = cx, y = cy, radius = r, thickness = 2, color = C.border })

  -- Horizon: rotated by roll, translated by pitch.
  p:line({
    thickness = 3, rounded = true, color = C.cyan,
    pts = function()
      local pitch = telem.live("pitch", 0) or 0
      local roll = telem.live("roll", 0) or 0
      local a = math.rad(roll)
      local ca, sa = math.cos(a), math.sin(a)
      local off = -pitch * (r / FS.pitch)
      local px, py = cx - sa * off, cy + ca * off
      local ln = r * 0.82
      return { { px - ca * ln, py - sa * ln }, { px + ca * ln, py + sa * ln } }
    end })

  -- Fixed aircraft-style reference marks: these are the RADIO's idea of level,
  -- so the gap between them and the horizon is the robot's lean at a glance.
  p:line({ thickness = 2, color = C.amber,
           pts = { { cx - r * 0.42, cy }, { cx - r * 0.14, cy } } })
  p:line({ thickness = 2, color = C.amber,
           pts = { { cx + r * 0.14, cy }, { cx + r * 0.42, cy } } })
  p:circle({ x = cx, y = cy, radius = 2, filled = true, color = C.amber })
end

local function bar(p, x, y, w, label, key, pattern, full, colour, scale)
  local BH = 9
  p:label({ x = x, y = y, font = SMLSIZE, text = label, color = C.dim })
  p:label({ x = x, y = y, w = w, align = RIGHT, font = SMLSIZE,
            text = num(key, pattern, scale), color = valColour(key) })
  p:rectangle({ x = x, y = y + 15, w = w, h = BH, rounded = 3,
                filled = true, color = C.track })
  local f = frac(key, full)
  p:rectangle({ x = x, y = y + 15, h = BH, rounded = 3, filled = true,
                w = 4, color = colour,
                size = function() return math.max(4, math.floor(w * f())), BH end })
end

-- ---------------------------------------------------------------------------
-- the three layouts
-- ---------------------------------------------------------------------------

local function buildHeader(root, L)
  local w = L.w
  root:rectangle({ x = 0, y = 0, w = w, h = L.hdr, filled = true, color = C.panel })
  root:line({ thickness = 2, color = C.accent,
              pts = { { 0, L.hdr }, { w, L.hdr } } })

  local chipW = math.min(96, math.floor(w * 0.24))
  root:build(stateChip(PAD, math.floor((L.hdr - 22) / 2), chipW, 22))

  if L.full then
    root:label({ x = PAD + chipW + 10, y = 9, font = BOLD, text = "WLR",
                 color = C.text })
    root:label({ x = PAD + chipW + 46, y = 11, font = SMLSIZE,
                 text = function()
                   local p = telem.live("profile", nil)
                   if p == nil then return "" end
                   return string.format("PROFILE %d", math.floor(p + 0.5) + 1)
                 end, color = C.dim })
  end

  -- Link and pack live top-right: the two things worth a glance every few
  -- seconds without reading anything else.
  root:label({ x = w - PAD - 150, y = 10, w = 74, align = RIGHT, font = SMLSIZE,
               text = function()
                 local lq, st = telem.get("lq")
                 if st == telem.MISSING or lq == nil then return "LQ --" end
                 return string.format("LQ %d", math.floor(lq + 0.5))
               end,
               color = function()
                 local lq, st = telem.get("lq")
                 if st ~= telem.LIVE or lq == nil then return C.off end
                 if lq < 50 then return C.warn end
                 if lq < 80 then return C.amber end
                 return C.ok
               end })
  root:label({ x = w - PAD - 72, y = 9, w = 72, align = RIGHT, font = BOLD,
               text = num("pack_v", "%.1fV"),
               color = function()
                 local v, st = telem.get("pack_v")
                 if st == telem.MISSING or v == nil then return C.off end
                 if st == telem.STALE then return C.amber end
                 if v <= 19.8 then return C.warn end
                 if v <= 21.0 then return C.amber end
                 return C.ok
               end })
end

local function buildRowA(root, L)
  local A, cA, cB, cC = L.rowA, L.colA, L.colB, L.colC

  -- attitude panel
  local pa = root:box({ x = cA.x, y = A.y, w = cA.w, h = A.h, scrollBar = false })
  pa:rectangle({ x = 0, y = 0, w = cA.w, h = A.h, rounded = 6, filled = true,
                 color = C.panel })
  local r = math.floor(math.min(cA.w, A.h - 22) / 2) - 6
  attitude(pa, math.floor(cA.w / 2), math.floor((A.h - 16) / 2), r)
  pa:label({ x = 0, y = A.h - 16, w = cA.w, align = CENTER, font = SMLSIZE,
             text = function()
               local p, ps = telem.get("pitch")
               local rl = telem.get("roll")
               if ps == telem.MISSING then return "PITCH --  ROLL --" end
               return string.format("PITCH %.1f  ROLL %.1f", p or 0, rl or 0)
             end,
             color = valColour("pitch", C.dim) })

  -- leg / gain-schedule panel. gain_sched_alpha is the number that explains
  -- most surprising robot behaviour, so it gets the big readout.
  local pb = root:box({ x = cB.x, y = A.y, w = cB.w, h = A.h, scrollBar = false })
  pb:rectangle({ x = 0, y = 0, w = cB.w, h = A.h, rounded = 6, filled = true,
                 color = C.panel })
  pb:label({ x = 10, y = 8, font = SMLSIZE, text = "LEG  alpha", color = C.dim })
  pb:label({ x = 0, y = 22, w = cB.w - 10, align = RIGHT, font = DBLSIZE,
             text = num("alpha", "%.2f"), color = valColour("alpha", C.cyan) })
  pb:rectangle({ x = 10, y = 56, w = cB.w - 20, h = 12, rounded = 3,
                 filled = true, color = C.track })
  local fa = frac("alpha", 1.0)
  pb:rectangle({ x = 10, y = 56, h = 12, rounded = 3, filled = true, w = 4,
                 color = C.cyan,
                 size = function()
                   return math.max(4, math.floor((cB.w - 20) * fa())), 12
                 end })
  pb:label({ x = 10, y = 74, font = SMLSIZE, text = "RET", color = C.off })
  pb:label({ x = 0, y = 74, w = cB.w - 10, align = RIGHT, font = SMLSIZE,
             text = "EXT", color = C.off })
  -- Phase line: only meaningful while jumping or standing up, so it hides
  -- itself the rest of the time instead of showing a stale "CROUCH".
  pb:label({ x = 10, y = A.h - 18, font = SMLSIZE, color = C.amber,
             visible = function()
               local s = stateOf()
               return s == 7 or s == 8
             end,
             text = function()
               local s = stateOf()
               if s == 7 then
                 local j = telem.live("jump", nil)
                 return "JUMP " .. (def.jumpPhase[math.floor((j or 0) + 1.5)] or "?")
               end
               local u = telem.live("standup", nil)
               return "STANDUP " .. (def.standupPhase[math.floor((u or 0) + 1.5)] or "?")
             end })

  -- pack gauge
  local pc = root:box({ x = cC.x, y = A.y, w = cC.w, h = A.h, scrollBar = false })
  pc:rectangle({ x = 0, y = 0, w = cC.w, h = A.h, rounded = 6, filled = true,
                 color = C.panel })
  local gr = math.floor(math.min(cC.w, A.h - 26) / 2) - 4
  local gcx, gcy = math.floor(cC.w / 2), math.floor((A.h - 18) / 2)
  -- 270 degree sweep opening at the bottom: 135 -> 405 in LVGL's clockwise,
  -- 3-o'clock-is-zero convention.
  pc:arc({ x = gcx, y = gcy, radius = gr, thickness = 8, rounded = true,
           startAngle = 135, endAngle = 405,
           color = C.track, bgColor = C.track, bgOpacity = 0 })
  local fp = function()
    local v, st = telem.get("pack_v")
    if st == telem.MISSING or v == nil then return 0 end
    local f = (v - FS.pack_lo) / (FS.pack_hi - FS.pack_lo)
    if f < 0 then return 0 end
    if f > 1 then return 1 end
    return f
  end
  pc:arc({ x = gcx, y = gcy, radius = gr, thickness = 8, rounded = true,
           startAngle = 135,
           endAngle = function() return 135 + math.floor(270 * fp()) end,
           color = function()
             local v, st = telem.get("pack_v")
             if st ~= telem.LIVE or v == nil then return C.off end
             if v <= 19.8 then return C.warn end
             if v <= 21.0 then return C.amber end
             return C.ok
           end,
           bgColor = C.track, bgOpacity = 0 })
  pc:label({ x = 0, y = gcy - 8, w = cC.w, align = CENTER, font = BOLD,
             text = num("pack_v", "%.1f"), color = valColour("pack_v") })
  -- Bus current is not instrumented, so the pack panel shows percentage
  -- instead of a current that would always read zero.
  pc:label({ x = 0, y = gcy + 8, w = cC.w, align = CENTER, font = SMLSIZE,
             text = num("pack_pct", "%.0f%%"), color = C.dim })
  pc:label({ x = 0, y = A.h - 16, w = cC.w, align = CENTER, font = SMLSIZE,
             text = "PACK", color = C.dim })
end

local function buildRowB(root, L)
  local B = L.rowB
  local p = root:box({ x = PAD, y = B.y, w = L.w - PAD * 2, h = B.h,
                       scrollBar = false })
  p:rectangle({ x = 0, y = 0, w = L.w - PAD * 2, h = B.h, rounded = 6,
                filled = true, color = C.panel })
  local half = math.floor((L.w - PAD * 2 - 36) / 2)
  local y0, y1 = 10, math.floor(B.h / 2) + 4
  bar(p, 12, y0, half, "WHEEL VEL", "wheel", "%.2f m/s", FS.wheel, C.accent)
  bar(p, 24 + half, y0, half, "PITCH", "pitch", "%.1f deg", FS.pitch, C.cyan)
  bar(p, 12, y1, half, "HIP TQ L", "hip_l", "%.2f Nm", FS.hip_nm, C.cyan)
  bar(p, 24 + half, y1, half, "HIP TQ R", "hip_r", "%.2f Nm", FS.hip_nm, C.cyan)
end

-- Health chips. Each is a bit out of health_flags, and each is a thing that
-- can be wrong without the robot being in ESTOP yet.
local HEALTH_ROW = {
  { "IMU",  "IMU_NOMINAL",      true },
  { "HIPL", "HIP_L_OK",         true },
  { "HIPR", "HIP_R_OK",         true },
  { "WHLL", "WM_L_OK",          true },
  { "WHLR", "WM_R_OK",          true },
  { "CAL",  "HIP_LIMITS_VALID", true },
  { "LQR",  "LQR_ACTIVE",       false },
  { "OVR",  "LOOP_OVERRUN",     false },
}

local function buildRowC(root, L)
  local Cr = L.rowC
  local p = root:box({ x = PAD, y = Cr.y, w = L.w - PAD * 2, h = Cr.h,
                       scrollBar = false })
  local n = #HEALTH_ROW
  local cw = math.floor((L.w - PAD * 2 - 92) / n)
  for i, item in ipairs(HEALTH_ROW) do
    local label, flagName, isGood = item[1], item[2], item[3]
    local mask = def.health[flagName].bit
    local x = (i - 1) * cw
    local colour = function()
      local flags, st = telem.get("health")
      if st == telem.MISSING or flags == nil then return C.off end
      local set = telem.bit(math.floor(flags + 0.5), mask)
      if isGood then return set and C.ok or C.warn end
      -- OVR (loop overrun) is bad when set; LQR is just informational.
      if flagName == "LOOP_OVERRUN" then return set and C.warn or C.off end
      return set and C.accent or C.off
    end
    p:rectangle({ x = x, y = 2, w = cw - 4, h = Cr.h - 8, rounded = 4,
                  thickness = 1, filled = false, color = colour })
    p:label({ x = x, y = math.floor((Cr.h - 8) / 2) - 4, w = cw - 4,
              align = CENTER, font = SMLSIZE, text = label, color = colour })
  end

  -- "GUI went quiet" vs "robot went quiet" -- distinguishable in the field.
  local ex = n * cw + 4
  p:label({ x = ex, y = 4, font = SMLSIZE, text = "ESP32",
            color = function()
              local v, st = telem.get("esp32")
              if st == telem.MISSING or v == nil then return C.off end
              return v > 0.5 and C.ok or C.warn
            end })
  p:label({ x = ex, y = math.floor(Cr.h / 2) + 2, font = SMLSIZE,
            text = function()
              local g, st = telem.get("glitch")
              if st == telem.MISSING or g == nil then return "GLCH --" end
              return string.format("GLCH %d", math.floor(g + 0.5))
            end,
            color = valColour("glitch", C.dim) })
end

-- Fault banner. Overlays the top row, because in ESTOP nothing above it
-- matters more than which fault fired and what it wants you to do.
local RECOVERY = {
  SOFT       = "reset from the GUI or rescue combo",
  REPOSITION = "pick it up, level it, then reset",
  GUI_FIX    = "fix the param in the GUI first",
  REBOOT     = "power-cycle -- reset will not clear it",
}

local function buildFaultBanner(root, L)
  local A = L.rowA
  local w = L.w - PAD * 2
  local p = root:box({ x = PAD, y = A.y, w = w, h = A.h, scrollBar = false,
                       visible = inFault })
  p:rectangle({ x = 0, y = 0, w = w, h = A.h, rounded = 6, filled = true,
                color = lcd.RGB(0x250C0C) })
  p:rectangle({ x = 0, y = 0, w = w, h = A.h, rounded = 6, thickness = 2,
                filled = false, color = C.warn })
  p:label({ x = 14, y = 10, font = BOLD, color = C.warn,
            text = faultLabel })
  p:label({ x = 14, y = 32, w = w - 28, font = SMLSIZE, color = C.text,
            text = function()
              local d = def.fault(faultOf()).desc or ""
              if #d > 62 then d = string.sub(d, 1, 59) .. "..." end
              return d
            end })
  p:label({ x = 14, y = A.h - 40, font = SMLSIZE, color = C.amber,
            text = function() return def.fault(faultOf()).tier end })
  p:label({ x = 14, y = A.h - 22, w = w - 28, font = SMLSIZE, color = C.dim,
            text = function()
              return RECOVERY[def.fault(faultOf()).tier] or ""
            end })
end

-- Shown until the robot's own telemetry frames exist. Honest about the gap
-- rather than drawing a HUD full of zeroes that look like real readings.
local function buildWaiting(root, L)
  local y = L.rowA and L.rowA.y or (L.hdr + PAD)
  local h = L.rowB and (L.rowB.y + L.rowB.h - y) or (L.h - y - PAD)
  local w = L.w - PAD * 2
  local p = root:box({ x = PAD, y = y, w = w, h = h, scrollBar = false,
                       visible = noRobot })
  p:rectangle({ x = 0, y = 0, w = w, h = h, rounded = 6, filled = true,
                color = C.panel })
  p:label({ x = 0, y = math.floor(h / 2) - 26, w = w, align = CENTER,
            font = BOLD, text = "NO ROBOT TELEMETRY", color = C.amber })
  p:label({ x = 0, y = math.floor(h / 2) - 2, w = w, align = CENTER,
            font = SMLSIZE, color = C.dim,
            text = "waiting for the CRSF telemetry frames" })
  p:label({ x = 0, y = math.floor(h / 2) + 16, w = w, align = CENTER,
            font = SMLSIZE, color = C.off,
            text = function()
              return telem.link and "link up, robot fields absent"
                                 or "no link"
            end })
end

-- One label pair: dim caption on the left, value right-aligned in the cell.
local function readout(p, x, y, w, caption, key, pattern, live)
  p:label({ x = x, y = y, font = SMLSIZE, text = caption, color = C.dim })
  p:label({ x = x, y = y, w = w, align = RIGHT, font = BOLD,
            text = num(key, pattern), color = valColour(key, live) })
end

-- Compact layout for anything between a tile and the full screen. A
-- half-screen zone on this radio is 480 wide but only ~150 tall: it has plenty
-- of room for four readings side by side, so it gets columns rather than the
-- same narrow stack a quarter-screen zone needs.
local COMPACT_ROWS = {
  { "PITCH", "pitch",  "%.1f",   nil },
  { "alpha", "alpha",  "%.2f",   nil },
  { "PACK",  "pack_v", "%.1fV",  nil },
  { "WHEEL", "wheel",  "%.2f",   nil },
  { "HIP L", "hip_l",  "%.1f",   nil },
  { "HIP R", "hip_r",  "%.1f",   nil },
}

local function buildCompact(root, L)
  root:rectangle({ x = 0, y = 0, w = L.w, h = L.hdr, filled = true,
                   color = C.panel })
  root:line({ thickness = 1, color = C.accent,
              pts = { { 0, L.hdr }, { L.w, L.hdr } } })
  root:build(stateChip(4, 3, math.min(84, L.w - 8), L.hdr - 6))

  -- Link and pack sit in the header whenever there is room for them.
  if L.w >= 300 then
    root:label({ x = 0, y = 5, w = L.w - 6, align = RIGHT, font = SMLSIZE,
                 text = function()
                   local v, st = telem.get("pack_v")
                   local lq = telem.get("lq")
                   if st == telem.MISSING then return "no telemetry" end
                   return string.format("LQ %d   %.1fV",
                                        math.floor((lq or 0) + 0.5), v or 0)
                 end,
                 color = valColour("pack_v", C.dim) })
  end

  local A = L.rowA
  local cols = (L.w >= 360) and 2 or 1
  local rows = math.min(#COMPACT_ROWS,
                        cols * math.max(1, math.floor((A.h - 18) / 20)))
  local cw = math.floor((L.w - 8 - (cols - 1) * 10) / cols)

  for i = 1, rows do
    local item = COMPACT_ROWS[i]
    local col = (i - 1) % cols
    local row = math.floor((i - 1) / cols)
    readout(root, 4 + col * (cw + 10), A.y + row * 20, cw,
            item[1], item[2], item[3], item[4])
  end

  -- The fault name always gets the bottom line, whatever else got dropped.
  root:label({ x = 4, y = L.h - 16, w = L.w - 8, font = SMLSIZE,
               color = C.warn, visible = inFault,
               text = faultLabel })
  root:label({ x = 4, y = L.h - 16, w = L.w - 8, font = SMLSIZE,
               color = C.amber, visible = noRobot,
               text = "no robot telemetry" })
end

-- Smallest zone: just the state, in the state's own colour.
local function buildTiny(root, L)
  root:build(stateChip(2, math.floor(L.h / 2) - 12, L.w - 4, 24))
end

-- ---------------------------------------------------------------------------
-- widget interface
-- ---------------------------------------------------------------------------

local function create(zone, opts)
  local wgt = {
    zone = zone,
    options = opts,
    ann = annunc.new(def),
    lastTick = 0,
  }
  return wgt
end

local function tick(wgt)
  -- getTime() is in 10 ms units; ms everywhere else in this widget.
  local now = getTime() * 10
  if now - wgt.lastTick < 100 then return end   -- 10 Hz is plenty
  wgt.lastTick = now
  telem.tick(now)
  wgt.ann.enabled = (wgt.options.Sounds ~= 0)
  annunc.tick(wgt.ann, telem, now)
end

local function update(wgt, opts)
  wgt.options = opts
  if lvgl == nil then return end

  lvgl.clear()

  local zone = wgt.zone
  local w = lvgl.isFullScreen and lvgl.isFullScreen() and LCD_W or zone.w
  local h = lvgl.isFullScreen and lvgl.isFullScreen() and LCD_H or zone.h
  local x = (w == zone.w) and zone.x or 0
  local y = (h == zone.h) and zone.y or 0

  local L = layout(w, h)

  -- One root box positioned over the zone, so every child coordinate below is
  -- zone-relative and the same layout code works full screen and in a tile.
  local root = lvgl.box({ x = x, y = y, w = w, h = h, scrollBar = false })
  root:rectangle({ x = 0, y = 0, w = w, h = h, filled = true, color = C.bg })

  if L.full then
    buildHeader(root, L)
    buildRowA(root, L)
    buildRowB(root, L)
    buildRowC(root, L)
    buildFaultBanner(root, L)
    buildWaiting(root, L)
  elseif L.mid then
    buildCompact(root, L)
  else
    buildTiny(root, L)
  end
end

local function refresh(wgt, event, touchState)
  if lvgl == nil then
    lcd.drawText(wgt.zone.x + 4, wgt.zone.y + 4,
                 "WLRHUD needs EdgeTX 2.11+", COLOR_THEME_WARNING)
    return
  end
  tick(wgt)
end

-- Runs when the HUD is not the visible screen. Annunciation must keep working
-- there: the whole point is that you are looking at the robot, not the radio.
local function background(wgt)
  tick(wgt)
end

return {
  name = "WLR HUD",
  options = options,
  create = create,
  update = update,
  refresh = refresh,
  background = background,
  useLvgl = true,
}
