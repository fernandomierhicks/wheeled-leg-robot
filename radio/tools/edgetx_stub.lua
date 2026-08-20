-- Minimal EdgeTX colour-radio API stub, for exercising the WLRHUD widget off
-- the radio. Loaded by radio/tools/check_lua.py.
--
-- This is not an emulator. It exists to answer one question that syntax
-- checking cannot: "does every code path in the widget actually run without
-- indexing a nil or doing arithmetic on a string?" -- including the property
-- functions, which the firmware calls but a plain `load()` never does.
--
-- It is also strict where the firmware is strict: lua_lvgl_widget.cpp raises
-- "Invalid property '<key>'" for any unrecognised key, so this stub does the
-- same. A typo'd property is a script that dies on the radio, in the field.

local S = {}
S.errors = {}
S.objectCount = 0
S.propCalls = 0

local function fail(fmt, ...)
  S.errors[#S.errors + 1] = string.format(fmt, ...)
end

-- --------------------------------------------------------------------------
-- constants
-- --------------------------------------------------------------------------

LCD_W, LCD_H = 480, 320

XXLSIZE, DBLSIZE, MIDSIZE, SMLSIZE, BOLD = 1, 2, 3, 4, 5
LEFT, RIGHT, CENTER, VCENTER, VTOP, VBOTTOM = 0, 16, 32, 64, 128, 256

BLACK, WHITE, GREY, RED, GREEN, BLUE, YELLOW, ORANGE = 0, 1, 2, 3, 4, 5, 6, 7
COLOR_THEME_PRIMARY1, COLOR_THEME_PRIMARY2, COLOR_THEME_PRIMARY3 = 10, 11, 12
COLOR_THEME_SECONDARY1, COLOR_THEME_SECONDARY2, COLOR_THEME_SECONDARY3 = 13, 14, 15
COLOR_THEME_FOCUS, COLOR_THEME_EDIT, COLOR_THEME_ACTIVE = 16, 17, 18
COLOR_THEME_WARNING, COLOR_THEME_DISABLED = 19, 20

BOOL, COLOR, VALUE, SOURCE, STRING = 0, 1, 2, 3, 4

-- --------------------------------------------------------------------------
-- scalar API
-- --------------------------------------------------------------------------

lcd = {
  RGB = function(r, g, b)
    if g == nil then return r end
    return r * 65536 + g * 256 + b
  end,
  drawText = function() end,
  exitFullScreen = function() end,
}

-- Simulated telemetry. check_lua.py drives S.sensors / S.rssi / S.time.
S.sensors = {}
S.rssi = 0
S.time = 0
S.played = {}

local nextId = 100
local idOf, nameOf = {}, {}

function getFieldInfo(name)
  if S.sensors[name] == nil then return nil end
  if idOf[name] == nil then
    nextId = nextId + 1
    idOf[name] = nextId
    nameOf[nextId] = name
  end
  return { id = idOf[name], name = name, desc = name, unit = 0, prec = 1 }
end

function getValue(src)
  local name = type(src) == "number" and nameOf[src] or src
  if name == nil then return 0 end
  return S.sensors[name]
end

function getRSSI() return S.rssi end
function getTime() return S.time end

function playFile(f, v) S.played[#S.played + 1] = { file = f, vol = v } end
function playTone(f, d, p, fl, inc) S.played[#S.played + 1] = { tone = f } end
function playHaptic(d, p, fl) S.played[#S.played + 1] = { haptic = d } end
function playNumber() end

function loadScript(path, mode, env)
  local f = io.open(S.root .. path, "r")
  if not f then error("loadScript: no such file " .. path) end
  local src = f:read("*a")
  f:close()
  local chunk, err = load(src, path)
  if not chunk then error("loadScript: " .. tostring(err)) end
  return chunk
end

-- --------------------------------------------------------------------------
-- lvgl stub
-- --------------------------------------------------------------------------

-- Property sets copied from lua_lvgl_widget.cpp parseParam() chains.
local COMMON = {
  x = true, y = true, w = true, h = true, color = true, pos = true,
  size = true, visible = true, floating = true, children = true,
  type = true, name = true, opacity = true,
}

local BOX = {
  flexFlow = true, flexPad = true, scrollBar = true, scrollDir = true,
  scrolled = true, scrollTo = true, align = true, borderPad = true,
}

local function merge(...)
  local out = {}
  for _, t in ipairs({ ... }) do
    for k in pairs(t) do out[k] = true end
  end
  return out
end

local PROPS = {
  label     = merge(COMMON, { text = true, font = true, align = true }),
  box       = merge(COMMON, BOX),
  rectangle = merge(COMMON, BOX, { thickness = true, filled = true, rounded = true }),
  circle    = merge(COMMON, { thickness = true, filled = true, radius = true }),
  arc       = merge(COMMON, { thickness = true, radius = true, rounded = true,
                              startAngle = true, endAngle = true,
                              bgColor = true, bgOpacity = true,
                              bgStartAngle = true, bgEndAngle = true }),
  line      = merge(COMMON, { rounded = true, thickness = true, pts = true }),
  hline     = merge(COMMON, { rounded = true, thickness = true }),
  vline     = merge(COMMON, { rounded = true, thickness = true }),
  triangle  = merge(COMMON, { pts = true, filled = true }),
  image     = merge(COMMON, { file = true, fill = true }),
}

-- Property functions the firmware calls back into. Calling them is the whole
-- point of this harness, so any that blow up are caught here and not on a
-- radio strapped to your neck.
local FUNC_PROPS = {
  color = 1, pos = 2, size = 2, visible = 1, opacity = 1, text = 1,
  font = 1, align = 1, filled = 1, radius = 1, startAngle = 1,
  endAngle = 1, bgColor = 1, bgOpacity = 1, bgStartAngle = 1,
  bgEndAngle = 1, pts = 1, scrollTo = 2,
}

local Obj = {}
Obj.__index = Obj

local function validate(kind, settings)
  local allowed = PROPS[kind]
  if allowed == nil then
    fail("unknown lvgl object type %q", tostring(kind))
    return
  end
  for k in pairs(settings) do
    if not allowed[k] then
      fail("Invalid property %q on lvgl.%s", tostring(k), kind)
    end
  end
end

local function callProps(kind, settings)
  for k, v in pairs(settings) do
    if type(v) == "function" and FUNC_PROPS[k] then
      S.propCalls = S.propCalls + 1
      local ok, a, b = pcall(v)
      if not ok then
        fail("lvgl.%s property %q raised: %s", kind, k, tostring(a))
      elseif a == nil and k ~= "scrollTo" then
        fail("lvgl.%s property %q returned nil", kind, k)
      elseif FUNC_PROPS[k] == 2 and b == nil then
        fail("lvgl.%s property %q must return two values", kind, k)
      elseif k == "text" and type(a) ~= "string" and type(a) ~= "number" then
        fail("lvgl.%s text function returned %s", kind, type(a))
      elseif k == "pts" and type(a) ~= "table" then
        fail("lvgl.%s pts function returned %s", kind, type(a))
      end
    end
  end
end

local function make(kind, settings)
  settings = settings or {}
  validate(kind, settings)
  callProps(kind, settings)
  S.objectCount = S.objectCount + 1
  local o = setmetatable({ kind = kind, settings = settings }, Obj)
  if settings.children then
    for _, child in ipairs(settings.children) do
      make(child.type, child)
    end
  end
  return o
end

for kind in pairs(PROPS) do
  Obj[kind] = function(self, settings) return make(kind, settings) end
end

function Obj:build(list)
  local named = {}
  for _, item in ipairs(list) do
    if type(item) ~= "table" or item.type == nil then
      fail("build entry missing 'type'")
    else
      local o = make(item.type, item)
      if item.name then named[item.name] = o end
    end
  end
  return named
end

function Obj:set(settings) validate(self.kind, settings) end
function Obj:show() end
function Obj:hide() end
function Obj:clear() end

lvgl = setmetatable({
  LCD_SCALE = 1.0,
  PERCENT_SIZE = 10000,
  PAGE_BODY_HEIGHT = 272,
  UI_ELEMENT_HEIGHT = 32,
  PAD_TINY = 2, PAD_SMALL = 4, PAD_MEDIUM = 6, PAD_LARGE = 8,
  PAD_OUTLINE = 2, PAD_BORDER = 1,
  FLOW_ROW = 0, FLOW_COLUMN = 1,
  SCROLL_OFF = 0, SCROLL_HOR = 1, SCROLL_VER = 2, SCROLL_ALL = 3,
  clear = function() S.objectCount = 0 end,
  isFullScreen = function() return S.fullScreen == true end,
  isAppMode = function() return false end,
  exitFullScreen = function() end,
}, {})

for kind in pairs(PROPS) do
  lvgl[kind] = function(settings) return make(kind, settings) end
end
lvgl.build = function(list) return Obj.build(nil, list) end

-- Re-run every property function without rebuilding. Simulates the firmware's
-- per-frame callRefs() pass, which is when a stale-value bug actually bites.
S.objects = {}
local origMake = make
make = function(kind, settings)
  local o = origMake(kind, settings)
  S.objects[#S.objects + 1] = { kind = kind, settings = settings }
  return o
end
for kind in pairs(PROPS) do
  Obj[kind] = function(self, settings) return make(kind, settings) end
  lvgl[kind] = function(settings) return make(kind, settings) end
end

function S.callRefs()
  for _, rec in ipairs(S.objects) do
    callProps(rec.kind, rec.settings)
  end
end

function S.reset()
  S.objects = {}
  S.objectCount = 0
  S.played = {}
end

return S
