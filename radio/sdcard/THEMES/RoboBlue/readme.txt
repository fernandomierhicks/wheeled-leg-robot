RoboBlue -- dark instrument theme for the wheeled-leg robot.

Near-black slate with a cyan-blue accent, tuned for reading robot state in
daylight. Companion to the WLRHUD widget, which mirrors this palette.

Colour roles, and what EdgeTX actually uses each for (the role names are not
self-explanatory; these follow EdgeTX's own Dark_Theme):

  PRIMARY1    0x39434F  slate -- borders, dividers, inactive chrome
  PRIMARY2    0xFFFFFF  white -- primary text on dark, AND the fill
                        colour of standard controls (buttons, fields)
  PRIMARY3    0xE2ECF7  near-white -- secondary text, units, labels
  SECONDARY1  0x121821  panel / header fill
  SECONDARY2  0x00A8FF  blue accent -- gauges, bars, selection fill
  SECONDARY3  0x070A0F  page background, the darkest surface
  FOCUS       0x0078D7  focus ring / selected row background
  EDIT        0x1B2735  edit-mode field background
  ACTIVE      0x00E5FF  cyan -- "this is live"
  WARNING     0xFF3B30  fault red
  DISABLED    0x8FA3B8  greyed-out controls
  QM_BG       0x0E141C  quick-menu background (EdgeTX 2.12+)
  QM_FG       0xFFFFFF  quick-menu foreground (EdgeTX 2.12+)

PRIMARY1 MUST STAY DARK. etx_lv_theme.cpp's etx_std_ctrl_colors() draws a
checked control as ACTIVE (cyan) background with PRIMARY1 text, so a white
PRIMARY1 makes every selected control unreadable. The same function fills a
normal control with PRIMARY2 and writes SECONDARY1 on it, which is why
PRIMARY2 has to stay light rather than becoming a dark "text" colour.

  state     background   text
  normal    PRIMARY2     SECONDARY1
  checked   ACTIVE       PRIMARY1
  edited    EDIT         PRIMARY2

SECONDARY3 is the page background and SECONDARY1 the panel fill. Getting those
backwards produces a theme that looks inverted.

theme.yml must contain NO comments. EdgeTX's YAML parser has no comment
support -- a '#' is read as an attribute name and parsing stops there, keeping
only what came before it. That is why this explanation lives in readme.txt,
which EdgeTX never parses.

All images are generated from one palette dict by radio/tools/make_assets.py,
so recolouring is a single edit.
