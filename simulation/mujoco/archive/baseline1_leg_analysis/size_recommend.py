"""size_recommend.py — Read force_log.csv peaks and recommend aluminium tubes + bearings.

Usage:
    python simulation/mujoco/baseline1_leg_analysis/size_recommend.py [path/to/force_log.csv]

If no path is given, reads force_log.csv in this directory.

Expected CSV columns (produced by viewer.py after the structural-logging update):
    time_s, pitch_deg, height_mm,
    fax_fem_N, flat_fem_N,
    fax_tib_N, flat_tib_N,
    fax_cpl_N,
    fbear_A_N, fbear_C_N, fbear_E_N, fbear_F_N, fbear_W_N,
    grf_L_N

Output:
    Prints a sizing report to stdout.
    Also saves size_report.txt in the same directory as the CSV.
"""

import csv
import math
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) >= 2:
    CSV_PATH = sys.argv[1]
else:
    CSV_PATH = os.path.join(_DIR, "force_log.csv")

REPORT_PATH = os.path.join(os.path.dirname(CSV_PATH), "size_report.txt")

# ── Material: 6061-T6 aluminium ───────────────────────────────────────────────
E_AL  = 69_000   # [MPa]  Young's modulus
SY_AL =    276   # [MPa]  yield strength (conservative)
RHO   =  2_700   # [kg/m³] density

# ── Safety factors ────────────────────────────────────────────────────────────
SF_LOAD      = 2.0   # multiplied onto sim peak before every sizing calc
SF_YIELD_MIN = 2.0   # minimum acceptable SF on yield after SF_LOAD applied
SF_BUCK_MIN  = 5.0   # minimum acceptable Euler-buckling SF
SF_BEAR_MIN  = 2.0   # minimum static bearing SF  s0 = C0 / F_peak

# ── Link geometry (from sim_config winning run) ───────────────────────────────
# L_mm: link length; F_ax_col / F_lat_col: CSV column names for peak forces
LINKS = {
    "femur": dict(
        L_mm     = 174,
        F_ax_col = "fax_fem_N",
        F_lat_col= "flat_fem_N",
        note     = "hip A → knee C, driven by AK45-10",
    ),
    "tibia": dict(
        L_mm     = 144,          # 129 mm shaft + 15 mm stub
        F_ax_col = "fax_tib_N",
        F_lat_col= "flat_tib_N",
        note     = "knee C → wheel W (plus 15 mm stub C → E)",
    ),
    "coupler": dict(
        L_mm     = 151,
        F_ax_col = "fax_cpl_N",
        F_lat_col= None,         # two-force member: lateral ≈ 0
        note     = "body pivot F → stub tip E  (4-bar closing link)",
    ),
}

# ── Bearing joints ────────────────────────────────────────────────────────────
BEAR_JOINTS = {
    "A": dict(col="fbear_A_N", note="hip pivot  (femur drive end)"),
    "C": dict(col="fbear_C_N", note="knee pivot (femur/tibia junction)"),
    "E": dict(col="fbear_E_N", note="4-bar closure  (stub tip / coupler end)"),
    "F": dict(col="fbear_F_N", note="coupler body pivot  (≈ E by symmetry)"),
    "W": dict(col="fbear_W_N", note="wheel axle"),
}

# ── Bearing catalog ───────────────────────────────────────────────────────────
# Sequence tried from smallest to largest; first one to pass SF_BEAR_MIN is chosen.
BEARING_CATALOG = [
    dict(key="608",  bore=8,  OD=22, width=7,  C0=1_370, C=3_450),
    dict(key="6000", bore=10, OD=26, width=8,  C0=1_960, C=4_500),
    dict(key="6001", bore=12, OD=28, width=8,  C0=2_850, C=5_580),
    dict(key="6200", bore=10, OD=30, width=9,  C0=2_850, C=5_100),
    dict(key="6201", bore=12, OD=32, width=10, C0=3_550, C=6_890),
    dict(key="6202", bore=15, OD=35, width=11, C0=4_500, C=7_800),
]

# ── Tube candidate list (OD mm, wall mm) sorted lightest first ────────────────
TUBE_CANDIDATES = [
    ( 8, 0.8), ( 8, 1.0),
    (10, 0.8), (10, 1.0), (10, 1.5),
    (12, 1.0), (12, 1.5), (12, 2.0),
    (14, 1.0), (14, 1.5), (14, 2.0),
    (16, 1.0), (16, 1.5), (16, 2.0),
    (18, 1.5), (18, 2.0),
    (20, 1.5), (20, 2.0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tube_props(OD, t):
    ID = OD - 2 * t
    A  = math.pi / 4  * (OD**2 - ID**2)
    I  = math.pi / 64 * (OD**4 - ID**4)
    return A, I   # [mm², mm⁴]


def _tube_check(OD, t, L_mm, F_ax_N, F_lat_N):
    """
    Return (sigma_MPa, SF_yield, SF_buck, mass_g) for a tube under
    combined axial + mid-span bending loads with SF_LOAD already applied.
    Returns None if wall is unrealistic (ID < 2 mm).
    """
    if OD - 2 * t < 2.0:
        return None
    A, I = _tube_props(OD, t)
    c    = OD / 2

    Fax_d = SF_LOAD * abs(F_ax_N)
    M_d   = SF_LOAD * abs(F_lat_N) * L_mm / 4   # pin-pin, midspan point load

    # Avoid division by zero if both loads are tiny
    if A < 1e-9 or I < 1e-9:
        return None

    sigma = Fax_d / A + M_d * c / I
    SF_y  = SY_AL / sigma if sigma > 0 else 999.0

    # Euler buckling (pin-pin, K=1)
    F_cr  = math.pi**2 * E_AL * I / L_mm**2
    SF_b  = F_cr / Fax_d if Fax_d > 1e-3 else 999.0

    mass  = RHO * (A * 1e-6) * (L_mm * 1e-3) * 1000   # grams
    return sigma, SF_y, SF_b, mass


def _recommend_tube(L_mm, F_ax_N, F_lat_N):
    """Return the lightest (OD, t) that satisfies both SF thresholds, or None."""
    passing = []
    for OD, t in TUBE_CANDIDATES:
        r = _tube_check(OD, t, L_mm, F_ax_N, F_lat_N)
        if r is None:
            continue
        sigma, SF_y, SF_b, mass = r
        if SF_y >= SF_YIELD_MIN and SF_b >= SF_BUCK_MIN:
            passing.append((mass, OD, t, sigma, SF_y, SF_b))
    if not passing:
        return None
    passing.sort(key=lambda x: x[0])
    mass, OD, t, sigma, SF_y, SF_b = passing[0]
    return OD, t, sigma, SF_y, SF_b, mass


def _recommend_bearing(F_peak_N):
    """Return first bearing in catalog where s0 = C0/F_peak ≥ SF_BEAR_MIN."""
    if F_peak_N <= 0:
        return BEARING_CATALOG[0], 999.0
    for b in BEARING_CATALOG:
        s0 = b["C0"] / F_peak_N
        if s0 >= SF_BEAR_MIN:
            return b, s0
    # Nothing passes — return largest with its actual s0
    b = BEARING_CATALOG[-1]
    return b, b["C0"] / F_peak_N


# ── Read CSV and find peaks ───────────────────────────────────────────────────
def _read_peaks(csv_path):
    """Return dict {col_name: peak_abs_value} from CSV."""
    peaks = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or has no header.")
        for col in reader.fieldnames:
            if col != "time_s":
                peaks[col] = 0.0
        for row in reader:
            for col in list(peaks.keys()):
                try:
                    v = abs(float(row[col]))
                    if v > peaks[col]:
                        peaks[col] = v
                except (ValueError, KeyError):
                    pass
    return peaks


# ── Report ────────────────────────────────────────────────────────────────────
def _build_report(peaks):
    lines = []

    def p(s=""):
        lines.append(s)

    rho = RHO
    _ = rho  # suppress unused warning

    p("=" * 72)
    p("  Structural Sizing Report — Wheeled-Leg Robot")
    p(f"  Source: {CSV_PATH}")
    p("=" * 72)
    p(f"  Material     : 6061-T6 aluminium  sy={SY_AL} MPa  E={E_AL} MPa")
    p(f"  Load SF      : {SF_LOAD}×  (simulation peaks × {SF_LOAD} before sizing)")
    p(f"  Min SF yield : {SF_YIELD_MIN}    Min SF buckling : {SF_BUCK_MIN}")
    p(f"  Min bearing s0 : {SF_BEAR_MIN}")
    p()

    # ── Peak loads table ──────────────────────────────────────────────────────
    p("  PEAK LOADS FROM SIMULATION")
    p("  " + "-" * 68)
    all_cols = (
        list(LINKS["femur"].values())
        + list(LINKS["tibia"].values())
        + list(LINKS["coupler"].values())
        + [j["col"] for j in BEAR_JOINTS.values()]
        + ["grf_L_N"]
    )
    # Collect just the string column names that are present in peaks
    load_cols = [
        ("fax_fem_N",  "Femur axial"),
        ("flat_fem_N", "Femur lateral"),
        ("fax_tib_N",  "Tibia axial"),
        ("flat_tib_N", "Tibia lateral"),
        ("fax_cpl_N",  "Coupler axial"),
        ("fbear_A_N",  "Bearing A  (hip)"),
        ("fbear_C_N",  "Bearing C  (knee)"),
        ("fbear_E_N",  "Bearing E  (4-bar)"),
        ("fbear_F_N",  "Bearing F  (coupler)"),
        ("fbear_W_N",  "Bearing W  (wheel)"),
        ("grf_L_N",    "GRF L  (wheel contact)"),
    ]
    missing_cols = []
    for col, label in load_cols:
        if col in peaks:
            p(f"  {label:<28s} {peaks[col]:>8.1f} N")
        else:
            missing_cols.append(col)
    if missing_cols:
        p()
        p("  [WARNING] Missing columns — run viewer.py to regenerate force_log.csv:")
        for c in missing_cols:
            p(f"    {c}")
    p()

    # ── Link sizing ───────────────────────────────────────────────────────────
    p("  LINK TUBE RECOMMENDATION  (6061-T6 aluminium, thin-walled round tube)")
    p("  " + "-" * 68)

    for link_name, linfo in LINKS.items():
        L_mm      = linfo["L_mm"]
        ax_col    = linfo["F_ax_col"]
        lat_col   = linfo["F_lat_col"]
        note      = linfo["note"]

        F_ax  = peaks.get(ax_col,  0.0)
        F_lat = peaks.get(lat_col, 0.0) if lat_col else 0.0

        p(f"  {link_name.upper()}  ({note})")
        p(f"    L = {L_mm} mm   F_ax_peak = {F_ax:.1f} N"
          + (f"   F_lat_peak = {F_lat:.1f} N" if lat_col else "  (two-force — no lateral)"))

        if F_ax < 0.1 and F_lat < 0.1:
            p("    [SKIP] No load data found for this link — column missing or zero.")
            p()
            continue

        rec = _recommend_tube(L_mm, F_ax, F_lat)
        if rec is None:
            p("    [FAIL] No candidate tube passed. Manual sizing required.")
        else:
            OD, t, sigma, SF_y, SF_b, mass = rec
            ID   = OD - 2 * t
            A, I = _tube_props(OD, t)
            p(f"    Recommended : OD {OD} × {t} mm wall  (ID = {ID:.0f} mm)")
            p(f"    sigma = {sigma:.0f} MPa   SF_yield = {SF_y:.2f}   SF_buck = {SF_b:.0f}")
            p(f"    Area = {A:.1f} mm²   I = {I:.0f} mm⁴   mass ≈ {mass:.1f} g/link")

        # Show next size up for margin context
        larger = []
        for OD2, t2 in TUBE_CANDIDATES:
            if rec and OD2 == rec[0] and t2 == rec[1]:
                continue
            r2 = _tube_check(OD2, t2, L_mm, F_ax, F_lat)
            if r2 and r2[1] >= SF_YIELD_MIN and r2[2] >= SF_BUCK_MIN:
                larger.append((r2[3], OD2, t2, r2[1], r2[2]))
        larger.sort(key=lambda x: x[0])
        # skip the one we already recommended
        candidates_display = [x for x in larger if rec is None or (x[1], x[2]) != (rec[0], rec[1])]
        if candidates_display and len(candidates_display) >= 1:
            m2, o2, t2b, sf2, sfb2 = candidates_display[0]
            p(f"    Next up     : OD {o2} × {t2b} mm  SF_y={sf2:.2f}  SF_b={sfb2:.0f}  m={m2:.1f} g")
        p()

    # ── Bearing sizing ────────────────────────────────────────────────────────
    p("  BEARING RECOMMENDATION  (all pivots currently sized as 608)")
    p("  Catalog: " + "  |  ".join(
        f"{b['key']} (bore {b['bore']}mm, C0={b['C0']}N)" for b in BEARING_CATALOG[:4]))
    p("  " + "-" * 68)

    for joint, jinfo in BEAR_JOINTS.items():
        col   = jinfo["col"]
        note  = jinfo["note"]
        F_pk  = peaks.get(col, 0.0)

        if F_pk < 0.1:
            status = "[NO DATA]"
            p(f"  Bearing {joint}  ({note})")
            p(f"    Peak load : N/A — column '{col}' missing or zero")
            p()
            continue

        b, s0 = _recommend_bearing(F_pk)
        flag  = "✓" if s0 >= SF_BEAR_MIN else "⚠ INSUFFICIENT"
        p(f"  Bearing {joint}  ({note})")
        p(f"    F_peak = {F_pk:.0f} N  →  {b['key']}  "
          f"(bore {b['bore']}mm, OD {b['OD']}mm, C0={b['C0']}N)  "
          f"s0={s0:.2f}  {flag}")
        p()

    # ── Summary table ─────────────────────────────────────────────────────────
    p("  SUMMARY")
    p("  " + "-" * 68)
    p(f"  {'Link':<12} {'Tube (OD×t)':<16} {'SF_y':>6} {'SF_b':>6} {'mass/link':>10}")
    p("  " + "-" * 52)
    for link_name, linfo in LINKS.items():
        L_mm  = linfo["L_mm"]
        F_ax  = peaks.get(linfo["F_ax_col"],  0.0)
        F_lat = peaks.get(linfo["F_lat_col"], 0.0) if linfo["F_lat_col"] else 0.0
        if F_ax < 0.1 and F_lat < 0.1:
            p(f"  {link_name:<12} {'— no data (missing column)':}")
            continue
        rec   = _recommend_tube(L_mm, F_ax, F_lat)
        if rec:
            OD, t, _, SF_y, SF_b, mass = rec
            p(f"  {link_name:<12} {OD}×{t} mm{'':<9} {SF_y:>6.2f} {SF_b:>6.0f} {mass:>8.1f} g")
        else:
            p(f"  {link_name:<12} {'— no recommendation':}")
    p()
    p(f"  {'Joint':<8} {'Bearing':<8} {'C0 [N]':>8} {'F_peak [N]':>11} {'s0':>6}")
    p("  " + "-" * 46)
    for joint, jinfo in BEAR_JOINTS.items():
        F_pk = peaks.get(jinfo["col"], 0.0)
        if F_pk > 0:
            b, s0 = _recommend_bearing(F_pk)
            p(f"  {joint:<8} {b['key']:<8} {b['C0']:>8} {F_pk:>11.0f} {s0:>6.2f}")
        else:
            p(f"  {joint:<8} {'n/a':<8} {'—':>8} {'—':>11} {'—':>6}")
    p()
    p("=" * 72)

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run viewer.py or run_sim.py first to generate force_log.csv.")
        sys.exit(1)

    print(f"Reading {CSV_PATH} ...")
    peaks = _read_peaks(CSV_PATH)

    report = _build_report(peaks)

    # Print safely on Windows (cp1252 terminals don't handle all Unicode)
    try:
        print(report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
