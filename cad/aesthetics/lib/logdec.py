r"""Append a decision to DECISIONS.md and archive the board that was shown.

Boards in out/ are disposable and get overwritten when a round is re-run, so the
archived copy in decisions/ is the permanent record.  Call this EVERY time
Fernando answers a board:

    python lib/logdec.py <board.png> "round 1 - language" "A" "liked the facets,
        wants the keel busier"

or from a session:

    import logdec; logdec.record(board, "round 1 - language", "A", "comments...")
"""
import os, re, shutil, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

DECISIONS = os.path.join(paths.ROOT, "DECISIONS.md")
ARCHIVE = os.path.join(paths.ROOT, "decisions")


def _next_index():
    """Highest index used so far, +1.

    Scans DECISIONS.md as well as the archive: an entry logged without a board
    writes no png, so archive-only scanning hands the next entry the SAME index.
    Two phase-4 entries came out as "14" that way.
    """
    os.makedirs(ARCHIVE, exist_ok=True)
    n = 0
    for f in os.listdir(ARCHIVE):
        m = re.match(r"(\d+)_", f)
        if m:
            n = max(n, int(m.group(1)))
    if os.path.exists(DECISIONS):
        with open(DECISIONS, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"##\s+(\d+)\s", line)
                if m:
                    n = max(n, int(m.group(1)))
    return n + 1


def record(board, round_name, choice, comments="", spec=None):
    """Archive `board`, append an entry.  Returns the archived path."""
    i = _next_index()
    stem = re.sub(r"[^a-z0-9]+", "-", round_name.lower()).strip("-")
    dest = os.path.join(ARCHIVE, f"{i:02d}_{stem}.png")
    if board and os.path.exists(board):
        shutil.copy2(board, dest)
    rel = os.path.relpath(dest, paths.ROOT).replace("\\", "/")
    today = datetime.date.today().isoformat()

    lines = [f"\n## {i:02d} · {round_name}", f"*{today}*", ""]
    if board and os.path.exists(board):
        lines += [f"![{round_name}]({rel})", ""]
    lines += [f"**Chose:** {choice}", ""]
    if comments:
        lines += [f"**Said:** {comments}", ""]
    if spec:
        keep = ("language", "aggression", "density", "relief", "palette")
        bits = ", ".join(f"{k}={spec[k]}" for k in keep if k in spec)
        lines += [f"**Spec:** `{bits}`", ""]
    lines.append("---")

    if not os.path.exists(DECISIONS):
        open(DECISIONS, "w", encoding="utf-8").write(
            "# Decision log\n\nEvery board shown, what was chosen, and why. "
            "Newest at the bottom.\nWritten by `lib/logdec.py` -- append, never rewrite.\n\n---\n")
    with open(DECISIONS, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"logged {i:02d} · {round_name} -> {rel}")
    return dest


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    record(sys.argv[1], sys.argv[2], sys.argv[3],
           sys.argv[4] if len(sys.argv) > 4 else "")
