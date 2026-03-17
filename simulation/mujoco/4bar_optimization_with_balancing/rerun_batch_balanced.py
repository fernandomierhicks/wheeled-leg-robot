"""
rerun_batch_balanced.py — Re-run specific experiments from results_balanced.csv.
"""
import csv
import shutil
import os
import sys

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_jump_balanced import evaluate, CSV_PATH, CSV_COLS, DEFAULT

def row_to_params(row):
    p = DEFAULT.copy()
    def get_val(k, scale=1.0):
        v = row.get(k)
        return float(v) / scale if v and v.strip() else None

    for key in ['L_femur', 'L_stub', 'L_tibia', 'Lc', 'F_X', 'F_Z', 'A_Z']:
        val = get_val(f"{key}_mm", 1000.0)
        if val is not None: p[key] = val
            
    for key in ['m_box', 'm_femur', 'm_tibia', 'm_coupler', 'm_wheel']:
        val = get_val(f"{key}_g", 1000.0)
        if val is not None: p[key] = val
    return p

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found."); return

    backup_path = CSV_PATH + ".bak"
    shutil.copy(CSV_PATH, backup_path)
    print(f"Backed up CSV to {backup_path}")

    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        print("File is not UTF-8, reading with latin-1 as fallback to repair it.")
        with open(CSV_PATH, newline="", encoding="latin-1") as f:
            all_rows = list(csv.DictReader(f))

    targets = []
    run_all = len(sys.argv) == 1
    if not run_all:
        for arg in sys.argv[1:]:
            try: targets.append(int(arg))
            except ValueError: pass

    if run_all: print(f"Loaded {len(all_rows)} rows. Re-running ALL experiments.")
    else: print(f"Loaded {len(all_rows)} rows. Targets: {targets or 'None'}")

    updated_rows, processed_count = [], 0
    for row in all_rows:
        try: rid = int(row["run_id"])
        except (ValueError, KeyError): updated_rows.append(row); continue

        if run_all or rid in targets:
            print(f"\n--- Re-running Run ID {rid} [{row.get('label','')}] ---")
            p = row_to_params(row)
            new_result = evaluate(p, label=row.get('label', ''), forced_run_id=rid, write_csv=False)
            updated_rows.append(new_result)
            processed_count += 1
        else:
            updated_rows.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\nComplete. Updated {processed_count} rows in {CSV_PATH}. "
          f"File is now correctly encoded as UTF-8.")

if __name__ == "__main__":
    main()