"""
rerun_batch.py — Re-run specific experiments from results.csv using current code.
"""
import csv
import shutil
import os
import sys

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_jump import evaluate, CSV_PATH, CSV_COLS, DEFAULT

def row_to_params(row):
    p = DEFAULT.copy()
    # Helper to parse float or keep default
    def get_val(k, scale=1.0):
        v = row.get(k)
        if v and v.strip():
            return float(v) / scale
        return None

    # Distance params (mm -> m)
    for key in ['L_femur', 'L_stub', 'L_tibia', 'Lc', 'F_X', 'F_Z', 'A_Z']:
        val = get_val(f"{key}_mm", 1000.0)
        if val is not None:
            p[key] = val
            
    # Mass params (g -> kg)
    for key in ['m_box', 'm_femur', 'm_tibia', 'm_coupler', 'm_wheel']:
        val = get_val(f"{key}_g", 1000.0)
        if val is not None:
            p[key] = val
    return p

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # 1. Backup
    backup_path = CSV_PATH + ".bak"
    shutil.copy(CSV_PATH, backup_path)
    print(f"Backed up CSV to {backup_path}")

    # 2. Read all rows
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
    except UnicodeDecodeError:
        print("File is not UTF-8, reading with latin-1 as fallback to repair it.")
        with open(CSV_PATH, newline="", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

    # Parse target IDs from CLI
    targets = []
    run_all = len(sys.argv) == 1
    if not run_all:
        for arg in sys.argv[1:]:
            try:
                targets.append(int(arg))
            except ValueError:
                pass  # ignore non-integer arguments

    if run_all:
        print(f"Loaded {len(all_rows)} rows. No IDs provided, re-running ALL experiments.")
    else:
        print(f"Loaded {len(all_rows)} rows. Targets: {targets if targets else 'None (no valid IDs provided)'}")

    # 3. Process
    updated_rows = []
    processed_count = 0
    
    for row in all_rows:
        try:
            rid = int(row["run_id"])
        except (ValueError, KeyError):
            updated_rows.append(row)
            continue

        if run_all or rid in targets:
            print(f"\n--- Re-running Run ID {rid} [{row.get('label','')}] ---")
            p = row_to_params(row)
            # Re-run using evaluate, but don't append to CSV
            new_result = evaluate(p, label=row.get('label', ''), 
                                  forced_run_id=rid, write_csv=False)
            updated_rows.append(new_result)
            processed_count += 1
        else:
            updated_rows.append(row)

    # 4. Write back
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\nComplete. Updated {processed_count} rows in {CSV_PATH}. "
          f"File is now correctly encoded as UTF-8.")

if __name__ == "__main__":
    main()