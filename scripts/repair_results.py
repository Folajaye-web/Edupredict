"""One-time maintenance utility for existing EduPredict databases.

Usage from project root:
    python scripts/repair_results.py

The utility:
1. creates a timestamped backup;
2. reports duplicate result keys;
3. retires duplicate rows while preserving snapshots in result_revisions;
4. keeps the latest current result;
5. enables the database unique guard.

It never runs automatically at application startup.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edupredict.db import SessionLocal, init_db, Result
from edupredict.results import backup_database, duplicate_result_groups, repair_duplicate_results


def main():
    init_db()
    backup = backup_database()
    with SessionLocal() as s:
        groups = duplicate_result_groups(s)
        print(f"Database backup: {backup}")
        print(f"Duplicate groups found: {len(groups)}")
        if not groups:
            print("No duplicate result keys found. No repair needed.")
            return
        for g in groups:
            print(g)
        answer = input("Proceed with repair, keeping the latest row in each duplicate group? [y/N]: ").strip().lower()
        if answer != "y":
            print("Repair cancelled. No result rows were changed.")
            return
        retired = repair_duplicate_results(s)
        remaining = duplicate_result_groups(s)
        print(f"Retired duplicate rows: {retired}")
        print(f"Duplicate groups remaining: {len(remaining)}")
        if remaining:
            raise SystemExit("Repair did not fully clear duplicate result keys.")
        print("Result integrity repair completed successfully.")


if __name__ == "__main__":
    main()
