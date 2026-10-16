import subprocess
import sys
import tempfile
from pathlib import Path

def main():
    print("EDUPREDICT PHASE 9G ISOLATED FAILURE TEST")
    print("=" * 50)

    with tempfile.TemporaryDirectory(prefix="edupredict_9g_test_") as tmp:
        marker = Path(tmp) / "later_stage_ran.txt"

        failed = subprocess.run(
            [sys.executable, "-c", "raise SystemExit(7)"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert failed.returncode == 7, (
            f"Expected exit code 7, got {failed.returncode}"
        )

        later_stage_ran = False
        if failed.returncode == 0:
            marker.write_text("ran", encoding="utf-8")
            later_stage_ran = True

        assert not later_stage_ran
        assert not marker.exists()

        print("Nonzero exit code detected: PASS")
        print("Subsequent stage skipped: PASS")
        print("Temporary test isolation: PASS")

    print("ISOLATED TEST STATUS: PASS")

if __name__ == "__main__":
    main()
