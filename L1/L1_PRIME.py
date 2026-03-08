import subprocess
import sys
from pathlib import Path

scripts = [
    "bartTorvik/bartTorvik_transform_L1.py",
    "bracketology/clean_espn_public_picks.py",
    "espnBPI/espnBPI_transform_L1.py",
    "kenPom/kenPom_transform_L1.py",
    #"LRMCB/LRMCB_transform_L1.py",
    "masseyComposite/masseyComposite_transform_L1.py",
    "powerRank/powerRank_transform_L1.py",
    "srcbb/srcbb_transform_L1.py",
    #"vegasOdds/vegasOdds_transform_L1.py",
]

def run_script(path: str) -> bool:
    print(f"\n{'='*50}")
    print(f"Running: {path}")
    print('='*50)
    script_path = Path(path).resolve()
    result = subprocess.run(
        [sys.executable, script_path.name],
        cwd=script_path.parent,  # run from script's own directory
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"ERROR: {path} exited with code {result.returncode}")
        return False
    print(f"Done: {path}")
    return True

if __name__ == "__main__":
    failed = []
    for script in scripts:
        if not Path(script).exists():
            print(f"SKIPPED (not found): {script}")
            failed.append(script)
            continue
        if not run_script(script):
            failed.append(script)

    print(f"\n{'='*50}")
    if failed:
        print(f"COMPLETED WITH ERRORS — {len(failed)} script(s) failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(scripts)} L1 SCRIPTS COMPLETED SUCCESSFULLY")
