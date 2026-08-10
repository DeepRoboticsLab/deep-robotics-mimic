#!/usr/bin/env python3
"""Interactive export script: export motion JSON and policy ONNX from training results.

Flow:
  1. Scan training run directories under logs/rsl_rl
  2. Read params/env.yaml to find motion_file (npz), convert to JSON via npz_to_json.py
  3. List model_*.pt files, convert selected one to ONNX via export_onnx_fast.py

Usage:
  python scripts/export_motion_and_policy.py
"""

import os
import sys
import subprocess
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Readline path Tab completion
# ---------------------------------------------------------------------------
try:
    import readline

    def _path_completer(text: str, state: int) -> str | None:
        """Tab completion: return full path to replace text.

        readline requires the return value to REPLACE text, not just the suffix.
        Therefore we must concatenate dirname + match to return the full path.
        """
        expanded = os.path.expanduser(text)
        if text.startswith("~") and not expanded.startswith("~"):
            prefix = text[: text.index("/") + 1] if "/" in text else text + "/"
            rest = expanded[len(prefix):]
        else:
            prefix = ""
            rest = expanded
        dirname, basename = os.path.split(rest)
        if dirname == "":
            dirname = "."
        if not os.path.isdir(dirname):
            return None
        try:
            entries = sorted(os.listdir(dirname))
        except OSError:
            return None
        matches = [e for e in entries if e.startswith(basename)]
        results = []
        for m in matches:
            full = os.path.join(dirname, m)
            if os.path.isdir(full):
                results.append(prefix + full + "/")
            else:
                results.append(prefix + full)
        if state < len(results):
            return results[state]
        return None

    readline.set_completer(_path_completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("set completion-ignore-case on")
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"
NPZ_TO_JSON = PROJECT_ROOT / "scripts" / "npz_to_json.py"
EXPORT_ONNX = PROJECT_ROOT / "scripts" / "rsl_rl" / "export_onnx_fast.py"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    """Print a framed title."""
    print(f"\n{GREEN}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{NC}\n")


def ask(prompt: str, default: str = "") -> str:
    """Interactive input with default value and Tab path completion."""
    if default:
        s = input(f"{CYAN}{prompt} [{default}]{NC}: ").strip()
        return s if s else default
    return input(f"{CYAN}{prompt}{NC}: ").strip()


def select_from_list(items: list, prompt: str, default_idx: int = 0) -> int:
    """Display candidates as a numbered list and let the user choose."""
    for i, item in enumerate(items):
        print(f"  [{i}] {item}")
    s = ask(prompt, str(default_idx))
    try:
        idx = int(s)
        if 0 <= idx < len(items):
            return idx
    except ValueError:
        pass
    for i, item in enumerate(items):
        if s.lower() in str(item).lower():
            return i
    return default_idx


def find_runs() -> list:
    """Scan logs/rsl_rl for all training run directories containing params/env.yaml."""
    runs = []
    if not LOG_ROOT.exists():
        return runs
    for subproj in sorted(LOG_ROOT.iterdir()):
        if not subproj.is_dir():
            continue
        for run_dir in sorted(subproj.iterdir()):
            if run_dir.is_dir() and (run_dir / "params" / "env.yaml").exists():
                runs.append(run_dir)
    return runs


def extract_motion_file(env_yaml_path: Path) -> str | None:
    """Extract commands.motion.motion_file path from env.yaml.

    env.yaml contains Python-specific YAML tags, requiring unsafe_load.
    """
    with open(env_yaml_path, "r") as f:
        data = yaml.unsafe_load(f)
    commands = data.get("commands", {})
    motion_cfg = commands.get("motion", {})
    return motion_cfg.get("motion_file", None)


def list_pt_files(run_dir: Path) -> list:
    """List model_*.pt files in the run directory, sorted by step count."""
    pt_files = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return pt_files


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main() -> None:
    banner("Export Motion JSON and Policy ONNX")

    # --- Select training run ---
    print(f"{YELLOW}Scanning training runs...{NC}")
    runs = find_runs()
    if not runs:
        print(f"{RED}No training runs found in {LOG_ROOT}.{NC}")
        sys.exit(1)

    runs.reverse()
    run_labels = [f"{r.parent.name}/{r.name}" for r in runs]
    print(f"\nFound {len(runs)} training runs:")
    idx = select_from_list(run_labels, "Select run index", 0)
    run_dir = runs[idx]
    print(f"\n{GREEN}Selected: {run_dir}{NC}")

    env_yaml = run_dir / "params" / "env.yaml"

    json_path = None

    # --- Step 1: NPZ -> JSON ---
    banner("Step 1/2: NPZ to JSON (motion data export)")

    motion_file = extract_motion_file(env_yaml)
    if not motion_file:
        print(f"{RED}commands.motion.motion_file not found in env.yaml{NC}")
    else:
        print(f"  Motion file: {motion_file}")
        motion_path = Path(motion_file)
        if not motion_path.exists():
            print(f"{RED}  Motion file not found: {motion_file}{NC}")
        else:
            default_json = str(run_dir / f"{motion_path.stem}.json")
            json_output = ask("JSON output path", default_json)
            json_path = Path(json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n{YELLOW}  Running npz_to_json.py ...{NC}")
            result = subprocess.run(
                [
                    sys.executable,
                    str(NPZ_TO_JSON),
                    "--input",
                    str(motion_path),
                    "--output",
                    str(json_path),
                ],
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                print(f"{GREEN}  JSON export done: {json_path}{NC}")
            else:
                print(f"{RED}  JSON export failed{NC}")

    # --- Step 2: PT -> ONNX ---
    banner("Step 2/2: PT to ONNX (policy model export)")

    pt_files = list_pt_files(run_dir)
    if not pt_files:
        print(f"{RED}No model_*.pt files found in {run_dir}{NC}")
        sys.exit(1)

    print(f"\nFound {len(pt_files)} model files:")
    pt_names = [f.name for f in pt_files]
    idx = select_from_list(pt_names, "Select model index", len(pt_files) - 1)
    pt_path = pt_files[idx]
    print(f"\n{GREEN}Selected: {pt_path}{NC}")

    default_onnx_name = f"{run_dir.name}_{pt_path.stem}.onnx"
    default_onnx_dir = str(PROJECT_ROOT / "scripts" / "onnx_policy")
    default_onnx = os.path.join(default_onnx_dir, default_onnx_name)

    onnx_output = ask("ONNX output path", default_onnx)
    onnx_path = Path(onnx_output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{YELLOW}  Running export_onnx_fast.py ...{NC}")
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORT_ONNX),
            "--checkpoint_path",
            str(pt_path),
            "--output_dir",
            str(onnx_path.parent),
            "--output_name",
            onnx_path.name,
            "--run_path",
            str(run_dir),
        ],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print(f"{GREEN}  ONNX export done: {onnx_path}{NC}")
    else:
        print(f"{RED}  ONNX export failed{NC}")

    # --- Summary ---
    banner("Export complete!")
    print(f"  Run directory: {run_dir}")
    if json_path and json_path.exists():
        print(f"  JSON:          {json_path}")
    if onnx_path.exists():
        print(f"  ONNX:          {onnx_path}")
    print()


if __name__ == "__main__":
    main()
