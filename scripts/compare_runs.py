"""
Compare agent.yaml and env.yaml between two training runs.

Usage:
    python scripts/compare_runs.py <path_run1> <path_run2>

Each path should be a run directory containing params/agent.yaml and params/env.yaml
(or agent.yaml / env.yaml directly in the given path).
"""

import argparse
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Custom YAML loader that converts Isaac Lab Python-tagged values to plain
# Python objects so they can be diffed without importing Isaac Lab.
# ---------------------------------------------------------------------------

class SafeIslabLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


def _construct_python_slice(loader, node):
    args = loader.construct_sequence(node)
    return f"slice({', '.join(str(a) for a in args)})"


def _construct_python_object_apply(loader, tag_suffix, node):
    args = loader.construct_sequence(node)
    return f"{tag_suffix}({', '.join(str(a) for a in args)})"


SafeIslabLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)
SafeIslabLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", _construct_python_object_apply
)
SafeIslabLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/new:", _construct_python_object_apply
)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=SafeIslabLoader) or {}


# ---------------------------------------------------------------------------
# Flatten nested dict to dot-notation keys
# ---------------------------------------------------------------------------

def flatten(d, prefix=""):
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items.update(flatten(v, key))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            key = f"{prefix}[{i}]"
            items.update(flatten(v, key))
    else:
        items[prefix] = d
    return items


# ---------------------------------------------------------------------------
# Pretty diff
# ---------------------------------------------------------------------------

def diff_flat(flat1: dict, flat2: dict, name1: str, name2: str):
    keys1 = set(flat1)
    keys2 = set(flat2)

    only_in_1 = sorted(keys1 - keys2)
    only_in_2 = sorted(keys2 - keys1)
    common = sorted(keys1 & keys2)
    changed = [(k, flat1[k], flat2[k]) for k in common if flat1[k] != flat2[k]]

    return only_in_1, only_in_2, changed


def print_diff(yaml_name: str, flat1, flat2, name1, name2):
    only1, only2, changed = diff_flat(flat1, flat2, name1, name2)

    if not (only1 or only2 or changed):
        print(f"  [identical]")
        return

    col_w = max(len(name1), len(name2), 6)

    if only1:
        print(f"\n  Keys only in {name1}:")
        for k in only1:
            print(f"    {k} = {flat1[k]!r}")

    if only2:
        print(f"\n  Keys only in {name2}:")
        for k in only2:
            print(f"    {k} = {flat2[k]!r}")

    if changed:
        print(f"\n  Changed values:")
        key_w = max(len(k) for k, *_ in changed)
        print(f"    {'key':<{key_w}}  {name1:<{col_w}}  {name2:<{col_w}}")
        print(f"    {'-'*key_w}  {'-'*col_w}  {'-'*col_w}")
        for k, v1, v2 in changed:
            print(f"    {k:<{key_w}}  {str(v1):<{col_w}}  {str(v2):<{col_w}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_yaml(run_dir: Path, filename: str) -> Path:
    """Search for filename directly in run_dir or in run_dir/params/."""
    candidates = [run_dir / filename, run_dir / "params" / filename]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find {filename} in {run_dir} or {run_dir}/params/"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two training run configs.")
    parser.add_argument("run1", help="Path to first run directory")
    parser.add_argument("run2", help="Path to second run directory")
    args = parser.parse_args()

    run1 = Path(args.run1)
    run2 = Path(args.run2)

    name1 = run1.name
    name2 = run2.name

    for yaml_name in ("agent.yaml", "env.yaml"):
        print(f"\n{'='*70}")
        print(f" {yaml_name}")
        print(f"{'='*70}")

        try:
            path1 = find_yaml(run1, yaml_name)
            path2 = find_yaml(run2, yaml_name)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        data1 = load_yaml(path1)
        data2 = load_yaml(path2)

        flat1 = flatten(data1)
        flat2 = flatten(data2)

        print_diff(yaml_name, flat1, flat2, name1, name2)

    print()


if __name__ == "__main__":
    main()
