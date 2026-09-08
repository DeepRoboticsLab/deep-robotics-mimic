"""Install Bash activation hooks for Isaac Sim's C++ runtime in the active Conda env.

Run with the environment's Python, then deactivate and reactivate the environment.
No packages or system libraries are modified. Re-running this script is safe.
"""

import os
from pathlib import Path
import sys


ACTIVATE = '''# Isaac Sim must load Conda's C++ runtime before the system copy (ICU needs CXXABI_1.3.15).
if [ -z "${_DR02_CXX_RUNTIME_ACTIVE+x}" ]; then
    export _DR02_CXX_RUNTIME_ACTIVE=1
    export _DR02_CXX_RUNTIME_PRELOAD_SET="${LD_PRELOAD+x}"
    export _DR02_CXX_RUNTIME_PRELOAD_OLD="${LD_PRELOAD-}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
'''

DEACTIVATE = '''# Restore the preload setting from before this environment was activated.
if [ "${_DR02_CXX_RUNTIME_ACTIVE-}" = 1 ]; then
    if [ "${_DR02_CXX_RUNTIME_PRELOAD_SET-}" = x ]; then
        export LD_PRELOAD="$_DR02_CXX_RUNTIME_PRELOAD_OLD"
    else
        unset LD_PRELOAD
    fi
    unset _DR02_CXX_RUNTIME_ACTIVE _DR02_CXX_RUNTIME_PRELOAD_SET _DR02_CXX_RUNTIME_PRELOAD_OLD
fi
'''


def main():
    if sys.platform != "linux":
        raise SystemExit("This runtime setup is for Linux Conda environments only.")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix or Path(conda_prefix).resolve() != Path(sys.prefix).resolve():
        raise SystemExit("Activate your Isaac Lab Conda environment and run this script with its Python.")

    prefix = Path(conda_prefix)
    runtime = prefix / "lib/libstdc++.so.6"
    if not runtime.is_file() or b"CXXABI_1.3.15\x00" not in runtime.read_bytes():
        raise SystemExit(
            "The environment needs a newer C++ runtime. Run:\n"
            '  conda install -c conda-forge "libstdcxx-ng>=15"\n'
            "Then re-run this script."
        )

    for directory, contents in (("activate.d", ACTIVATE), ("deactivate.d", DEACTIVATE)):
        hook = prefix / "etc/conda" / directory / "dr02-cxx-runtime.sh"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(contents)
        print(f"Installed {hook}")
    print("Deactivate and reactivate this environment before launching training.")


if __name__ == "__main__":
    main()
