"""Machine-specific absolute paths, kept out of git.

Tracked conf.json files hold only repo-relative paths. A path that exists on
one machine only is written there as "${key}/rest", and each machine defines
`key` in its own machine_paths.json next to this file (untracked; template:
machine_paths.example.json). A key the machine does not define is a clear
error, never a silent fallback.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MACHINE_PATHS_JSON = os.path.join(HERE, "machine_paths.json")
_VAR = re.compile(r"^\$\{([A-Za-z0-9_]+)\}")


def _machine():
    if not os.path.exists(MACHINE_PATHS_JSON):
        return {}
    with open(MACHINE_PATHS_JSON) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def resolve(value, base_dir=HERE):
    """'${key}/rest' -> <this machine's key>/rest; other relative paths ->
    base_dir/value; absolute paths and None unchanged."""
    if not value:
        return value
    m = _VAR.match(value)
    if m:
        machine = _machine()
        if m.group(1) not in machine:
            raise SystemExit(
                "%r needs '%s', which this machine does not define: add it to %s "
                "(template: machine_paths.example.json)" % (value, m.group(1), MACHINE_PATHS_JSON))
        return os.path.join(machine[m.group(1)], value[m.end():].lstrip("/"))
    return value if os.path.isabs(value) else os.path.join(base_dir, value)
