"""Tiny interactive-prompt helpers shared by the scripts.

Each script keeps its argparse flags (for scripting / overrides), but when run with NO
flags from a terminal it drops into an interactive prompt that fills the same parameters
— listing prior runs / files to pick from where relevant. `interactive()` is the gate.
"""
import glob
import os
import re
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "probes")


def interactive():
    """True when the script was launched bare (no flags) from a real terminal."""
    return len(sys.argv) == 1 and sys.stdin.isatty()


def _ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("cancelled")


def ask_str(label, default=None):
    d = f" [{default}]" if default not in (None, "") else ""
    return _ask(f"{label}{d}: ") or default


def ask_int(label, default):
    while True:
        v = _ask(f"{label} [{default}]: ")
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            print("  please enter an integer")


def ask_float(label, default):
    while True:
        v = _ask(f"{label} [{default}]: ")
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            print("  please enter a number")


def ask_bool(label, default=False):
    hint = "Y/n" if default else "y/N"
    v = _ask(f"{label} [{hint}]: ").lower()
    if not v:
        return default
    return v[0] == "y"


def choose(label, options, default_idx=0, allow_new=False, new_label="enter a new value"):
    """Numbered menu. `options` is a list of (value, display) tuples or plain strings.

    Returns the chosen value. With allow_new, an extra entry lets the user type a value.
    """
    opts = [(o, o) if isinstance(o, str) else o for o in options]
    print(f"\n{label}")
    for i, (_, disp) in enumerate(opts):
        star = "  <- default" if i == default_idx else ""
        print(f"  {i + 1}. {disp}{star}")
    new_idx = None
    if allow_new:
        new_idx = len(opts) + 1
        print(f"  {new_idx}. ({new_label})")
    while True:
        raw = _ask("  choice: ")
        if not raw and opts:
            return opts[default_idx][0]
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(opts):
                return opts[n - 1][0]
            if allow_new and n == new_idx:
                return ask_str("  value")
        print("  invalid choice")


def list_runs(pattern, regex, base=RESULTS_DIR):
    """Run ids extracted from filenames matching `pattern`, newest-first.

    `pattern` is a glob (e.g. "part1_refusal_nla_*.json"); `regex` captures the run id
    in group 1 from the basename. Returns a list of (run_id, mtime) sorted newest-first.
    """
    seen = {}
    for f in glob.glob(os.path.join(base, pattern)):
        m = re.match(regex, os.path.basename(f))
        if m:
            seen[m.group(1)] = max(seen.get(m.group(1), 0), os.path.getmtime(f))
    return sorted(seen.items(), key=lambda kv: -kv[1])


def choose_run(pattern, regex, label="run", base=RESULTS_DIR, allow_new=False):
    """Interactively pick a run id from those on disk (newest first)."""
    runs = list_runs(pattern, regex, base)
    if not runs and not allow_new:
        sys.exit(f"No runs found matching {pattern} in {base}")
    options = [(rid, rid) for rid, _ in runs]
    return choose(f"Select {label}:", options, default_idx=0, allow_new=allow_new,
                  new_label=f"enter a new {label}")


def choose_file(pattern, label="file", base=RESULTS_DIR):
    """Interactively pick a file path from a glob (newest first)."""
    files = sorted(glob.glob(os.path.join(base, pattern)), key=os.path.getmtime, reverse=True)
    if not files:
        sys.exit(f"No files matching {pattern} in {base}")
    options = [(f, os.path.basename(f)) for f in files]
    return choose(f"Select {label}:", options, default_idx=0)
