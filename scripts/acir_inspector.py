#!/usr/bin/env python3
"""
ACIR Inspector — compile a Nargo project and explore the resulting ACIR.

Usage:
    python scripts/acir_inspector.py <project_dir> [options]

    <project_dir>  Path to a Nargo project (containing Nargo.toml) or to any
                   file inside such a project (e.g. src/main.nr).

Options:
    --stats              Print opcode-type breakdown and exit.
    --filter-type TYPE   Show only opcodes whose type matches TYPE (case-
                         insensitive substring); e.g. ASSERT, BLACKBOX, BRILLIG.
    --witness WN         Show only opcodes that reference witness WN (e.g. w42).
    --func N             Show only ACIR function N (0-indexed).
    --no-brillig         Suppress the unconstrained Brillig function listings.
    --no-color           Disable ANSI colour output.
    --no-source          Do not print the Noir source above the ACIR.
    --force              Pass --force to nargo (recompile from scratch).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA = "\033[95m"
BLUE   = "\033[94m"

_color_enabled = True

def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _color_enabled else text

# ---------------------------------------------------------------------------
# Opcode classification
# ---------------------------------------------------------------------------

OPCODE_PATTERNS = [
    ("BLACKBOX",     re.compile(r"^BLACKBOX::")),
    ("BRILLIG_CALL", re.compile(r"^BRILLIG CALL")),
    ("ASSERT",       re.compile(r"^ASSERT ")),
    ("EXPRESSION",   re.compile(r"^EXPRESSION ")),
    ("MEMORY_INIT",  re.compile(r"^MEMORY INIT")),
    ("MEMORY_OP",    re.compile(r"^MEMORY OP")),
    ("MEM_INIT",     re.compile(r"^INIT ")),
    ("MEM_READ",     re.compile(r"^READ ")),
    ("MEM_WRITE",    re.compile(r"^WRITE ")),
]

OPCODE_COLORS = {
    "BLACKBOX":     YELLOW,
    "BRILLIG_CALL": CYAN,
    "ASSERT":       GREEN,
    "EXPRESSION":   MAGENTA,
    "MEMORY_INIT":  BLUE,
    "MEMORY_OP":    BLUE,
    "MEM_INIT":     BLUE,
    "MEM_READ":     BLUE,
    "MEM_WRITE":    BLUE,
}

def classify(line: str) -> str:
    for name, pat in OPCODE_PATTERNS:
        if pat.match(line):
            return name
    return "OTHER"

def colorize_line(line: str, kind: str) -> str:
    color = OPCODE_COLORS.get(kind, RESET)
    # Dim inline comments
    if "//" in line:
        code, _, comment = line.partition("//")
        return _c(color, code.rstrip()) + " " + _c(DIM, "//" + comment)
    return _c(color, line)

def witnesses_in(line: str) -> set[str]:
    """Return the set of witness identifiers (wN) referenced in a line."""
    return set(re.findall(r"\bw\d+\b", line))

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class AcirFunc:
    def __init__(self, index: int, name: str):
        self.index = index
        self.name = name
        self.header_lines: list[str] = []   # private/public/return
        self.opcodes: list[tuple[str, str]] = []  # (kind, raw_line)
        self.is_brillig = False

    def opcode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kind, _ in self.opcodes:
            counts[kind] = counts.get(kind, 0) + 1
        return counts


def parse_acir(raw: str) -> list[AcirFunc]:
    funcs: list[AcirFunc] = []
    current: AcirFunc | None = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Skip the trailing stats table
        if line.startswith("+---") or line.startswith("| Package") or line.startswith("|="):
            break

        # New constrained function
        m = re.match(r"^func (\d+)$", line)
        if m:
            current = AcirFunc(int(m.group(1)), f"func {m.group(1)}")
            funcs.append(current)
            continue

        # New unconstrained (Brillig) function
        m = re.match(r"^unconstrained func (\d+):\s*(.+)$", line)
        if m:
            current = AcirFunc(int(m.group(1)), m.group(2).strip())
            current.is_brillig = True
            funcs.append(current)
            continue

        if current is None:
            continue

        # Header lines inside a function
        if line.startswith(("private parameters:", "public parameters:", "return values:")):
            current.header_lines.append(line)
            continue

        # Brillig bytecode (lines like "0: @10 = const u32 2")
        if current.is_brillig and re.match(r"^\d+:", line):
            current.opcodes.append(("BRILLIG_INSN", line))
            continue

        # ACIR opcodes
        kind = classify(line)
        current.opcodes.append((kind, line))

    return funcs

# ---------------------------------------------------------------------------
# Locate and compile the project
# ---------------------------------------------------------------------------

def find_nargo_toml(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        candidate = p / "Nargo.toml"
        if candidate.exists():
            return p
    return None


def compile_acir(project_dir: Path, force: bool) -> str:
    cmd = ["nargo", "info", "--print-acir", "--silence-warnings"]
    if force:
        cmd.append("--force")
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    # nargo writes ACIR to stdout, errors to stderr
    if result.returncode != 0 and not result.stdout.strip():
        print(_c(RED, "nargo failed:"), file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_func_header(func: AcirFunc) -> None:
    prefix = "unconstrained " if func.is_brillig else ""
    print(_c(BOLD, f"\n{prefix}func {func.index}: {func.name}"))
    for h in func.header_lines:
        print(_c(DIM, "  " + h))

def print_stats(funcs: list[AcirFunc]) -> None:
    print(_c(BOLD, "\n=== ACIR opcode statistics ==="))
    for func in funcs:
        kind_label = "unconstrained " if func.is_brillig else ""
        print(_c(BOLD, f"\n  {kind_label}func {func.index}: {func.name}"))
        counts = func.opcode_counts()
        total = sum(counts.values())
        for kind, n in sorted(counts.items(), key=lambda x: -x[1]):
            color = OPCODE_COLORS.get(kind, RESET)
            bar = "█" * min(n, 40)
            print(f"    {_c(color, f'{kind:<16}')}  {n:>5}  {_c(DIM, bar)}")
        print(f"    {'TOTAL':<16}  {total:>5}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _color_enabled

    ap = argparse.ArgumentParser(
        description="Inspect the ACIR generated from a Nargo project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("project", help="Nargo project directory or a file inside it")
    ap.add_argument("--stats",       action="store_true", help="Print opcode-type breakdown")
    ap.add_argument("--filter-type", metavar="TYPE",      help="Show only opcodes matching TYPE")
    ap.add_argument("--witness",     metavar="WN",        help="Show only opcodes referencing witness WN")
    ap.add_argument("--func",        type=int, default=-1, metavar="N", help="Show only function N")
    ap.add_argument("--no-brillig",  action="store_true", help="Hide unconstrained Brillig functions")
    ap.add_argument("--no-color",    action="store_true", help="Disable colour output")
    ap.add_argument("--no-source",   action="store_true", help="Do not print the Noir source")
    ap.add_argument("--force",       action="store_true", help="Force recompilation")
    args = ap.parse_args()

    if args.no_color:
        _color_enabled = False

    # Locate project root
    start = Path(args.project).resolve()
    if start.is_file():
        start = start.parent
    project_dir = find_nargo_toml(start)
    if project_dir is None:
        print(_c(RED, f"No Nargo.toml found at or above {start}"), file=sys.stderr)
        sys.exit(1)

    print(_c(DIM, f"Project: {project_dir}"))

    # Optionally show Noir source
    if not args.no_source:
        main_nr = project_dir / "src" / "main.nr"
        if main_nr.exists():
            print(_c(BOLD, "\n=== Noir source (src/main.nr) ==="))
            src = main_nr.read_text()
            lines = src.splitlines()
            width = len(str(len(lines)))
            for i, l in enumerate(lines, 1):
                print(f"  {_c(DIM, str(i).rjust(width))}  {l}")

    # Compile and parse
    raw = compile_acir(project_dir, args.force)
    funcs = parse_acir(raw)

    if args.stats:
        print_stats(funcs)
        return

    # Apply function filter
    if args.func >= 0:
        funcs = [f for f in funcs if f.index == args.func]
        if not funcs:
            print(_c(RED, f"No function with index {args.func} found."), file=sys.stderr)
            sys.exit(1)

    if args.no_brillig:
        funcs = [f for f in funcs if not f.is_brillig]

    # Normalise filter options
    type_filter: str | None = args.filter_type.upper() if args.filter_type else None
    wit_filter:  str | None = args.witness.lower() if args.witness else None

    # Print
    print(_c(BOLD, "\n=== ACIR ==="))
    for func in funcs:
        print_func_header(func)

        shown = 0
        for kind, raw_line in func.opcodes:
            if type_filter and type_filter not in kind:
                continue
            if wit_filter and wit_filter not in {w.lower() for w in witnesses_in(raw_line)}:
                continue
            print("  " + colorize_line(raw_line, kind))
            shown += 1

        if (type_filter or wit_filter) and shown == 0:
            print(_c(DIM, "  (no matching opcodes)"))

    print()


if __name__ == "__main__":
    main()
