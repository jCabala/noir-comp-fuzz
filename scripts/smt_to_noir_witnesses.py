#!/usr/bin/env python3
"""
Translate SMT-LIB formulas to Noir circuits, enumerate satisfying models
from Z3, and run `nargo execute` to generate witness files.

Usage:
    python scripts/smt_to_noir_witnesses.py <input_dir> [options]

For each .smt / .smt2 file in <input_dir>:
  1. Parse to Circuit IR using the project's parser.
  2. Emit Noir source via IR2NoirVisitor + EmitVisitor.
  3. Use Z3 subprocess to enumerate up to --max-witnesses satisfying models
     (blocking clause enumeration).
  4. For each model, write a self-contained Noir project and run nargo execute.

Supported logics:
  - Boolean core (declare-fun + bool operators)
  - QF_FF (finite-field; requires a Z3 build with FiniteField support)
"""

import argparse
import hashlib
import json
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root so that `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.smt_to_ir.parser import parse_smtlib2
from src.backends.noir.ir2noir import IR2NoirVisitor, _fold_expr_constant
from src.backends.noir.emitter import EmitVisitor
from src.backends.noir.types import IntegerType
from src.ir.nodes import VariableType, Variable, Circuit, BinaryExpression, Integer as IRInteger


# ---------------------------------------------------------------------------
# Z3 helpers
# ---------------------------------------------------------------------------

def _strip_check_commands(smt: str) -> str:
    """Remove check-sat, get-model, get-value commands so we can add our own."""
    out = []
    for line in smt.splitlines():
        s = line.strip()
        if s.startswith(("(check-sat", "(get-model", "(get-value", "(exit")):
            continue
        out.append(line)
    return "\n".join(out)


_DEFINE_FUN_RE = re.compile(
    r'\(define-fun\s+(\S+)\s+\(\)\s+(\S+)\s+((?:[^()]+|\([^()]*\))*)\)',
    re.DOTALL,
)

_I64_MIN = -(2 ** 63)
_I64_MAX = 2 ** 63 - 1

# All supported integer types for randomized assignment.
_INT_TYPE_POOL = [
    IntegerType(8,  True),   # i8
    IntegerType(16, True),   # i16
    IntegerType(32, True),   # i32
    IntegerType(64, True),   # i64
    IntegerType(8,  False),  # u8
    IntegerType(16, False),  # u16
    IntegerType(32, False),  # u32
    IntegerType(64, False),  # u64  (Z3 bounds capped at [0, 2^63-1] for safe negation)
]

def _type_bounds(t: IntegerType) -> tuple[int, int]:
    """Return the (lo, hi) Z3 integer bounds for a Noir integer type."""
    if t.signed:
        lo = -(2 ** (t.bits - 1))
        hi =  (2 ** (t.bits - 1)) - 1
    else:
        lo = 0
        # Cap u64 at i64 max so unary negation (via i64 promotion) never overflows.
        hi = (2 ** min(t.bits, 63)) - 1
    return lo, hi


def _effective_mul_factor(expr, var_name: str) -> int | None:
    """
    Return the total constant multiplication factor accumulated on top of
    var_name in expr, or None if var_name doesn't appear in expr.

    For ((v * 95) * 95) * 81:  returns 95 * 95 * 81 = 731,025
    Only follows MUL chains where one side is purely constant at each step.
    """
    from src.ir.nodes import BinaryExpression as IRBin, Variable as IRVar, Operator as IROp
    if isinstance(expr, IRVar):
        return 1 if expr.name == var_name else None
    if isinstance(expr, IRBin) and expr.op == IROp.MUL:
        lc = _fold_expr_constant(expr.lhs)
        rc = _fold_expr_constant(expr.rhs)
        if rc is not None:
            sub = _effective_mul_factor(expr.lhs, var_name)
            if sub is not None:
                return abs(rc) * sub
        if lc is not None:
            sub = _effective_mul_factor(expr.rhs, var_name)
            if sub is not None:
                return abs(lc) * sub
    return None


def _collect_mul_factors(circuit: Circuit) -> dict[str, int]:
    """
    Walk all assertions and return, for each INTEGER variable, the maximum
    effective multiplication factor seen in any MUL chain containing it.
    """
    from src.ir.nodes import (
        BinaryExpression as IRBin, UnaryExpression as IRUnary,
        TernaryExpression as IRTernary, Assertion, Variable as IRVar, Operator as IROp,
    )
    factors: dict[str, int] = {}

    def _update(name: str, factor: int) -> None:
        factors[name] = max(factors.get(name, 1), factor)

    def _walk(node) -> None:
        if isinstance(node, IRBin):
            if node.op == IROp.MUL:
                # Try to find a variable on either side and compute its factor.
                for var in circuit.inputs:
                    if var.variable_type != VariableType.INTEGER:
                        continue
                    f = _effective_mul_factor(node, var.name)
                    if f is not None and f > 1:
                        _update(var.name, f)
            _walk(node.lhs)
            _walk(node.rhs)
        elif isinstance(node, IRUnary):
            _walk(node.value)
        elif isinstance(node, IRTernary):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, Assertion):
            _walk(node.value)

    for stmt in circuit.statements:
        _walk(stmt)
    return factors


def _collect_var_const_ranges(circuit: Circuit) -> dict[str, tuple[int, int]]:
    """
    Walk every assertion in the circuit and collect, for each INTEGER variable,
    the min/max integer literals that appear as its **direct sibling** in a
    binary expression.

    "Direct sibling" means the two children of the same BinaryExpression node
    are exactly one Variable and one Integer — not buried deeper.

    Returns {var_name: (min_const, max_const)}.
    """
    ranges: dict[str, tuple[int, int]] = {}

    def _update(name: str, val: int) -> None:
        lo, hi = ranges.get(name, (val, val))
        ranges[name] = (min(lo, val), max(hi, val))

    def _walk(node) -> None:
        from src.ir.nodes import (
            BinaryExpression as IRBin, UnaryExpression as IRUnary,
            TernaryExpression as IRTernary, Assertion, Variable as IRVar,
        )
        if isinstance(node, IRBin):
            lhs, rhs = node.lhs, node.rhs
            # One side is an INTEGER variable, the other is a constant-only
            # expression (bare literal OR compound like 29*29).  Fold the
            # constant side to get the effective value seen by the variable.
            lhs_const = _fold_expr_constant(lhs) if not isinstance(lhs, IRVar) else None
            rhs_const = _fold_expr_constant(rhs) if not isinstance(rhs, IRVar) else None
            if isinstance(lhs, IRVar) and lhs.variable_type == VariableType.INTEGER \
                    and rhs_const is not None:
                _update(lhs.name, rhs_const)
            elif isinstance(rhs, IRVar) and rhs.variable_type == VariableType.INTEGER \
                    and lhs_const is not None:
                _update(rhs.name, lhs_const)
            _walk(lhs)
            _walk(rhs)
        elif isinstance(node, IRUnary):
            _walk(node.value)
        elif isinstance(node, IRTernary):
            _walk(node.condition)
            _walk(node.if_expr)
            _walk(node.else_expr)
        elif isinstance(node, Assertion):
            _walk(node.value)

    for stmt in circuit.statements:
        _walk(stmt)

    return ranges


def assign_int_types(
    int_var_names: list[str],
    smt_content: str,
    circuit: Circuit | None = None,
) -> dict[str, IntegerType]:
    """
    Deterministically assign a Noir integer type to each Int variable.

    Filters the type pool by two constraints per variable:
    1. Direct-sibling constant range: type must cover all constants directly
       compared with / added to the variable.
    2. Effective multiplication factor K: type.max // K >= 1, so the full
       chain v * K never overflows (Z3 bounds will be clamped to ±type.max//K).
    Falls back to i64 when nothing else fits.
    """
    seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16)
    rng = random.Random(seed)

    const_ranges = _collect_var_const_ranges(circuit) if circuit else {}
    mul_factors  = _collect_mul_factors(circuit)       if circuit else {}

    result: dict[str, IntegerType] = {}
    for name in int_var_names:
        min_c, max_c = const_ranges.get(name, (0, 0))
        k = mul_factors.get(name, 1)
        valid = [
            t for t in _INT_TYPE_POOL
            if _type_bounds(t)[0] <= min_c
            and _type_bounds(t)[1] >= max_c
            and _type_bounds(t)[1] // k >= 1   # product chain fits in type
        ]
        result[name] = rng.choice(valid) if valid else IntegerType(64, True)

    return result


def _z3_val_to_str(raw: str) -> str:
    """Normalise a Z3 value token to a plain decimal string.

    Z3 represents negative integers as (- N) rather than -N.
    """
    s = raw.strip()
    m = re.match(r'^\(-\s*(\d+)\)?$', s)
    if m:
        return f"-{m.group(1)}"
    return s


def _parse_z3_model(output: str) -> dict[str, tuple[str, str]] | None:
    """
    Parse z3's stdout into {var_name: (sort, value_str)}.
    Returns None when the result is not 'sat'.
    """
    lines = output.strip().splitlines()
    if not lines or lines[0].strip() != "sat":
        return None
    model_text = "\n".join(lines[1:])
    result: dict[str, tuple[str, str]] = {}
    for m in _DEFINE_FUN_RE.finditer(model_text):
        name = m.group(1)
        sort = m.group(2).strip()
        value = _z3_val_to_str(m.group(3))
        result[name] = (sort, value)
    return result


def _blocking_clause(model: dict[str, tuple[str, str]], declared: set[str]) -> str:
    """Build (assert (not (and (= v val) ...))) over declared variables only."""
    equalities = [
        f"(= {name} {value})"
        for name, (_, value) in model.items()
        if name in declared
    ]
    if not equalities:
        return ""
    body = equalities[0] if len(equalities) == 1 else f"(and {' '.join(equalities)})"
    return f"(assert (not {body}))"


def _bounds_clauses(
    int_vars: set[str],
    type_map: dict[str, IntegerType],
    mul_factors: dict[str, int] | None = None,
) -> list[str]:
    """
    Return (assert ...) clauses bounding each integer variable.
    When a multiplication factor K > 1 is known for a variable, the bound is
    clamped to ±type.max//K so that v * K never overflows the assigned type.
    """
    factors = mul_factors or {}
    clauses = []
    for v in sorted(int_vars):
        t = type_map.get(v, IntegerType(64, True))
        lo, hi = _type_bounds(t)
        k = factors.get(v, 1)
        if k > 1:
            safe = hi // k
            lo, hi = -safe, safe
        clauses.append(f"(assert (>= {v} {lo}))")
        clauses.append(f"(assert (<= {v} {hi}))")
    return clauses


def enumerate_models(
    smt_content: str,
    declared: set[str],
    max_models: int,
    timeout: int = 30,
    int_vars: set[str] | None = None,
    int_type_map: dict[str, IntegerType] | None = None,
    mul_factors: dict[str, int] | None = None,
) -> list[dict[str, tuple[str, str]]]:
    """
    Call Z3 repeatedly with blocking clauses to collect up to max_models
    satisfying assignments.

    Bounds for each integer variable are derived from its assigned Noir type
    and clamped by the effective multiplication factor to prevent overflow.
    """
    base = _strip_check_commands(smt_content)
    bounds = _bounds_clauses(int_vars or set(), int_type_map or {}, mul_factors)
    extra: list[str] = bounds[:]
    models: list[dict[str, tuple[str, str]]] = []

    for _ in range(max_models):
        query = base + "\n" + "\n".join(extra) + "\n(check-sat)\n(get-model)\n"
        proc = subprocess.run(
            ["z3", "-in"],
            input=query,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        model = _parse_z3_model(proc.stdout)
        if model is None:
            break
        models.append(model)
        clause = _blocking_clause(model, declared)
        if not clause:
            break
        extra.append(clause)

    return models


# ---------------------------------------------------------------------------
# Prover.toml generation
# ---------------------------------------------------------------------------

def model_to_prover_toml(
    model: dict[str, tuple[str, str]],
    circuit,
) -> str:
    """
    Convert a Z3 model to Noir's Prover.toml format.
    Boolean vars → TOML booleans; Field vars → quoted decimal strings.
    Variables missing from the model get a zero/false default.
    """
    lines: list[str] = []
    for var in circuit.inputs:
        name = var.name
        if var.variable_type == VariableType.BOOLEAN:
            if name in model:
                _, raw = model[name]
                lines.append(f"{name} = {raw}")      # true / false
            else:
                lines.append(f"{name} = false")
        else:
            if name in model:
                _, raw = model[name]
                # Strip 'ff' prefix that CVC5/Z3-FF uses for field constants.
                if raw.startswith("ff"):
                    raw = raw[2:]
                lines.append(f'{name} = "{raw}"')
            else:
                lines.append(f'{name} = "0"')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Noir project helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convert an arbitrary string to a valid Noir package name."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "circuit"


def create_noir_project(
    project_dir: Path,
    noir_source: str,
    prover_toml: str,
    package_name: str,
    smt_source: str = "",
    smt_filename: str = "",
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "main.nr").write_text(noir_source)
    (project_dir / "Prover.toml").write_text(prover_toml)
    (project_dir / "Nargo.toml").write_text(
        f'[package]\nname = "{package_name}"\ntype = "bin"\nauthors = [""]\n\n[dependencies]\n'
    )
    if smt_source and smt_filename:
        (project_dir / smt_filename).write_text(smt_source)


def save_error(
    errors_dir: Path,
    category: str,
    folder_name: str,
    noir_source: str,
    prover_toml: str,
    package_name: str,
    smt_content: str,
    smt_filename: str,
    nargo_output: str,
) -> None:
    dest = errors_dir / category / folder_name
    create_noir_project(dest, noir_source, prover_toml, package_name,
                        smt_source=smt_content, smt_filename=smt_filename)
    (dest / "nargo_output.txt").write_text(nargo_output)


def run_nargo_execute(project_dir: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run `nargo execute` in project_dir.  Returns (success, combined output)."""
    proc = subprocess.run(
        ["nargo", "execute"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# smtfuzz generation
# ---------------------------------------------------------------------------

def generate_smt(config: dict, timeout: int = 30) -> str:
    """Call smtfuzz with flags from config dict and return the generated formula.
    List values are resolved by picking one element at random (allows e.g. multiple logics)."""
    cmd = ["smtfuzz"]
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = random.choice(value)
        cmd += [f"--{key}", str(value)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"smtfuzz failed: {proc.stderr.strip()}")
    return proc.stdout


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------

def _boundary_values(t: IntegerType) -> list[int]:
    """Return boundary integers for a given Noir integer type."""
    lo, hi = _type_bounds(t)
    candidates = [lo, lo + 1, -1, 0, 1, hi - 1, hi]
    return sorted(set(v for v in candidates if lo <= v <= hi))


def build_boundary_prover_toml(
    circuit,
    int_type_map: dict[str, IntegerType],
    rng: random.Random,
    mul_factors: dict[str, int] | None = None,
) -> str:
    """Build a Prover.toml by sampling boundary values for each variable.
    Integer bounds are clamped by the multiplication factor to prevent overflow."""
    factors = mul_factors or {}
    lines: list[str] = []
    for var in circuit.inputs:
        name = var.name
        if var.variable_type == VariableType.BOOLEAN:
            lines.append(f"{name} = {rng.choice(['true', 'false'])}")
        else:
            t = int_type_map.get(name, IntegerType(64, True))
            lo, hi = _type_bounds(t)
            k = factors.get(name, 1)
            if k > 1:
                safe = hi // k
                lo, hi = -safe, safe
            candidates = [lo, lo + 1, -1, 0, 1, hi - 1, hi]
            pool = sorted(set(v for v in candidates if lo <= v <= hi))
            val = rng.choice(pool)
            lines.append(f'{name} = "{val}"')
    return "\n".join(lines) + "\n"


def _is_unsat_with_fix(
    smt_content: str,
    var_name: str,
    fixed_value: str,
    int_vars: set[str],
    int_type_map: dict[str, IntegerType],
    timeout: int = 10,
) -> bool:
    """Return True if the formula is UNSAT when var_name is forced to fixed_value."""
    base = _strip_check_commands(smt_content)
    bounds = _bounds_clauses(int_vars, int_type_map)
    fix = f"(assert (= {var_name} {fixed_value}))"
    query = base + "\n" + "\n".join(bounds) + "\n" + fix + "\n(check-sat)\n"
    proc = subprocess.run(
        ["z3", "-in"], input=query,
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout.strip().startswith("unsat")


def find_sat_oracle_input(
    smt_content: str,
    circuit,
    model: dict,
    int_vars: set[str],
    int_type_map: dict[str, IntegerType],
) -> dict | None:
    """
    Try flipping each boolean variable in model one at a time.
    Return the first modified model that Z3 confirms is UNSAT, or None.
    """
    for var in circuit.inputs:
        if var.variable_type != VariableType.BOOLEAN:
            continue
        current = model.get(var.name, ('Bool', 'false'))[1]
        flipped = 'false' if current == 'true' else 'true'
        try:
            if _is_unsat_with_fix(smt_content, var.name, flipped,
                                  int_vars, int_type_map):
                return {**model, var.name: ('Bool', flipped)}
        except subprocess.TimeoutExpired:
            continue
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMT-LIB → Noir circuit translation + witness generation."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Directory containing .smt / .smt2 files. "
             "If omitted, formulas are generated on-the-fly using smtfuzz.",
    )
    parser.add_argument(
        "--smtfuzz-config",
        default="smtfuzz_config.json",
        metavar="FILE",
        help="JSON file with smtfuzz flags (default: smtfuzz_config.json). "
             "Used only in generate mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="obj",
        help="Root directory for all runs (default: ./obj). Each run is stored under <output-dir>/run-<ID>/.",
    )
    parser.add_argument(
        "--max-witnesses",
        type=int,
        default=1,
        metavar="N",
        help="Maximum satisfying witnesses to generate per formula (default: 1).",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of input formulas to process (default: unlimited).",
    )
    parser.add_argument(
        "--z3-timeout",
        type=int,
        default=30,
        metavar="SECS",
        help="Seconds to allow Z3 per query (default: 30).",
    )
    parser.add_argument(
        "--nargo-timeout",
        type=int,
        default=120,
        metavar="SECS",
        help="Seconds to allow nargo execute per witness (default: 120).",
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=10,
        metavar="N",
        help="Keep only the N most recent run directories, deleting older ones (default: 10).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generate_mode = args.input_dir is None

    # --- folder mode ---
    if not generate_mode:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            sys.exit(f"Error: {input_dir} is not a directory.")
        smt_files = sorted(input_dir.glob("*.smt")) + sorted(input_dir.glob("*.smt2"))
        if not smt_files:
            sys.exit(f"No .smt / .smt2 files found in {input_dir}.")
        if args.max_programs is not None:
            smt_files = smt_files[: args.max_programs]

    # --- generate mode: load smtfuzz config ---
    smtfuzz_config: dict = {}
    if generate_mode:
        config_path = Path(args.smtfuzz_config)
        if not config_path.exists():
            sys.exit(f"Error: smtfuzz config not found at {config_path}. "
                     f"Create it or pass --smtfuzz-config <file>.")
        with config_path.open() as f:
            smtfuzz_config = json.load(f)
        n_generate = args.max_programs or 100   # default 100 in generate mode
        print(f"Generate mode: smtfuzz config={config_path}, target={n_generate} formula(s)")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id  = secrets.token_hex(4)
    run_dir = output_dir / f"run-{timestamp}-{run_id}"
    sat_dir          = run_dir / "out" / "sat"
    unsat_dir        = run_dir / "out" / "unsat"
    sat_to_unsat_dir = run_dir / "out" / "sat_to_unsat"
    errors_dir       = run_dir / "errors"
    sat_dir.mkdir(parents=True, exist_ok=True)
    unsat_dir.mkdir(parents=True, exist_ok=True)
    sat_to_unsat_dir.mkdir(parents=True, exist_ok=True)
    (errors_dir / "error").mkdir(parents=True, exist_ok=True)
    (errors_dir / "timeout").mkdir(parents=True, exist_ok=True)

    # Enforce --keep-runs: delete only the out/ of old runs — errors/ is never touched.
    all_runs = sorted(output_dir.glob("run-*"))   # lexicographic = chronological
    excess = len(all_runs) - args.keep_runs
    for old_run in all_runs[:excess]:
        out_to_delete = old_run / "out"
        if out_to_delete.exists():
            shutil.rmtree(out_to_delete, ignore_errors=True)
            print(f"[cleanup] cleared out/ from: {old_run.name}")

    if generate_mode:
        print(f"Run ID: {run_id}  →  {run_dir}")
    else:
        print(f"Found {len(smt_files)} formula(s) in {input_dir} (cap: {args.max_programs or 'none'})")
        print(f"Run ID: {run_id}  →  {run_dir}")

    emitter = EmitVisitor()
    ok = n_error = n_timeout = skipped = 0

    def _iter_formulas():
        """Yield (stem, smt_content) for either folder or generate mode."""
        if not generate_mode:
            for smt_file in smt_files:
                yield smt_file.stem, smt_file.read_text()
        else:
            idx = 0
            while idx < n_generate:
                try:
                    content = generate_smt(smtfuzz_config)
                except Exception as exc:
                    print(f"  [smtfuzz] generation error: {exc}")
                    continue
                yield f"gen-{idx:04d}", content
                idx += 1

    for stem, smt_content in _iter_formulas():
        smt_filename = f"{stem}.smt2"
        print(f"\n=== {stem} ===")

        # 1. Parse → Circuit IR
        try:
            circuit = parse_smtlib2(smt_content)
        except Exception as exc:
            print(f"  SKIP (parse error): {exc}")
            skipped += 1
            continue

        declared = {v.name for v in circuit.inputs}
        int_var_names = [v.name for v in circuit.inputs if v.variable_type == VariableType.INTEGER]
        int_vars = set(int_var_names)

        int_type_map: dict[str, IntegerType] = {}
        mul_factors: dict[str, int] = {}
        if int_var_names:
            mul_factors  = _collect_mul_factors(circuit)
            int_type_map = assign_int_types(int_var_names, smt_content, circuit)
            types_summary = ", ".join(f"{n}:{t}" for n, t in int_type_map.items())
            print(f"  Types: {types_summary}")

        # 2. Circuit IR → Noir source
        try:
            noir_doc = IR2NoirVisitor(int_type_map=int_type_map).transform(circuit)
            noir_source = emitter.emit(noir_doc)
        except Exception as exc:
            print(f"  SKIP (Noir translation error): {exc}")
            skipped += 1
            continue

        # 3. Enumerate satisfying models with Z3
        try:
            models = enumerate_models(
                smt_content,
                declared,
                args.max_witnesses,
                timeout=args.z3_timeout,
                int_vars=int_vars,
                int_type_map=int_type_map,
                mul_factors=mul_factors,
            )
        except subprocess.TimeoutExpired:
            print("  SKIP (Z3 timed out)")
            skipped += 1
            continue
        except FileNotFoundError:
            sys.exit("Error: 'z3' not found in PATH.")
        except Exception as exc:
            print(f"  SKIP (Z3 error): {exc}")
            skipped += 1
            continue

        if not models:
            # UNSAT oracle: run N boundary-value samples — all must be rejected.
            seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 1
            rng_unsat = random.Random(seed)
            for idx in range(args.max_witnesses):
                folder_name = f"obj.{stem}-{idx}"
                pkg_name    = _sanitize_name(folder_name)
                project_dir = unsat_dir / folder_name
                boundary_toml = build_boundary_prover_toml(circuit, int_type_map, rng_unsat, mul_factors)
                create_noir_project(project_dir, noir_source, boundary_toml, pkg_name,
                                    smt_source=smt_content, smt_filename=smt_filename)
                print(f"  [{folder_name}] unsat boundary  ... ", end="", flush=True)
                timed_out = False
                try:
                    u_ok, u_out = run_nargo_execute(project_dir, timeout=args.nargo_timeout)
                except subprocess.TimeoutExpired:
                    timed_out, u_ok, u_out = True, False, ""
                except FileNotFoundError:
                    sys.exit("Error: 'nargo' not found in PATH.")
                if u_ok:
                    print("BUG")
                    n_error += 1
                    save_error(errors_dir, "error", folder_name + "_unsat_oracle",
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT ORACLE VIOLATION: nargo accepted boundary input\n" + u_out)
                elif timed_out:
                    print("TIMEOUT")
                    n_timeout += 1
                else:
                    print("ok (correctly rejected)")
            continue

        print(f"  {len(models)} satisfying model(s) found")

        # 4. For each model: create Noir project + run nargo execute
        for idx, model in enumerate(models):
            folder_name = f"obj.{stem}-{idx}"
            pkg_name = _sanitize_name(folder_name)
            project_dir = sat_dir / folder_name

            prover_toml = model_to_prover_toml(model, circuit)
            create_noir_project(
                project_dir, noir_source, prover_toml, pkg_name,
                smt_source=smt_content, smt_filename=smt_filename,
            )

            print(f"  [{folder_name}] nargo execute   ... ", end="", flush=True)
            timed_out = False
            try:
                success, output = run_nargo_execute(project_dir, timeout=args.nargo_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                success = False
                output = ""
            except FileNotFoundError:
                sys.exit("Error: 'nargo' not found in PATH.")

            if success:
                witness_path = project_dir / "target" / f"{pkg_name}.gz"
                print(f"OK  →  {witness_path}")
                ok += 1

                # SAT oracle: find a minimal UNSAT modification and assert nargo rejects it.
                oracle_model = find_sat_oracle_input(
                    smt_content, circuit, model, int_vars, int_type_map)
                if oracle_model is not None:
                    oracle_toml = model_to_prover_toml(oracle_model, circuit)
                    oracle_dir = sat_to_unsat_dir / folder_name
                    create_noir_project(oracle_dir, noir_source, oracle_toml, pkg_name,
                                        smt_source=smt_content, smt_filename=smt_filename)
                    print(f"  [{folder_name}] sat oracle      ... ", end="", flush=True)
                    try:
                        o_ok, o_out = run_nargo_execute(oracle_dir, timeout=args.nargo_timeout)
                    except subprocess.TimeoutExpired:
                        o_ok, o_out = False, ""
                    if o_ok:
                        print("BUG")
                        n_error += 1
                        save_error(errors_dir, "error", folder_name + "_sat_oracle",
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT ORACLE VIOLATION: nargo accepted a Z3-UNSAT input\n" + o_out)
                    else:
                        print("ok (correctly rejected)")
                else:
                    print(f"  [{folder_name}] sat oracle      ... skipped (no boolean flip found UNSAT)")

            elif timed_out:
                print("TIMEOUT")
                n_timeout += 1
                save_error(errors_dir, "timeout", folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, "")
            else:
                print("ERROR")
                first_error = next(
                    (l for l in output.splitlines() if "error" in l.lower()),
                    output.splitlines()[0] if output.strip() else "(no output)"
                )
                print(f"    {first_error}")
                n_error += 1
                save_error(errors_dir, "error", folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, output)

    print(f"\n{'='*60}")
    print(f"Results: {ok} ok  |  {n_error} error  |  {n_timeout} timeout  |  {skipped} skipped")


if __name__ == "__main__":
    main()
