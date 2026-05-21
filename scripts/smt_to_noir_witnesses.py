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
from src.backends.noir.ir2noir import IR2NoirVisitor, _fold_expr_constant, recompute_types
from src.backends.noir.emitter import EmitVisitor
from src.backends.noir.types import IntegerType
from src.ir.nodes import VariableType, Variable, Circuit, BinaryExpression, Integer as IRInteger
from src.witnesses.generator import find_witness


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


def ir_expr_to_smt(expr) -> str | None:
    """Convert an IR expression to its SMT-LIB2 string, or None if not convertible."""
    from src.ir.nodes import Integer as IRInt, Variable as IRVar, BinaryExpression as IRBin, \
        UnaryExpression as IRUnary, TernaryExpression as IRTernary, Operator as IROp, VariableType
    if isinstance(expr, IRInt):
        return f"(- {abs(expr.value)})" if expr.value < 0 else str(expr.value)
    if isinstance(expr, IRVar):
        return expr.name if expr.variable_type == VariableType.INTEGER else None
    if isinstance(expr, IRUnary) and expr.op == IROp.SUB:
        inner = ir_expr_to_smt(expr.value)
        return f"(- {inner})" if inner is not None else None
    if isinstance(expr, IRBin):
        # Comparison operators for ternary conditions
        cmp_ops = {IROp.EQU: '=', IROp.NEQ: None, IROp.LTH: '<',
                   IROp.LEQ: '<=', IROp.GTH: '>', IROp.GEQ: '>='}
        arith_ops = {IROp.ADD: '+', IROp.SUB: '-', IROp.MUL: '*'}
        smt_op = arith_ops.get(expr.op) or cmp_ops.get(expr.op)
        if smt_op is None:
            return None
        l = ir_expr_to_smt(expr.lhs)
        r = ir_expr_to_smt(expr.rhs)
        return f"({smt_op} {l} {r})" if l is not None and r is not None else None
    if isinstance(expr, IRTernary):
        cond = ir_expr_to_smt(expr.condition)
        if_s = ir_expr_to_smt(expr.if_expr)
        else_s = ir_expr_to_smt(expr.else_expr)
        if cond is not None and if_s is not None and else_s is not None:
            return f"(ite {cond} {if_s} {else_s})"
    return None


def smt_expr_type(expr, int_type_map: dict):
    """Standalone type inference for IR expressions — mirrors ir2noir._expr_type. Never returns None."""
    from src.ir.nodes import Integer as IRInt, Variable as IRVar, BinaryExpression as IRBin, \
        UnaryExpression as IRUnary, TernaryExpression as IRTernary, Operator as IROp, VariableType
    from src.backends.noir.ir2noir import _common_int_type, _min_signed_type, \
        _UNSIGNED_NEG_PROMOTION, _fold_expr_constant
    from src.backends.noir.types import IntegerType, BoolType
    if isinstance(expr, IRVar):
        if expr.variable_type == VariableType.INTEGER:
            return int_type_map.get(expr.name, IntegerType(64, True))
        return BoolType()
    if isinstance(expr, IRInt):
        return _min_signed_type(expr.value)
    if isinstance(expr, IRUnary):
        if expr.op == IROp.SUB:
            t = smt_expr_type(expr.value, int_type_map)
            if isinstance(t, IntegerType) and not t.signed:
                return IntegerType(_UNSIGNED_NEG_PROMOTION.get(t.bits, 64), True)
            return t
        return BoolType()
    if isinstance(expr, IRTernary):
        lt = smt_expr_type(expr.if_expr, int_type_map)
        rt = smt_expr_type(expr.else_expr, int_type_map)
        if isinstance(lt, IntegerType) and isinstance(rt, IntegerType):
            return _common_int_type(lt, rt)
        if isinstance(lt, IntegerType):
            return lt
        if isinstance(rt, IntegerType):
            return rt
        return BoolType()
    if isinstance(expr, IRBin) and expr.op in (IROp.ADD, IROp.SUB, IROp.MUL):
        val = _fold_expr_constant(expr)
        if val is not None:
            return _min_signed_type(val)
        lt = smt_expr_type(expr.lhs, int_type_map)
        rt = smt_expr_type(expr.rhs, int_type_map)
        if isinstance(lt, IntegerType) and isinstance(rt, IntegerType):
            return _common_int_type(lt, rt)
        if isinstance(lt, IntegerType):
            return lt
        if isinstance(rt, IntegerType):
            return rt
        return BoolType()
    return BoolType()


def build_type_bounds(circuit) -> list[str]:
    """
    Walk every expression in the circuit. For each integer-typed subexpression,
    add (assert (>= expr MIN)) and (assert (<= expr MAX)) using the type stored
    on expr.noir_type (set by recompute_types).
    """
    from src.ir.nodes import (
        Expression as IRExpr, Integer as IRNodes_Integer, BinaryExpression as IRBin,
        UnaryExpression as IRUnary, TernaryExpression as IRTernary, Assertion, Assume,
        Assignment, Operator as IROp,
    )
    from src.backends.noir.types import IntegerType
    clauses: list[str] = []
    seen: set[str] = set()

    def _walk(node) -> None:
        if isinstance(node, IRBin):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, IRUnary):
            _walk(node.value)
        elif isinstance(node, IRTernary):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)

        if not isinstance(node, IRExpr):
            return
        if isinstance(node, IRNodes_Integer):
            return  # constants are always within their own range by construction
        t = node.noir_type
        if not isinstance(t, IntegerType):
            return
        smt_str = ir_expr_to_smt(node)
        if smt_str is None or smt_str in seen:
            return
        seen.add(smt_str)
        lo, hi = _type_bounds(t)
        clauses.append(f"(assert (>= {smt_str} {lo}))")
        clauses.append(f"(assert (<= {smt_str} {hi}))")

    for stmt in circuit.statements:
        if isinstance(stmt, Assertion):
            _walk(stmt.value)
        elif isinstance(stmt, Assume):
            _walk(stmt.condition)
        elif isinstance(stmt, Assignment):
            _walk(stmt.rhs)
    return clauses


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

    result: dict[str, IntegerType] = {}
    for name in int_var_names:
        min_c, max_c = const_ranges.get(name, (0, 0))
        valid = [
            t for t in _INT_TYPE_POOL
            if _type_bounds(t)[0] <= min_c
            and _type_bounds(t)[1] >= max_c
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


def build_safe_math_source(fns_needed: set[tuple[str, int, bool]]) -> str:
    """Generate safe_math.nr source — only abs functions are emitted."""
    lines: list[str] = []
    for op_name, bits, signed in sorted(fns_needed):
        if op_name != "abs":
            continue
        type_str = f"{'i' if signed else 'u'}{bits}"
        fn_name = f"{op_name}_{type_str}"
        if not signed:
            lines += [
                f"pub fn {fn_name}(a: {type_str}) -> {type_str} {{",
                f"    a",
                f"}}",
                "",
            ]
        else:
            lines += [
                f"pub fn {fn_name}(a: {type_str}) -> {type_str} {{",
                f"    if a >= 0 {{ a }} else {{ -a }}",
                f"}}",
                "",
            ]
    return "\n".join(lines)




def _solver_cmd(solver: str) -> list[str]:
    if solver == "cvc5":
        return ["cvc5", "--produce-models", "--lang=smt2", "-"]
    return ["z3", "-in"]


def enumerate_models(
    smt_content: str,
    declared: set[str],
    max_models: int,
    timeout: int = 30,
    circuit=None,
    solver: str = "z3",
) -> list[dict[str, tuple[str, str]]]:
    """
    Call the SMT solver repeatedly with blocking clauses to collect up to max_models
    satisfying assignments, bounded by the type of every integer subexpression.
    """
    base = _strip_check_commands(smt_content)
    bounds = build_type_bounds(circuit) if circuit is not None else []
    extra: list[str] = bounds[:]
    models: list[dict[str, tuple[str, str]]] = []

    for _ in range(max_models):
        query = base + "\n" + "\n".join(extra) + "\n(check-sat)\n(get-model)\n"
        proc = subprocess.run(
            _solver_cmd(solver),
            input=query,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Raise on unexpected solver errors so they get saved to the errors directory.
        # "model is not available" is expected when the solver proves UNSAT — not a real error.
        _EXPECTED_SOLVER_ERRORS = (
            "model is not available",          # Z3: UNSAT then get-model
            "cannot get model unless after",   # cvc5: UNSAT then get-model
        )
        solver_errors = [
            l for l in proc.stdout.splitlines()
            if l.strip().startswith("(error")
            and not any(e in l for e in _EXPECTED_SOLVER_ERRORS)
        ]
        if solver_errors:
            raise RuntimeError(f"{solver} error: {solver_errors[0].strip()}")

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

def _var_toml_value(var, model: dict[str, tuple[str, str]] | None, boundary_val: str | None = None) -> str:
    """Return the TOML value string for a single variable."""
    name = var.name
    if var.variable_type == VariableType.BOOLEAN:
        if model and name in model:
            return model[name][1]
        return boundary_val or "false"
    else:
        if model and name in model:
            raw = model[name][1]
            if raw.startswith("ff"):
                raw = raw[2:]
            return f'"{raw}"'
        return f'"{boundary_val}"' if boundary_val else '"0"'


def model_to_prover_toml(
    model: dict[str, tuple[str, str]],
    circuit,
    struct_map: dict[str, tuple[str, str]] | None = None,
    array_map: dict[str, tuple[str, int]] | None = None,
    tuple_map: dict[str, tuple[str, int]] | None = None,
    nesting_map: dict[str, tuple[str, str | int]] | None = None,
    param_name_map: dict[str, str] | None = None,
) -> str:
    """Convert a Z3 model to Noir's Prover.toml, respecting struct/array/tuple groupings."""
    sm = struct_map or {}
    am = array_map or {}
    tm = tuple_map or {}
    nm = nesting_map or {}
    pnm = param_name_map or {}  # original param → Noir param (may have _ prefix)

    def _pname(p: str) -> str:
        return pnm.get(p, p)

    array_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in am:
            param, idx = am[var.name]
            array_contents.setdefault(param, []).append((idx, var))

    tuple_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in tm:
            param, idx = tm[var.name]
            tuple_contents.setdefault(param, []).append((idx, var))

    # Params that are containers (not leaves) — must NOT be emitted in the flat loop.
    container_params: set[str] = {cont for _, (cont, _) in nm.items()}

    # Helper: render a leaf group as a TOML value string (for nesting into tuples).
    def _leaf_toml_value(leaf: str) -> str:
        if leaf in array_contents:
            members = sorted(array_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_var_toml_value(v, model) for _, v in members) + "]"
        if leaf in tuple_contents:
            members = sorted(tuple_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_var_toml_value(v, model) for _, v in members) + "]"
        # Struct rendered as TOML inline table.
        fields = [(f, v) for v in circuit.inputs
                  if v.name in sm and sm[v.name][0] == leaf
                  for _, f in [sm[v.name]]]
        if fields:
            inner = ", ".join(f"{f} = {_var_toml_value(v, model)}" for f, v in fields)
            return "{" + inner + "}"
        return '""'

    flat_lines: list[str] = []
    struct_lines: list[str] = []
    emitted: set[str] = set()

    # Emit flat/array/tuple top-level params (only those NOT nested inside something else).
    for var in circuit.inputs:
        name = var.name
        if name in am:
            param, _ = am[name]
            if param not in emitted and param not in nm and param not in container_params:
                emitted.add(param)
                members = sorted(array_contents[param], key=lambda x: x[0])
                vals = ", ".join(_var_toml_value(v, model) for _, v in members)
                flat_lines.append(f"{_pname(param)} = [{vals}]")
        elif name in tm:
            param, _ = tm[name]
            if param not in emitted and param not in nm and param not in container_params:
                emitted.add(param)
                members = sorted(tuple_contents[param], key=lambda x: x[0])
                vals = ", ".join(_var_toml_value(v, model) for _, v in members)
                flat_lines.append(f"{_pname(param)} = [{vals}]")
        elif name not in sm:
            flat_lines.append(f"{_pname(name)} = {_var_toml_value(var, model)}")

    # Build struct blocks, including nested leaves as struct fields.
    struct_blocks: dict[str, list[str]] = {}
    for var in circuit.inputs:
        name = var.name
        if name in sm:
            param, field = sm[name]
            if param not in nm:  # only top-level structs here
                struct_blocks.setdefault(_pname(param), []).append(f"{field} = {_var_toml_value(var, model)}")

    # Inject nested leaves into their struct containers.
    for leaf, (container, access_key) in nm.items():
        if not container.startswith("g"):
            continue  # tuple container handled below
        field_name = access_key  # str for struct containers
        leaf_val = _leaf_toml_value(leaf)
        struct_blocks.setdefault(_pname(container), []).append(f"{field_name} = {leaf_val}")

    for pkey, field_lines in struct_blocks.items():
        struct_lines.append(f"[{pkey}]")
        struct_lines.extend(field_lines)

    # Handle tuple containers: tuples that have structs nested inside.
    # Collect per-tuple: list of (index, value_str) to build the full tuple array.
    tuple_nest_extra: dict[str, dict[int, str]] = {}
    for leaf, (container, access_key) in nm.items():
        if not container.startswith("tup_"):
            continue
        idx = access_key  # int for tuple containers
        leaf_val = _leaf_toml_value(leaf)
        tuple_nest_extra.setdefault(container, {})[idx] = leaf_val

    for container, extra in tuple_nest_extra.items():
        if container in emitted:
            continue
        emitted.add(container)
        # Build tuple array: regular members + nested struct elements.
        base_members = sorted(tuple_contents.get(container, []), key=lambda x: x[0])
        # Combine: each position is either a nested leaf or a regular var.
        all_positions: list[str] = []
        base_idx = 0
        struct_idx = 0
        positions: dict[int, str] = {i: _var_toml_value(v, model) for i, v in base_members}
        # Renumber: nested structs occupy positions 0..len(extra)-1, regular vars follow.
        # Actually, emit struct elements first in the tuple, then vars.
        struct_vals = [extra[k] for k in sorted(extra.keys())]
        var_vals = [_var_toml_value(v, model) for _, v in base_members]
        all_vals = struct_vals + var_vals
        flat_lines.append(f"{_pname(container)} = [{', '.join(all_vals)}]")

    return "\n".join(flat_lines + struct_lines) + "\n"


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
    safe_math_source: str = "",
    z3_query: str = "",
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "main.nr").write_text(noir_source)
    if safe_math_source:
        (src_dir / "safe_math.nr").write_text(safe_math_source)
    (project_dir / "Prover.toml").write_text(prover_toml)
    (project_dir / "Nargo.toml").write_text(
        f'[package]\nname = "{package_name}"\ntype = "bin"\nauthors = [""]\n\n[dependencies]\n'
    )
    if smt_source and smt_filename:
        (project_dir / smt_filename).write_text(smt_source)
    if z3_query:
        (project_dir / "z3_query.smt2").write_text(z3_query)


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
    safe_math_source: str = "",
    z3_query: str = "",
) -> None:
    dest = errors_dir / category / folder_name
    create_noir_project(dest, noir_source, prover_toml, package_name,
                        smt_source=smt_content, smt_filename=smt_filename,
                        safe_math_source=safe_math_source, z3_query=z3_query)
    (dest / "nargo_output.txt").write_text(nargo_output)


def _is_assertion_failure(output: str) -> bool:
    """Return True only when nargo rejected a witness via circuit assertions.
    A compilation/linking failure (missing module, type error, etc.) is NOT
    a meaningful oracle rejection and must not be counted as one."""
    lower = output.lower()
    return "assertion failed" in lower or "failed assertion" in lower


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
# Variable grouping (structs + arrays)
# ---------------------------------------------------------------------------

def compute_groupings(
    circuit,
    smt_content: str,
    allow_nesting: bool = False,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, int]], dict[str, tuple[str, int]]]:
    """
    Randomly assign each input variable to one of:
      flat / struct_N / array_N (booleans only) / tuple_N (any type)

    Groups with < 2 members are demoted to flat.
    Seed = int(md5(smt_content), 16).

    Returns:
      struct_map : {var_name → (param_name, field_name)}
      array_map  : {var_name → (param_name, index)}
      tuple_map  : {var_name → (param_name, index)}
    """
    if len(circuit.inputs) < 4:
        return {}, {}, {}, {}

    seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16)
    rng = random.Random(seed)

    buckets = ["flat", "struct_0", "struct_1", "array_0", "array_1", "tuple_0", "tuple_1"]
    assignments: dict[str, str] = {}
    for var in circuit.inputs:
        choice = rng.choice(buckets)
        if choice.startswith("array") and var.variable_type != VariableType.BOOLEAN:
            choice = "flat"
        assignments[var.name] = choice

    # Demote singleton groups to flat.
    from collections import Counter
    counts = Counter(assignments.values())
    for name, bucket in list(assignments.items()):
        if bucket != "flat" and counts[bucket] < 2:
            assignments[name] = "flat"

    struct_map: dict[str, tuple[str, str]] = {}
    array_counters: dict[str, int] = {}
    array_map: dict[str, tuple[str, int]] = {}

    for var in circuit.inputs:
        bucket = assignments[var.name]
        if bucket.startswith("struct"):
            param = bucket.replace("_", "")   # struct_0 → struct0 → g0
            param = "g" + bucket[-1]          # g0 / g1
            struct_map[var.name] = (param, var.name)
        elif bucket.startswith("array"):
            param = "bools_" + bucket[-1]     # bools_0 / bools_1
            idx = array_counters.get(param, 0)
            array_counters[param] = idx + 1
            array_map[var.name] = (param, idx)

    tuple_counters: dict[str, int] = {}
    tuple_map: dict[str, tuple[str, int]] = {}
    for var in circuit.inputs:
        bucket = assignments[var.name]
        if bucket.startswith("tuple"):
            param = "tup_" + bucket[-1]
            idx = tuple_counters.get(param, 0)
            tuple_counters[param] = idx + 1
            tuple_map[var.name] = (param, idx)

    # Demote singleton tuple groups to flat.
    from collections import Counter
    tuple_counts = Counter(p for p, _ in tuple_map.values())
    tuple_map = {k: v for k, v in tuple_map.items() if tuple_counts[v[0]] >= 2}

    if not allow_nesting:
        return struct_map, array_map, tuple_map, {}

    # --- Second-level nesting (recursive mode only) ---
    # Randomly assign leaf groups as fields/elements of container groups.
    # nesting_map: {leaf_group → (container_group, access_key)}
    #   access_key is a field name (str) for structs, or an int index for tuples.
    # Rules:
    #   - Only struct_0 and struct_1 can be containers (structs are the only composite type)
    #   - OR tuple_0 / tuple_1 as containers (struct can be a tuple element)
    #   - A container cannot itself be nested
    #   - Each container holds at most one nested group per nesting run
    nesting_map: dict[str, tuple[str, str | int]] = {}

    # Collect which leaf groups exist (non-empty).
    leaf_groups = (
        [f"bools_{i}" for i in range(2) if any(v[0] == f"bools_{i}" for v in array_map.values())] +
        [f"g{i}"    for i in range(2) if any(v[0] == f"g{i}"    for v in struct_map.values())] +
        [f"tup_{i}" for i in range(2) if any(v[0] == f"tup_{i}" for v in tuple_map.values())]
    )
    struct_containers = [f"g{i}" for i in range(2) if any(v[0] == f"g{i}" for v in struct_map.values())]
    tuple_containers  = [f"tup_{i}" for i in range(2) if any(v[0] == f"tup_{i}" for v in tuple_map.values())]
    all_containers = struct_containers + tuple_containers

    used_as_container: set[str] = set()
    used_as_nested: set[str] = set()
    field_counters: dict[str, int] = {}  # how many fields added to each container
    tuple_nest_counters: dict[str, int] = {}

    for leaf in rng.sample(leaf_groups, len(leaf_groups)):  # random order
        if leaf in used_as_nested:
            continue
        eligible_containers = [c for c in all_containers
                                if c != leaf and c not in used_as_nested
                                and field_counters.get(c, 0) < 2]  # max 2 nested per container
        if not eligible_containers:
            continue
        if not rng.choice([True, False]):  # 50% chance of nesting
            continue
        container = rng.choice(eligible_containers)
        used_as_container.add(container)
        used_as_nested.add(leaf)
        field_counters[container] = field_counters.get(container, 0) + 1
        if container.startswith("tup_"):
            idx = tuple_nest_counters.get(container, 0)
            tuple_nest_counters[container] = idx + 1
            nesting_map[leaf] = (container, idx)
        else:
            # struct container: use a field name based on the leaf group
            field_name = leaf.replace("_", "").replace("bools", "arr").replace("tup", "tup")
            nesting_map[leaf] = (container, field_name)

    return struct_map, array_map, tuple_map, nesting_map


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
    struct_map: dict[str, tuple[str, str]] | None = None,
    array_map: dict[str, tuple[str, int]] | None = None,
    tuple_map: dict[str, tuple[str, int]] | None = None,
    nesting_map: dict[str, tuple[str, str | int]] | None = None,
    param_name_map: dict[str, str] | None = None,
) -> str:
    """Build a boundary-value Prover.toml respecting struct/array/tuple/nesting groupings."""
    sm = struct_map or {}
    am = array_map or {}
    tm = tuple_map or {}
    nm = nesting_map or {}
    pnm = param_name_map or {}

    def _pname(p: str) -> str:
        return pnm.get(p, p)

    def _bval(var) -> str:
        if var.variable_type == VariableType.BOOLEAN:
            return rng.choice(["true", "false"])
        t = int_type_map.get(var.name, IntegerType(64, True))
        lo, hi = _type_bounds(t)
        pool = sorted(set(v for v in [lo, lo+1, -1, 0, 1, hi-1, hi] if lo <= v <= hi))
        return str(rng.choice(pool))

    def _bfmt(var) -> str:
        v = _bval(var)
        return v if var.variable_type == VariableType.BOOLEAN else f'"{v}"'

    array_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in am:
            param, idx = am[var.name]
            array_contents.setdefault(param, []).append((idx, var))

    tuple_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in tm:
            param, idx = tm[var.name]
            tuple_contents.setdefault(param, []).append((idx, var))

    def _leaf_boundary(leaf: str) -> str:
        if leaf in array_contents:
            members = sorted(array_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_bval(v) for _, v in members) + "]"
        if leaf in tuple_contents:
            members = sorted(tuple_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_bfmt(v) for _, v in members) + "]"
        fields = [(f, v) for v in circuit.inputs
                  if v.name in sm and sm[v.name][0] == leaf
                  for _, f in [sm[v.name]]]
        if fields:
            inner = ", ".join(f"{f} = {_bfmt(v)}" for f, v in fields)
            return "{" + inner + "}"
        return '""'

    container_params_b: set[str] = {cont for _, (cont, _) in nm.items()}

    flat_lines: list[str] = []
    struct_lines: list[str] = []
    emitted: set[str] = set()

    for var in circuit.inputs:
        name = var.name
        if name in am:
            param, _ = am[name]
            if param not in emitted and param not in nm and param not in container_params_b:
                emitted.add(param)
                members = sorted(array_contents[param], key=lambda x: x[0])
                flat_lines.append(f"{_pname(param)} = [{', '.join(_bval(v) for _, v in members)}]")
        elif name in tm:
            param, _ = tm[name]
            if param not in emitted and param not in nm and param not in container_params_b:
                emitted.add(param)
                members = sorted(tuple_contents[param], key=lambda x: x[0])
                flat_lines.append(f"{_pname(param)} = [{', '.join(_bfmt(v) for _, v in members)}]")
        elif name not in sm:
            flat_lines.append(f"{_pname(name)} = {_bfmt(var)}")

    struct_blocks: dict[str, list[str]] = {}
    for var in circuit.inputs:
        name = var.name
        if name in sm:
            param, field = sm[name]
            if param not in nm:
                struct_blocks.setdefault(_pname(param), []).append(f"{field} = {_bfmt(var)}")

    for leaf, (container, access_key) in nm.items():
        if container.startswith("g"):
            struct_blocks.setdefault(_pname(container), []).append(f"{access_key} = {_leaf_boundary(leaf)}")

    for pkey, field_lines in struct_blocks.items():
        struct_lines.append(f"[{pkey}]")
        struct_lines.extend(field_lines)

    for leaf, (container, access_key) in nm.items():
        if container.startswith("tup_") and container not in emitted:
            emitted.add(container)
            struct_vals = [_leaf_boundary(leaf)]
            var_vals = [_bfmt(v) for _, v in sorted(tuple_contents.get(container, []), key=lambda x: x[0])]
            flat_lines.append(f"{_pname(container)} = [{', '.join(struct_vals + var_vals)}]")

    return "\n".join(flat_lines + struct_lines) + "\n"


def _is_unsat_with_fix(
    smt_content: str,
    var_name: str,
    fixed_value: str,
    circuit,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Return (is_unsat, z3_query) — query saved for reproduction if oracle fires."""
    base = _strip_check_commands(smt_content)
    bounds = build_type_bounds(circuit)
    fix = f"(assert (= {var_name} {fixed_value}))"
    query = base + "\n" + "\n".join(bounds) + "\n" + fix + "\n(check-sat)\n"
    proc = subprocess.run(
        ["z3", "-in"], input=query,
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout.strip().startswith("unsat"), query


_INT_FLIP_DELTAS = [1, -1]


def find_sat_oracle_input(
    smt_content: str,
    circuit,
    model: dict,
    int_type_map: dict[str, IntegerType],
) -> tuple[dict, str] | None:
    """
    Try minimal single-variable modifications that Z3 confirms make the formula UNSAT.
    Returns (modified_model, z3_query) or None.
    """
    # --- Boolean flips ---
    for var in circuit.inputs:
        if var.variable_type != VariableType.BOOLEAN:
            continue
        current = model.get(var.name, ('Bool', 'false'))[1]
        flipped = 'false' if current == 'true' else 'true'
        try:
            is_unsat, query = _is_unsat_with_fix(smt_content, var.name, flipped, circuit)
            if is_unsat:
                return {**model, var.name: ('Bool', flipped)}, query
        except subprocess.TimeoutExpired:
            continue

    # --- Integer mutations (±deltas) ---
    for var in circuit.inputs:
        if var.variable_type != VariableType.INTEGER:
            continue
        raw = model.get(var.name, ('Int', '0'))[1]
        try:
            current_val = int(raw)
        except ValueError:
            continue
        t = int_type_map.get(var.name, IntegerType(64, True))
        lo, hi = _type_bounds(t)
        for delta in _INT_FLIP_DELTAS:
            new_val = current_val + delta
            if not (lo <= new_val <= hi):
                continue
            try:
                is_unsat, query = _is_unsat_with_fix(smt_content, var.name, str(new_val), circuit)
                if is_unsat:
                    return {**model, var.name: ('Int', str(new_val))}, query
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
        "--generator-config",
        default="generator_config.json",
        metavar="FILE",
        help="JSON file with generator options (default: generator_config.json).",
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

    gen_config: dict = {}
    gen_config_path = Path(args.generator_config)
    if gen_config_path.exists():
        with gen_config_path.open() as f:
            gen_config = json.load(f)
    variables_bundling = gen_config.get("variables_bundling", "simple")
    if variables_bundling not in ("flat", "simple", "recursive"):
        sys.exit(f"Error: variables_bundling must be flat, simple or recursive, got '{variables_bundling}'")
    solver = gen_config.get("solver", "z3")
    if solver not in ("z3", "cvc5", "z3_to_cvc5"):
        sys.exit(f"Error: solver must be z3, cvc5, or z3_to_cvc5, got '{solver}'")
    # CLI args override config values; config values override hardcoded defaults.
    if args.z3_timeout == 30:      # still at default → use config
        args.z3_timeout = gen_config.get("solver_timeout", 30)
    if args.nargo_timeout == 120:  # still at default → use config
        args.nargo_timeout = gen_config.get("nargo_timeout", 120)


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
    (errors_dir / "overflow").mkdir(parents=True, exist_ok=True)
    (errors_dir / "timeout").mkdir(parents=True, exist_ok=True)
    (errors_dir / "smt_solver_errors").mkdir(parents=True, exist_ok=True)
    (errors_dir / "z3_error_cvc5_succ").mkdir(parents=True, exist_ok=True)
    (errors_dir / "sat_to_unsat_pipeline").mkdir(parents=True, exist_ok=True)
    (errors_dir / "unsat_pipeline").mkdir(parents=True, exist_ok=True)

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
    sat_ok = sat_error = sat_overflow = 0
    unsat_ok = unsat_error = unsat_pipeline_error = 0
    oracle_ok = oracle_bug = oracle_skip = oracle_pipeline_error = 0
    n_timeout = skipped = 0

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
            print(f"  ERROR (parse error): {exc}")
            sat_error += 1
            # Store as error with only the smt file (no Noir source yet).
            dest = errors_dir / "error" / f"obj.{stem}-parse"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(f"PARSE ERROR: {exc}\n")
            continue

        declared = {v.name for v in circuit.inputs}
        int_var_names = [v.name for v in circuit.inputs if v.variable_type == VariableType.INTEGER]

        int_type_map: dict[str, IntegerType] = {}
        if int_var_names:
            int_type_map = assign_int_types(int_var_names, smt_content, circuit)
            types_summary = ", ".join(f"{n}:{t}" for n, t in int_type_map.items())
            print(f"  Types: {types_summary}")
        const_type_rng = random.Random(int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 5)
        recompute_types(circuit, int_type_map, rng=const_type_rng)

        if variables_bundling == "flat":
            struct_map, array_map, tuple_map, nesting_map = {}, {}, {}, {}
        else:
            struct_map, array_map, tuple_map, nesting_map = compute_groupings(
                circuit, smt_content,
                allow_nesting=(variables_bundling == "recursive"),
            )
        # Seeded RNG for per-expression randomization (comptime, helpers).
        # Offset by 3 to be independent of the grouping seed (offset 0) and
        # the UNSAT boundary seed (offset 1).
        ir_rng = random.Random(int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 3)

        # 2. Circuit IR → Noir source
        try:
            visitor = IR2NoirVisitor(
                int_type_map=int_type_map,
                struct_map=struct_map,
                array_map=array_map,
                tuple_map=tuple_map,
                nesting_map=nesting_map,
                rng=ir_rng,
            )
            noir_doc = visitor.transform(circuit)
            noir_source = emitter.emit(noir_doc)
            safe_math_src = build_safe_math_source(visitor.safe_fns_needed) if visitor.safe_fns_needed else ""
            param_name_map = visitor.param_name_map
        except Exception as exc:
            import traceback
            print(f"  ERROR (Noir translation error): {exc}")
            sat_error += 1
            dest = errors_dir / "error" / f"obj.{stem}-translation"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(
                f"TRANSLATION ERROR: {exc}\n\n{traceback.format_exc()}"
            )
            continue

        # Build the Z3 query for SAT model enumeration (saved alongside any SAT errors
        # so they can be reproduced independently outside the fuzzer).
        _type_bounds_clauses = build_type_bounds(circuit)
        _z3_sat_query = (
            _strip_check_commands(smt_content)
            + "\n"
            + "\n".join(_type_bounds_clauses)
            + "\n(check-sat)\n(get-model)\n"
        )

        # 3. Find a satisfying witness using the witness generator module.
        z3_failed_error: Exception | None = None
        active_solver = solver
        models: list[dict] = []

        def _run_solver(s: str) -> tuple[list[dict], str]:
            model, query = find_witness(smt_content, circuit, solver=s, timeout=args.z3_timeout)
            return ([model] if model is not None else []), query

        try:
            if solver == "z3_to_cvc5":
                try:
                    models, _z3_sat_query = _run_solver("z3")
                    active_solver = "z3"
                except Exception as z3_exc:
                    z3_failed_error = z3_exc
                    print(f"  Z3 failed ({z3_exc}), retrying with cvc5 ...")
                    models, _z3_sat_query = _run_solver("cvc5")
                    active_solver = "cvc5"
            else:
                models, _z3_sat_query = _run_solver(solver)
                active_solver = solver
        except subprocess.TimeoutExpired:
            print(f"  ERROR (solver timed out)")
            n_timeout += 1
            continue
        except FileNotFoundError as exc:
            sys.exit(f"Error: solver not found in PATH: {exc}")
        except Exception as exc:
            label = "z3_then_cvc5" if solver == "z3_to_cvc5" else active_solver
            print(f"  ERROR ({label} error): {exc}")
            sat_error += 1
            dest = errors_dir / "smt_solver_errors" / f"obj.{stem}-{label}error"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "src").mkdir(exist_ok=True)
            (dest / "src" / "main.nr").write_text(noir_source)
            (dest / "nargo_output.txt").write_text(f"SOLVER ERROR: {exc}\n")
            continue

        # If Z3 failed but cvc5 succeeded — save to dedicated folder for analysis.
        if z3_failed_error is not None:
            dest = errors_dir / "z3_error_cvc5_succ" / f"obj.{stem}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(
                f"Z3 ERROR: {z3_failed_error}\ncvc5 found {len(models)} model(s)\n"
            )
            # Continue processing — use the cvc5 models for witness generation.

        if not models:
            # UNSAT oracle: run N boundary-value samples — all must be rejected.
            seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 1
            rng_unsat = random.Random(seed)
            # Build the exact Z3 query that declared this formula UNSAT so it can
            # be reproduced independently (saved alongside any oracle violations).
            _z3_unsat_query = (
                _strip_check_commands(smt_content)
                + "\n"
                + "\n".join(_type_bounds_clauses)
                + "\n(check-sat)\n(get-model)\n"
            )
            for idx in range(args.max_witnesses):
                folder_name = f"obj.{stem}-{idx}"
                pkg_name    = _sanitize_name(folder_name)
                project_dir = unsat_dir / folder_name
                boundary_toml = build_boundary_prover_toml(circuit, int_type_map, rng_unsat, struct_map, array_map, tuple_map, nesting_map, param_name_map)
                create_noir_project(project_dir, noir_source, boundary_toml, pkg_name,
                                    smt_source=smt_content, smt_filename=smt_filename,
                                    safe_math_source=safe_math_src, z3_query=_z3_unsat_query)
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
                    unsat_error += 1
                    dest_path = errors_dir / "error" / (folder_name + "_unsat_oracle")
                    save_error(errors_dir, "error", folder_name + "_unsat_oracle",
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT ORACLE VIOLATION: nargo accepted boundary input\n" + u_out,
                               safe_math_src, _z3_unsat_query)
                elif timed_out:
                    print("TIMEOUT")
                    n_timeout += 1
                elif _is_assertion_failure(u_out):
                    print("ok (correctly rejected)")
                    unsat_ok += 1
                else:
                    print("PIPELINE ERROR")
                    unsat_pipeline_error += 1
                    save_error(errors_dir, "unsat_pipeline", folder_name,
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT PIPELINE ERROR: nargo failed but not via assertion\n" + u_out,
                               safe_math_src, _z3_unsat_query)
            continue

        print(f"  {len(models)} satisfying model(s) found")

        # 4. For each model: create Noir project + run nargo execute
        for idx, model in enumerate(models):
            folder_name = f"obj.{stem}-{idx}"
            pkg_name = _sanitize_name(folder_name)
            project_dir = sat_dir / folder_name

            prover_toml = model_to_prover_toml(model, circuit, struct_map, array_map, tuple_map, nesting_map, param_name_map)
            create_noir_project(
                project_dir, noir_source, prover_toml, pkg_name,
                smt_source=smt_content, smt_filename=smt_filename,
                safe_math_source=safe_math_src, z3_query=_z3_sat_query,
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
                sat_ok += 1

                # SAT oracle: find a minimal UNSAT modification and assert nargo rejects it.
                oracle_result = find_sat_oracle_input(
                    smt_content, circuit, model, int_type_map)
                if oracle_result is not None:
                    oracle_model, oracle_z3_query = oracle_result
                    oracle_toml = model_to_prover_toml(oracle_model, circuit, struct_map, array_map, tuple_map, nesting_map, param_name_map)
                    oracle_dir = sat_to_unsat_dir / folder_name
                    create_noir_project(oracle_dir, noir_source, oracle_toml, pkg_name,
                                        smt_source=smt_content, smt_filename=smt_filename,
                                        safe_math_source=safe_math_src, z3_query=oracle_z3_query)
                    print(f"  [{folder_name}] sat oracle      ... ", end="", flush=True)
                    try:
                        o_ok, o_out = run_nargo_execute(oracle_dir, timeout=args.nargo_timeout)
                    except subprocess.TimeoutExpired:
                        o_ok, o_out = False, ""
                    if o_ok:
                        print("BUG")
                        oracle_bug += 1
                        save_error(errors_dir, "error", folder_name + "_sat_oracle",
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT ORACLE VIOLATION: nargo accepted a Z3-UNSAT input\n" + o_out,
                                   safe_math_src, oracle_z3_query)
                    elif _is_assertion_failure(o_out):
                        print("ok (correctly rejected)")
                        oracle_ok += 1
                    else:
                        print("PIPELINE ERROR")
                        oracle_pipeline_error += 1
                        save_error(errors_dir, "sat_to_unsat_pipeline", folder_name,
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT_TO_UNSAT PIPELINE ERROR: nargo failed but not via assertion\n" + o_out,
                                   safe_math_src, oracle_z3_query)
                else:
                    print(f"  [{folder_name}] sat oracle      ... skipped (no UNSAT mutation found)")
                    oracle_skip += 1

            elif timed_out:
                print("TIMEOUT")
                n_timeout += 1
                save_error(errors_dir, "timeout", folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, "",
                           safe_math_src, _z3_sat_query)
            else:
                is_overflow = "overflow" in output.lower()
                category = "overflow" if is_overflow else "error"
                print("OVERFLOW" if is_overflow else "ERROR")
                first_error = next(
                    (l for l in output.splitlines() if "error" in l.lower()),
                    output.splitlines()[0] if output.strip() else "(no output)"
                )
                print(f"    {first_error}")
                if is_overflow:
                    sat_overflow += 1
                else:
                    sat_error += 1
                save_error(errors_dir, category, folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, output,
                           safe_math_src, _z3_sat_query)

    total_bugs = sat_error + unsat_error + oracle_bug
    print(f"\n{'='*60}")
    print(f"sat         : {sat_ok} ok  |  {sat_error} error  |  {sat_overflow} overflow")
    print(f"unsat       : {unsat_ok} ok  |  {unsat_error} bug  |  {unsat_pipeline_error} pipeline")
    print(f"sat_to_unsat: {oracle_ok} ok  |  {oracle_bug} bug  |  {oracle_skip} no-flip  |  {oracle_pipeline_error} pipeline")
    print(f"timeout     : {n_timeout}")
    print(f"{'─'*40}")
    print(f"total bugs  : {total_bugs}  (overflows not counted as bugs)")


if __name__ == "__main__":
    main()
