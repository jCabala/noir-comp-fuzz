"""
Witness generation: given an SMT formula and its Circuit IR, find a satisfying
variable assignment bounded to the circuit's Noir types.
"""

import hashlib
import random
import re
import subprocess
from pathlib import Path

from src.ir.nodes import Circuit, VariableType
from src.backends.noir.ir2noir import recompute_types, _UNSIGNED_NEG_PROMOTION
from src.backends.noir.types import IntegerType, BoolType


# ---------------------------------------------------------------------------
# Integer type pool and bounds
# ---------------------------------------------------------------------------

_INT_TYPE_POOL = [
    IntegerType(8,  True),   # i8
    IntegerType(16, True),   # i16
    IntegerType(32, True),   # i32
    IntegerType(64, True),   # i64
    IntegerType(8,  False),  # u8
    IntegerType(16, False),  # u16
    IntegerType(32, False),  # u32
    IntegerType(64, False),  # u64  (capped at i64 max for safe negation)
]


def type_bounds(t: IntegerType) -> tuple[int, int]:
    """Return the (lo, hi) Z3 integer bounds for a Noir integer type."""
    if t.signed:
        return -(2 ** (t.bits - 1)), (2 ** (t.bits - 1)) - 1
    else:
        return 0, (2 ** min(t.bits, 63)) - 1


# ---------------------------------------------------------------------------
# Type assignment
# ---------------------------------------------------------------------------

def _collect_var_const_ranges(circuit: Circuit) -> dict[str, tuple[int, int]]:
    """For each integer variable, find the min/max constants it's directly compared with."""
    from src.ir.nodes import BinaryExpression, Variable, Integer, Operator
    ranges: dict[str, tuple[int, int]] = {}

    def _update(name: str, val: int) -> None:
        lo, hi = ranges.get(name, (0, 0))
        ranges[name] = (min(lo, val), max(hi, val))

    def _walk(node) -> None:
        from src.ir.nodes import UnaryExpression, TernaryExpression, Assertion, Assume, Assignment
        if isinstance(node, BinaryExpression):
            if node.op in (Operator.ADD, Operator.SUB, Operator.EQU, Operator.NEQ,
                           Operator.LTH, Operator.LEQ, Operator.GTH, Operator.GEQ):
                if isinstance(node.lhs, Variable) and isinstance(node.rhs, Integer):
                    _update(node.lhs.name, node.rhs.value)
                if isinstance(node.rhs, Variable) and isinstance(node.lhs, Integer):
                    _update(node.rhs.name, node.lhs.value)
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, UnaryExpression):
            _walk(node.value)
        elif isinstance(node, TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, Assertion):
            _walk(node.value)
        elif isinstance(node, Assume):
            _walk(node.condition)
        elif isinstance(node, Assignment):
            _walk(node.rhs)

    for stmt in circuit.statements:
        _walk(stmt)
    return ranges


def assign_int_types(
    int_var_names: list[str],
    smt_content: str,
    circuit: Circuit,
) -> dict[str, IntegerType]:
    """Deterministically assign a Noir integer type to each integer variable."""
    seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    const_ranges = _collect_var_const_ranges(circuit)

    result: dict[str, IntegerType] = {}
    for name in int_var_names:
        min_c, max_c = const_ranges.get(name, (0, 0))
        valid = [
            t for t in _INT_TYPE_POOL
            if type_bounds(t)[0] <= min_c and type_bounds(t)[1] >= max_c
        ]
        result[name] = rng.choice(valid) if valid else IntegerType(64, True)
    return result


# ---------------------------------------------------------------------------
# Expression → SMT string
# ---------------------------------------------------------------------------

def _expr_to_smt(expr) -> str | None:
    """Convert an IR expression to its SMT-LIB2 string, or None if not representable."""
    from src.ir.nodes import (Integer, Variable, BinaryExpression, UnaryExpression,
                               TernaryExpression, Operator, VariableType)
    if isinstance(expr, Integer):
        return f"(- {abs(expr.value)})" if expr.value < 0 else str(expr.value)
    if isinstance(expr, Variable):
        return expr.name if expr.variable_type == VariableType.INTEGER else None
    if isinstance(expr, UnaryExpression) and expr.op == Operator.SUB:
        inner = _expr_to_smt(expr.value)
        return f"(- {inner})" if inner is not None else None
    if isinstance(expr, BinaryExpression):
        ops = {Operator.ADD: '+', Operator.SUB: '-', Operator.MUL: '*',
               Operator.EQU: '=', Operator.LTH: '<', Operator.LEQ: '<=',
               Operator.GTH: '>', Operator.GEQ: '>='}
        smt_op = ops.get(expr.op)
        if smt_op is None:
            return None
        l, r = _expr_to_smt(expr.lhs), _expr_to_smt(expr.rhs)
        return f"({smt_op} {l} {r})" if l is not None and r is not None else None
    if isinstance(expr, TernaryExpression):
        c = _expr_to_smt(expr.condition)
        t = _expr_to_smt(expr.if_expr)
        e = _expr_to_smt(expr.else_expr)
        return f"(ite {c} {t} {e})" if all(x is not None for x in (c, t, e)) else None
    return None


# ---------------------------------------------------------------------------
# Type bounds for all integer subexpressions
# ---------------------------------------------------------------------------

def build_type_bounds(circuit: Circuit) -> list[str]:
    """
    For every integer-typed subexpression in the circuit, emit
    (assert (>= expr MIN)) and (assert (<= expr MAX)).
    Requires recompute_types() to have been called first.
    """
    from src.ir.nodes import (Expression, BinaryExpression, UnaryExpression,
                               TernaryExpression, Assertion, Assume, Assignment)
    clauses: list[str] = []
    seen: set[str] = set()

    def _walk(node) -> None:
        if isinstance(node, BinaryExpression):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, UnaryExpression):
            _walk(node.value)
        elif isinstance(node, TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)

        if not isinstance(node, Expression):
            return
        from src.ir.nodes import Integer as IRInteger_
        if isinstance(node, IRInteger_):
            return  # constants are always within their own range by construction
        t = node.noir_type
        if not isinstance(t, IntegerType):
            return
        smt_str = _expr_to_smt(node)
        if smt_str is None or smt_str in seen:
            return
        seen.add(smt_str)
        lo, hi = type_bounds(t)
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


# ---------------------------------------------------------------------------
# Z3 model parsing
# ---------------------------------------------------------------------------

_DEFINE_FUN_RE = re.compile(
    r'\(define-fun\s+(\S+)\s+\(\)\s+(\S+)\s+((?:[^()]+|\([^()]*\))*)\)',
    re.DOTALL,
)


def _parse_z3_val(raw: str) -> str:
    s = raw.strip()
    m = re.match(r'^\(-\s*(\d+)\)?$', s)
    return f"-{m.group(1)}" if m else s


def _parse_z3_model(output: str) -> dict[str, tuple[str, str]] | None:
    lines = output.strip().splitlines()
    if not lines or lines[0].strip() != "sat":
        return None
    model_text = "\n".join(lines[1:])
    result: dict[str, tuple[str, str]] = {}
    for m in _DEFINE_FUN_RE.finditer(model_text):
        result[m.group(1)] = (m.group(2).strip(), _parse_z3_val(m.group(3)))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_witness(
    smt_content: str,
    circuit: Circuit,
    solver: str = "z3",
    timeout: int = 30,
) -> tuple[dict[str, tuple[str, str]] | None, str]:
    """
    Find one satisfying variable assignment for the SMT formula, bounded to
    the circuit's Noir types.

    Returns (model, z3_query) where model is the raw Z3 model dict (or None
    if UNSAT/timeout) and z3_query is the full augmented query that was sent.
    """
    int_var_names = [v.name for v in circuit.inputs if v.variable_type == VariableType.INTEGER]
    int_type_map = assign_int_types(int_var_names, smt_content, circuit) if int_var_names else {}
    const_type_rng = random.Random(int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 5)
    recompute_types(circuit, int_type_map, rng=const_type_rng)

    bounds = build_type_bounds(circuit)
    base = _strip_check_commands(smt_content)
    query = base + "\n" + "\n".join(bounds) + "\n(check-sat)\n(get-model)\n"

    cmd = ["cvc5", "--produce-models", "--lang=smt2", "-"] if solver == "cvc5" else ["z3", "-in"]
    try:
        proc = subprocess.run(cmd, input=query, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, query

    return _parse_z3_model(proc.stdout), query


def _strip_check_commands(smt: str) -> str:
    out = []
    for line in smt.splitlines():
        s = line.strip()
        if s.startswith(("(check-sat", "(get-model", "(get-value", "(exit")):
            continue
        out.append(line)
    return "\n".join(out)
