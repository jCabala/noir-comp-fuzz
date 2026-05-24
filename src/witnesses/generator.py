"""
Witness generation: given an SMT formula and its Circuit IR, find a satisfying
variable assignment bounded to the circuit's Noir types.
"""

import re
import subprocess

from src.ir.nodes import Circuit
from src.backends.noir.ir2noir import recompute_types, _type_bounds
from src.backends.noir.types import IntegerType


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
    from src.ir.nodes import (Expression, Integer as IRInteger, BinaryExpression,
                               UnaryExpression, TernaryExpression, Assertion, Assume, Assignment)
    # Track tightest (lo, hi) seen per unique SMT string across all occurrences.
    # The same expression can appear many times with different noir_types (because
    # constants get independently random types), so we take max(lo) and min(hi).
    tightest: dict[str, tuple[int, int]] = {}

    def _walk(node) -> None:
        if isinstance(node, BinaryExpression):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, UnaryExpression):
            _walk(node.value)
        elif isinstance(node, TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)

        if not isinstance(node, Expression):
            return
        if isinstance(node, IRInteger):
            return  # constants are always within their own range by construction
        t = node.noir_type
        if not isinstance(t, IntegerType):
            return
        smt_str = _expr_to_smt(node)
        if smt_str is None:
            return
        lo, hi = _type_bounds(t)
        if smt_str in tightest:
            old_lo, old_hi = tightest[smt_str]
            tightest[smt_str] = (max(old_lo, lo), min(old_hi, hi))
        else:
            tightest[smt_str] = (lo, hi)

    for stmt in circuit.statements:
        if isinstance(stmt, Assertion):
            _walk(stmt.value)
        elif isinstance(stmt, Assume):
            _walk(stmt.condition)
        elif isinstance(stmt, Assignment):
            _walk(stmt.rhs)

    clauses: list[str] = []
    for smt_str, (lo, hi) in tightest.items():
        clauses.append(f"(assert (>= {smt_str} {lo}))")
        clauses.append(f"(assert (<= {smt_str} {hi}))")
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


def _strip_check_commands(smt: str) -> str:
    out = []
    for line in smt.splitlines():
        s = line.strip()
        if s.startswith(("(check-sat", "(get-model", "(get-value", "(exit")):
            continue
        out.append(line)
    return "\n".join(out)


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
    recompute_types(circuit, smt_content)

    bounds = build_type_bounds(circuit)
    base = _strip_check_commands(smt_content)
    # Upgrade to NIA so nonlinear type-bound expressions (e.g. products of constants
    # with variables) are accepted without errors.
    base = re.sub(r'\(set-logic\s+QF_LIA\b', '(set-logic QF_NIA', base)
    query = base + "\n" + "\n".join(bounds) + "\n(check-sat)\n(get-model)\n"

    cmd = ["cvc5", "--produce-models", "--lang=smt2", "-"] if solver == "cvc5" else ["z3", "-in"]
    try:
        proc = subprocess.run(cmd, input=query, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, query

    return _parse_z3_model(proc.stdout), query
