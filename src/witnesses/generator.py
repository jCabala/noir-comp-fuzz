"""
Witness generation: given an SMT formula and its Circuit IR, find a satisfying
variable assignment bounded to the circuit's Noir types.
"""

import re
import subprocess

from src.ir.nodes import Circuit
from src.backends.noir.ir2noir import recompute_types, _type_bounds
from src.backends.noir.types import IntegerType, ArrayType


def _expr_type_bounds(t: IntegerType, is_variable: bool) -> tuple[int, int]:
    """
    Return (lo, hi) bounds for Z3.

    For input variables we apply the signed-companion cap (u8 → 0..127, etc.)
    so that a cast 'var as iN' is guaranteed not to wrap.  For intermediate
    expression results we use the real type range (u8 → 0..255) because the
    expression itself is computed at that type and wrapping in a later cast
    is handled by the parent expression's own bounds.
    """
    if is_variable:
        return _type_bounds(t)          # signed-companion cap for variables
    if t.signed:
        return -(2 ** (t.bits - 1)), (2 ** (t.bits - 1)) - 1
    return 0, (2 ** t.bits) - 1        # full unsigned range for expressions


# ---------------------------------------------------------------------------
# Expression → SMT string
# ---------------------------------------------------------------------------

def _expr_to_smt(expr) -> str | None:
    """Convert an IR expression to its SMT-LIB2 string, or None if not representable."""
    from src.ir.nodes import (Integer, Variable, BinaryExpression, UnaryExpression,
                               TernaryExpression, SelectExpression, StoreExpression,
                               Operator, VariableType)
    if isinstance(expr, Integer):
        return f"(- {abs(expr.value)})" if expr.value < 0 else str(expr.value)
    if isinstance(expr, Variable):
        if expr.variable_type in (VariableType.INTEGER, VariableType.BOOLEAN, VariableType.ARRAY):
            return expr.name
        return None
    if isinstance(expr, SelectExpression):
        arr = _expr_to_smt(expr.array)
        idx = _expr_to_smt(expr.index)
        return f"(select {arr} {idx})" if arr is not None and idx is not None else None
    if isinstance(expr, StoreExpression):
        arr = _expr_to_smt(expr.array)
        idx = _expr_to_smt(expr.index)
        val = _expr_to_smt(expr.value)
        if arr is not None and idx is not None and val is not None:
            return f"(store {arr} {idx} {val})"
        return None
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
    from src.ir.nodes import (Expression, Integer as IRInteger, Variable as IRVariable,
                               BinaryExpression, UnaryExpression, TernaryExpression,
                               SelectExpression, StoreExpression, Assertion, Assume, Assignment)
    # For each unique SMT expression string, collect the tightest (lo, hi) across
    # all occurrences (intersection of ranges).
    #
    # Two subtleties:
    # - Variables get the signed-companion-capped range so that 'var as iN'
    #   casts are always safe.  Expressions get the actual type range (e.g. u8 →
    #   0..255, not 0..127) because arithmetic within an unsigned type is fine up
    #   to the full unsigned max; wrapping via later casts is handled by the
    #   parent expression's own bounds.
    # - Taking the tightest (intersection) means Z3 only finds values that are
    #   valid for *every* occurrence of the same expression, preventing overflow
    #   in any typed context.
    bounds_map: dict[str, tuple[int, int]] = {}

    def _walk(node) -> None:
        if isinstance(node, BinaryExpression):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, UnaryExpression):
            _walk(node.value)
        elif isinstance(node, TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, SelectExpression):
            _walk(node.array); _walk(node.index)
        elif isinstance(node, StoreExpression):
            _walk(node.array); _walk(node.index); _walk(node.value)

        if not isinstance(node, Expression):
            return
        if isinstance(node, IRInteger):
            return  # constants are always within their own range by construction

        # For select/store expressions, bound the index to [0, array_size) so that
        # Z3 only picks indices that are valid for our finite Noir array.
        if isinstance(node, (SelectExpression, StoreExpression)):
            arr_type = getattr(node.array, 'noir_type', None)
            if isinstance(arr_type, ArrayType):
                size = arr_type.size
                idx_smt = _expr_to_smt(node.index)
                if idx_smt is not None:
                    bounds_map.setdefault(idx_smt, (0, size - 1))
                    old_lo, old_hi = bounds_map[idx_smt]
                    bounds_map[idx_smt] = (max(old_lo, 0), min(old_hi, size - 1))
            if isinstance(node, SelectExpression):
                return  # the select result's range is handled by element-type bounds below

        # When a BinaryExpression has mixed-type operands, ir2noir will cast the
        # narrower/differently-signed operand to the common type.  Tighten the
        # operand's bound to the common type's range so Z3 cannot pick a value
        # that changes meaning after the cast (e.g. u64 product >= 2^63 becoming
        # negative when reinterpreted as i64).  This must run for all BinaryExpression
        # nodes including Bool comparisons, before the early return on non-integer types.
        if isinstance(node, BinaryExpression):
            from src.backends.noir.ir2noir import _common_int_type
            lt, rt = node.lhs.noir_type, node.rhs.noir_type
            if isinstance(lt, IntegerType) and isinstance(rt, IntegerType) and lt != rt:
                common = _common_int_type(lt, rt)
                for operand in (node.lhs, node.rhs):
                    if operand.noir_type == common:
                        continue
                    op_smt = _expr_to_smt(operand)
                    if op_smt is None or op_smt not in bounds_map:
                        continue
                    c_lo, c_hi = _type_bounds(common)
                    old_lo, old_hi = bounds_map[op_smt]
                    bounds_map[op_smt] = (max(old_lo, c_lo), min(old_hi, c_hi))

        t = node.noir_type
        if not isinstance(t, IntegerType):
            return
        smt_str = _expr_to_smt(node)
        if smt_str is None:
            return
        is_var = isinstance(node, IRVariable)
        lo, hi = _expr_type_bounds(t, is_var)
        if smt_str in bounds_map:
            old_lo, old_hi = bounds_map[smt_str]
            # Tightest: take intersection across all occurrences.
            bounds_map[smt_str] = (max(old_lo, lo), min(old_hi, hi))
        else:
            bounds_map[smt_str] = (lo, hi)

    for stmt in circuit.statements:
        if isinstance(stmt, Assertion):
            _walk(stmt.value)
        elif isinstance(stmt, Assume):
            _walk(stmt.condition)
        elif isinstance(stmt, Assignment):
            _walk(stmt.rhs)

    clauses: list[str] = []
    for smt_str, (lo, hi) in bounds_map.items():
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


def _parse_z3_int_literal(expr) -> int | None:
    """Parse a Z3 model integer literal from a parsed s-expression node."""
    if isinstance(expr, str):
        try:
            return int(expr)
        except ValueError:
            return None
    if isinstance(expr, list) and len(expr) == 2 and expr[0] == "-":
        try:
            return -int(expr[1])
        except (ValueError, TypeError):
            return None
    return None


def _eval_z3_lambda_body(body, var: str, idx: int) -> int:
    """Evaluate a Z3 lambda body s-expression with the bound variable substituted.

    Returns an int (0 or 1 for booleans, arbitrary int for integer arrays).
    Handles the subset of expressions Z3 emits in array model lambdas.
    """
    if body == var:
        return idx
    if body == "true":
        return 1
    if body == "false":
        return 0
    v = _parse_z3_int_literal(body)
    if v is not None:
        return v
    if not isinstance(body, list) or not body:
        return 0
    op = body[0]
    def ev(e):
        return _eval_z3_lambda_body(e, var, idx)
    if op == "=" and len(body) == 3:
        return 1 if ev(body[1]) == ev(body[2]) else 0
    if op == "distinct" and len(body) >= 3:
        vals = [ev(a) for a in body[1:]]
        return 1 if len(vals) == len(set(vals)) else 0
    if op == "not" and len(body) == 2:
        return 1 if ev(body[1]) == 0 else 0
    if op == "and":
        return 1 if all(ev(a) for a in body[1:]) else 0
    if op == "or":
        return 1 if any(ev(a) for a in body[1:]) else 0
    if op == "=>" and len(body) == 3:
        return 1 if ev(body[1]) == 0 or ev(body[2]) != 0 else 0
    if op == "<" and len(body) == 3:
        return 1 if ev(body[1]) < ev(body[2]) else 0
    if op == "<=" and len(body) == 3:
        return 1 if ev(body[1]) <= ev(body[2]) else 0
    if op == ">" and len(body) == 3:
        return 1 if ev(body[1]) > ev(body[2]) else 0
    if op == ">=" and len(body) == 3:
        return 1 if ev(body[1]) >= ev(body[2]) else 0
    if op == "+" :
        return sum(ev(a) for a in body[1:])
    if op == "-" and len(body) == 2:
        return -ev(body[1])
    if op == "-" and len(body) == 3:
        return ev(body[1]) - ev(body[2])
    if op == "*":
        r = 1
        for a in body[1:]:
            r *= ev(a)
        return r
    if op == "ite" and len(body) == 4:
        return ev(body[2]) if ev(body[1]) else ev(body[3])
    return 0


def _eval_z3_array_expr(expr) -> dict | tuple:
    """Evaluate a Z3 array model expression.
    Returns dict mapping index → value, with -1 as the default key.
    Values are ints for leaf arrays or dicts for nested arrays.
    May return a ('lambda', var, body) tuple for deferred lambda evaluation.
    """
    if expr is None:
        return {-1: 0}

    if isinstance(expr, list) and expr:
        head = expr[0]

        # (store base idx val)
        if head == "store" and len(expr) == 4:
            base = _eval_z3_array_expr(expr[1])
            idx = _parse_z3_int_literal(expr[2])
            val_int = _parse_z3_int_literal(expr[3])
            if val_int is not None:
                val = val_int
            elif isinstance(expr[3], list):
                val = _eval_z3_array_expr(expr[3])
            else:
                val = 0
            if idx is not None and isinstance(base, dict):
                result = dict(base)
                result[idx] = val
                return result
            return base if isinstance(base, dict) else {-1: 0}

        # (lambda ((x!1 Int)) body)
        if head == "lambda" and len(expr) == 3:
            params = expr[1]
            body = expr[2]
            var = params[0][0] if (isinstance(params, list) and params
                                   and isinstance(params[0], list)) else "x!1"
            return ("lambda", var, body)

        # ((as const (Array Int ...)) default_val)
        if len(expr) == 2:
            head_expr, val_expr = expr
            if (isinstance(head_expr, list) and len(head_expr) >= 2
                    and head_expr[0] == "as" and head_expr[1] == "const"):
                default_int = _parse_z3_int_literal(val_expr)
                if default_int is not None:
                    return {-1: default_int}
                if val_expr == "true":
                    return {-1: 1}
                if val_expr == "false":
                    return {-1: 0}
                if isinstance(val_expr, list):
                    return {-1: _eval_z3_array_expr(val_expr)}
                return {-1: 0}

    return {-1: 0}


def _arrays_equal(a, b, depth: int = 0) -> bool:
    """Compare two Z3 array representations (dict or lambda tuple) for equality.
    Checks over all explicitly mentioned indices plus a sample range."""
    if depth > 4:
        return True  # avoid infinite recursion on nested arrays
    # Collect all explicitly mentioned indices from both.
    indices: set[int] = set()
    if isinstance(a, dict):
        indices |= {k for k in a if k != -1}
    if isinstance(b, dict):
        indices |= {k for k in b if k != -1}
    # Also check a small range around 0 to catch lambda patterns.
    indices |= set(range(min(32, max((max(indices) + 2 if indices else 0), 8))))

    def _get(arr, i):
        if isinstance(arr, tuple) and arr[0] == "lambda":
            _, var, body = arr
            return _eval_z3_lambda_body(body, var, i)
        if isinstance(arr, dict):
            v = arr.get(i, arr.get(-1, 0))
            return v
        return 0

    for i in indices:
        va, vb = _get(a, i), _get(b, i)
        if isinstance(va, dict) or isinstance(va, tuple):
            if not _arrays_equal(va, vb, depth + 1):
                return False
        elif va != vb:
            return False
    return True


def _eval_z3_bool_expr(expr, env: dict | None = None) -> bool:
    """Evaluate a Z3 boolean expression, resolving let-bindings via env.
    Returns a Python bool.
    """
    env = env or {}
    if expr == "true":
        return True
    if expr == "false":
        return False
    if isinstance(expr, str):
        # local let-binding reference
        if expr in env:
            return bool(env[expr])
        return False
    if not isinstance(expr, list) or not expr:
        return False
    op = expr[0]

    # (let ((a!1 e1) (a!2 e2) ...) body)
    if op == "let" and len(expr) == 3:
        bindings, body = expr[1], expr[2]
        new_env = dict(env)
        for binding in bindings:
            name, val_expr = binding[0], binding[1]
            # Bindings can be arrays or booleans — store raw evaluated form.
            new_env[name] = _eval_z3_expr_generic(val_expr, new_env)
        return _eval_z3_bool_expr(body, new_env)

    if op == "not" and len(expr) == 2:
        return not _eval_z3_bool_expr(expr[1], env)
    if op == "and":
        return all(_eval_z3_bool_expr(a, env) for a in expr[1:])
    if op == "or":
        return any(_eval_z3_bool_expr(a, env) for a in expr[1:])
    if op == "=>" and len(expr) == 3:
        return (not _eval_z3_bool_expr(expr[1], env)) or _eval_z3_bool_expr(expr[2], env)
    if op == "=" and len(expr) == 3:
        lv = _eval_z3_expr_generic(expr[1], env)
        rv = _eval_z3_expr_generic(expr[2], env)
        if isinstance(lv, (dict, tuple)) or isinstance(rv, (dict, tuple)):
            la = lv if isinstance(lv, (dict, tuple)) else {-1: lv}
            ra = rv if isinstance(rv, (dict, tuple)) else {-1: rv}
            return _arrays_equal(la, ra)
        return lv == rv
    if op == "distinct":
        vals = [_eval_z3_expr_generic(a, env) for a in expr[1:]]
        return len(vals) == len(set(vals))
    if op in ("<", "<=", ">", ">="):
        lv = _eval_z3_expr_generic(expr[1], env)
        rv = _eval_z3_expr_generic(expr[2], env)
        if op == "<":  return lv < rv
        if op == "<=": return lv <= rv
        if op == ">":  return lv > rv
        if op == ">=": return lv >= rv
    return False


def _eval_z3_expr_generic(expr, env: dict | None = None):
    """Evaluate a Z3 expression that may be boolean, integer, or array."""
    env = env or {}
    if expr == "true":
        return 1
    if expr == "false":
        return 0
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        v = _parse_z3_int_literal(expr)
        return v if v is not None else 0
    if not isinstance(expr, list) or not expr:
        return 0
    # Array expression?
    if expr[0] in ("store", "lambda") or (
            len(expr) == 2 and isinstance(expr[0], list)
            and expr[0] and expr[0][0] == "as"):
        result = _eval_z3_array_expr(expr)
        # Resolve env references if needed (store base from env)
        return result
    # Let binding
    if expr[0] == "let" and len(expr) == 3:
        bindings, body = expr[1], expr[2]
        new_env = dict(env)
        for binding in bindings:
            new_env[binding[0]] = _eval_z3_expr_generic(binding[1], new_env)
        return _eval_z3_expr_generic(body, new_env)
    # Boolean ops
    if expr[0] in ("not", "and", "or", "=>", "=", "distinct", "<", "<=", ">", ">="):
        return 1 if _eval_z3_bool_expr(expr, env) else 0
    # Integer arithmetic
    op = expr[0]
    if op == "+" :
        return sum(_eval_z3_expr_generic(a, env) for a in expr[1:])
    if op == "-" and len(expr) == 2:
        return -_eval_z3_expr_generic(expr[1], env)
    if op == "-" and len(expr) == 3:
        return _eval_z3_expr_generic(expr[1], env) - _eval_z3_expr_generic(expr[2], env)
    if op == "*":
        r = 1
        for a in expr[1:]:
            r *= _eval_z3_expr_generic(a, env)
        return r
    if op == "ite" and len(expr) == 4:
        cond = _eval_z3_bool_expr(expr[1], env)
        return _eval_z3_expr_generic(expr[2] if cond else expr[3], env)
    return 0


def _parse_z3_model(output: str) -> dict | None:
    """Parse Z3 model output.

    Returns a dict mapping name → one of:
      ("Int",  "42")               — integer scalar
      ("Bool", "true"/"false")     — boolean scalar
      ("Array", {idx: val, -1: default})  — (Array Int Int) value

    Returns None if output doesn't start with 'sat'.
    """
    lines = output.strip().splitlines()
    if not lines or lines[0].strip() != "sat":
        return None

    model_text = "\n".join(lines[1:]).strip()
    if not model_text:
        return {}

    # Parse with the s-expression parser to handle arrays and nested types.
    try:
        from src.smt_to_ir.parser import _parse_script, _strip_smt_comments
        cleaned = _strip_smt_comments(model_text)
        forms = _parse_script(cleaned)

        # Unwrap the outer list: Z3 wraps the model in (...).
        if len(forms) == 1 and isinstance(forms[0], list):
            outer = forms[0]
            if outer and outer[0] == "model":
                entries = outer[1:]
            elif outer and isinstance(outer[0], list) and outer[0] and outer[0][0] == "define-fun":
                entries = outer
            elif outer and isinstance(outer[0], str) and outer[0] == "define-fun":
                entries = [outer]
            else:
                entries = []
        else:
            entries = forms

        result: dict = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 4 or entry[0] != "define-fun":
                continue
            name = entry[1] if isinstance(entry[1], str) else str(entry[1])
            sort = entry[3]
            val_expr = entry[4] if len(entry) > 4 else None

            if sort == "Int":
                v = _parse_z3_int_literal(val_expr)
                result[name] = ("Int", str(v) if v is not None else "0")
            elif sort == "Bool":
                if val_expr in ("true", "false"):
                    result[name] = ("Bool", val_expr)
                elif isinstance(val_expr, list):
                    # Complex expression (let-binding, array equality, etc.) — evaluate it.
                    result[name] = ("Bool", "true" if _eval_z3_bool_expr(val_expr) else "false")
                else:
                    result[name] = ("Bool", "false")
            elif isinstance(sort, list) and len(sort) == 3 and sort[0] == "Array":
                result[name] = ("Array", _eval_z3_array_expr(val_expr))
        return result
    except Exception:
        pass

    # Fallback: regex-based parsing for simple scalar models.
    result = {}
    for m in _DEFINE_FUN_RE.finditer(model_text):
        type_str = m.group(2).strip()
        if type_str == "Int":
            result[m.group(1)] = ("Int", _parse_z3_val(m.group(3)))
        elif type_str == "Bool":
            result[m.group(1)] = ("Bool", m.group(3).strip().lower())
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
    # Upgrade logic to support both nonlinear arithmetic and arrays.
    # QF_LIA/QF_NIA → QF_NIA (handles nonlinear type-bound expressions).
    # QF_ANIA → QF_ANIA (arrays already supported; leave as-is since Z3 handles it).
    base = re.sub(r'\(set-logic\s+QF_LIA\b', '(set-logic QF_NIA', base)
    query = base + "\n" + "\n".join(bounds) + "\n(check-sat)\n(get-model)\n"

    cmd = ["cvc5", "--produce-models", "--lang=smt2", "-"] if solver == "cvc5" else ["z3", "-in"]
    try:
        proc = subprocess.run(cmd, input=query, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, query

    return _parse_z3_model(proc.stdout), query
