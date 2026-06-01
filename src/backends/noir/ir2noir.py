import random
import src.ir.nodes as IRNodes

from .nodes import *
from .operators import Operator
from .types import BoolType, FieldType, IntegerType, AliasType, StructType, ArrayType, TupleType, NoirType

# Promotion for unsigned unary negation: u{N} → i{2N} (min i16, max i64).
_UNSIGNED_NEG_PROMOTION = {8: 16, 16: 32, 32: 64, 64: 64}


def _collect_used_variables(expr) -> list:
    """Return a deduplicated list of IRNodes.Variable nodes appearing in expr,
    in order of first appearance."""
    import src.ir.nodes as _IR
    seen: set[str] = set()
    result: list = []

    def _walk(node) -> None:
        if isinstance(node, _IR.Variable):
            if node.name not in seen:
                seen.add(node.name)
                result.append(node)
        elif isinstance(node, _IR.UnaryExpression):
            _walk(node.value)
        elif isinstance(node, _IR.BinaryExpression):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, _IR.TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, _IR.SelectExpression):
            _walk(node.array); _walk(node.index)
        elif isinstance(node, _IR.StoreExpression):
            _walk(node.array); _walk(node.index); _walk(node.value)

    _walk(expr)
    return result


def _collect_array_offsets(circuit: IRNodes.Circuit) -> dict[str, int]:
    """For each root array variable, return the shift needed to make all constant
    depth-0 indices non-negative.  offset = max(0, -min_constant_index_seen)."""
    mins: dict[str, int] = {}

    def _root(node):
        if isinstance(node, IRNodes.Variable) and node.variable_type == IRNodes.VariableType.ARRAY:
            return node.name
        if isinstance(node, IRNodes.StoreExpression):
            return _root(node.array)
        return None

    def _walk(node) -> None:
        if isinstance(node, (IRNodes.SelectExpression, IRNodes.StoreExpression)):
            r = _root(node.array)
            if r is not None and isinstance(node.index, IRNodes.Integer) and node.index.value < 0:
                mins[r] = min(mins.get(r, 0), node.index.value)
            _walk(node.index)
            if isinstance(node, IRNodes.StoreExpression):
                _walk(node.value)
        elif isinstance(node, IRNodes.BinaryExpression):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, IRNodes.UnaryExpression):
            _walk(node.value)
        elif isinstance(node, IRNodes.TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, IRNodes.Assertion):
            _walk(node.value)
        elif isinstance(node, IRNodes.Assume):
            _walk(node.condition)
        elif isinstance(node, IRNodes.Assignment):
            _walk(node.rhs)

    for stmt in circuit.statements:
        _walk(stmt)
    return {name: -v for name, v in mins.items()}


def _collect_array_sizes(circuit: IRNodes.Circuit,
                         offsets: dict[str, int] | None = None) -> dict[str, list[int]]:
    """For each array variable, collect the minimum required size at each nesting depth.
    Returns {var_name: [size_depth_0, size_depth_1, ...]}.
    Defaults to 8 at any depth with no observed constant indices.
    If offsets is provided, constant indices are shifted before computing sizes so that
    negative-index accesses contribute to the correct (larger) size."""
    _offsets = offsets or {}
    sizes: dict[str, list[int]] = {}

    def _update(name: str, depth: int, min_size: int) -> None:
        sizes.setdefault(name, [])
        while len(sizes[name]) <= depth:
            sizes[name].append(8)
        sizes[name][depth] = max(sizes[name][depth], min_size)

    def _walk_select(node) -> tuple[str | None, int]:
        """Walk a select/store chain bottom-up.
        Returns (root_var_name, depth_of_this_node_in_the_chain)."""
        if isinstance(node, IRNodes.SelectExpression):
            root, depth = _walk_select(node.array)
            if root is not None:
                if isinstance(node.index, IRNodes.Integer):
                    shifted = node.index.value + (_offsets.get(root, 0) if depth == 0 else 0)
                    _update(root, depth, max(shifted + 1, 8))
                else:
                    _update(root, depth, 8)
            return root, depth + 1
        if isinstance(node, IRNodes.StoreExpression):
            root, depth = _walk_select(node.array)
            if root is not None:
                if isinstance(node.index, IRNodes.Integer):
                    shifted = node.index.value + (_offsets.get(root, 0) if depth == 0 else 0)
                    _update(root, depth, max(shifted + 1, 8))
                else:
                    _update(root, depth, 8)
            return root, depth
        if isinstance(node, IRNodes.Variable) and node.variable_type == IRNodes.VariableType.ARRAY:
            return node.name, 0
        return None, 0

    def _walk(node) -> None:
        if isinstance(node, IRNodes.SelectExpression):
            _walk_select(node)
            _walk(node.index)
        elif isinstance(node, IRNodes.StoreExpression):
            _walk_select(node)
            _walk(node.index)
            _walk(node.value)
        elif isinstance(node, IRNodes.BinaryExpression):
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, IRNodes.UnaryExpression):
            _walk(node.value)
        elif isinstance(node, IRNodes.TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, IRNodes.Assertion):
            _walk(node.value)
        elif isinstance(node, IRNodes.Assume):
            _walk(node.condition)
        elif isinstance(node, IRNodes.Assignment):
            _walk(node.rhs)

    for stmt in circuit.statements:
        _walk(stmt)

    for var in circuit.inputs:
        if var.variable_type == IRNodes.VariableType.ARRAY:
            sizes.setdefault(var.name, [8])
    return sizes


def _build_noir_array_type(smt_sort, sizes: list[int], depth: int = 0) -> ArrayType:
    """Recursively build a Noir ArrayType from an SMT sort and per-depth sizes."""
    outer_size = sizes[depth] if depth < len(sizes) else 8
    val_sort = smt_sort[2]
    if isinstance(val_sort, list) and val_sort[0] == "Array":
        elem_type: NoirType = _build_noir_array_type(val_sort, sizes, depth + 1)
    elif val_sort == "Bool":
        elem_type = BoolType()
    else:
        elem_type = IntegerType(64, True)
    return ArrayType(elem_type, outer_size)


def _fold_expr_constant(expr) -> int | None:
    """Return the integer value of a constant-only IR expression, or None if it has variables."""
    import src.ir.nodes as _IR
    match expr:
        case _IR.Integer():
            return expr.value
        case _IR.UnaryExpression() if expr.op == _IR.Operator.SUB:
            v = _fold_expr_constant(expr.value)
            return -v if v is not None else None
        case _IR.BinaryExpression():
            lv = _fold_expr_constant(expr.lhs)
            rv = _fold_expr_constant(expr.rhs)
            if lv is None or rv is None:
                return None
            match expr.op:
                case _IR.Operator.ADD: return lv + rv
                case _IR.Operator.SUB: return lv - rv
                case _IR.Operator.MUL: return lv * rv
                case _: return None
        case _:
            return None


def _min_signed_type(val: int) -> IntegerType:
    """All integer constants use i64 to avoid out-of-range literal errors."""
    return IntegerType(64, True)


# Default pool: unsigned, capped at u32.
# - Signed types excluded: mixed-signedness comparisons require casts that wrap
#   (e.g. u64 cast to i64 turns large values negative, making impossible constraints
#   like 78 <= x <= 43 satisfiable via overflow).
# - u64 excluded: array indices are cast to u32 in Noir, so a u64 >= 2^32 wraps to a
#   small index and silently bypasses the ACIR bounds check (which operates on the cast
#   value, not the original).  u32 fits exactly into u32, so index casts are lossless.
_SIGNED_POOL   = [IntegerType(8, True),  IntegerType(16, True),  IntegerType(32, True),  IntegerType(64, True)]
_UNSIGNED_POOL = [IntegerType(8, False), IntegerType(16, False), IntegerType(32, False)]
_MIXED_POOL    = _SIGNED_POOL + _UNSIGNED_POOL

def _make_int_type_pool(signedness: str) -> list[IntegerType]:
    """Return the integer type pool for the given signedness config value."""
    if signedness == "signed":
        return _SIGNED_POOL
    if signedness == "mixed":
        return _MIXED_POOL
    return _UNSIGNED_POOL  # default: "unsigned" (u8/u16/u32 only)

def _type_bounds(t: IntegerType) -> tuple[int, int]:
    if t.signed:
        return -(2 ** (t.bits - 1)), (2 ** (t.bits - 1)) - 1
    # Cap unsigned upper bound at the signed companion max (2^(bits-1)-1) so that
    # any uN value is safely castable to iN without wrapping.
    return 0, (2 ** (t.bits - 1)) - 1


def _pick_const_type(val: int, rng: random.Random, sample: bool = True,
                     pool: list[IntegerType] | None = None) -> IntegerType:
    """Pick a random Noir type that can hold val. Negative values require signed types."""
    if not sample:
        return IntegerType(64, True)
    pool = pool if pool is not None else _SIGNED_POOL
    valid = [t for t in pool
             if (t.signed or val >= 0) and _type_bounds(t)[0] <= val <= _type_bounds(t)[1]]
    return rng.choice(valid) if valid else IntegerType(64, True)


def _collect_var_const_ranges(circuit: IRNodes.Circuit) -> dict[str, tuple[int, int]]:
    """For each integer variable, find the min/max constants it's directly paired with."""
    ranges: dict[str, tuple[int, int]] = {}

    def _update(name: str, val: int) -> None:
        lo, hi = ranges.get(name, (0, 0))
        ranges[name] = (min(lo, val), max(hi, val))

    def _walk(node) -> None:
        if isinstance(node, IRNodes.BinaryExpression):
            if isinstance(node.lhs, IRNodes.Variable) and node.lhs.variable_type == IRNodes.VariableType.INTEGER \
                    and isinstance(node.rhs, IRNodes.Integer):
                _update(node.lhs.name, node.rhs.value)
            if isinstance(node.rhs, IRNodes.Variable) and node.rhs.variable_type == IRNodes.VariableType.INTEGER \
                    and isinstance(node.lhs, IRNodes.Integer):
                _update(node.rhs.name, node.lhs.value)
            _walk(node.lhs); _walk(node.rhs)
        elif isinstance(node, IRNodes.UnaryExpression):
            _walk(node.value)
        elif isinstance(node, IRNodes.TernaryExpression):
            _walk(node.condition); _walk(node.if_expr); _walk(node.else_expr)
        elif isinstance(node, IRNodes.SelectExpression):
            _walk(node.array); _walk(node.index)
        elif isinstance(node, IRNodes.StoreExpression):
            _walk(node.array); _walk(node.index); _walk(node.value)
        elif isinstance(node, IRNodes.Assertion):
            _walk(node.value)
        elif isinstance(node, IRNodes.Assume):
            _walk(node.condition)
        elif isinstance(node, IRNodes.Assignment):
            _walk(node.rhs)

    for stmt in circuit.statements:
        _walk(stmt)
    return ranges


def _pick_var_type(name: str, const_ranges: dict, rng: random.Random, sample: bool = True,
                   pool: list[IntegerType] | None = None) -> IntegerType:
    if not sample:
        return IntegerType(64, True)
    pool = pool if pool is not None else _SIGNED_POOL
    min_c, max_c = const_ranges.get(name, (0, 0))
    valid = [t for t in pool
             if _type_bounds(t)[0] <= min_c and _type_bounds(t)[1] >= max_c]
    return rng.choice(valid) if valid else IntegerType(64, True)


def recompute_types(circuit: IRNodes.Circuit, smt_content: str, sample_int_type: bool = True,
                    integer_signedness: str = "signed") -> None:
    """Assign Noir types to every expression in the circuit and store on expr.noir_type.

    Variable types are randomly assigned (seeded from smt_content) filtered by the
    constants they appear alongside. Integer literal types are also randomly chosen.
    All other expression types are propagated bottom-up.
    If sample_int_type is False, all integer variables and constants are forced to i64.
    integer_signedness controls the pool: "signed", "unsigned", or "mixed".
    QF_ANIA always uses unsigned u32-capped regardless of the config: signed types cause
    cast-wrapping bugs in comparisons, and u64 wraps when cast to u32 for array indexing.
    """
    if "QF_ANIA" in smt_content:
        integer_signedness = "unsigned"
    pool = _make_int_type_pool(integer_signedness)
    import hashlib as _hashlib
    seed = int(_hashlib.md5(smt_content.encode()).hexdigest(), 16)
    var_rng   = random.Random(seed)        # for input variable type assignment
    const_rng = random.Random(seed + 5)   # for integer literal type assignment

    const_ranges  = _collect_var_const_ranges(circuit)
    array_offsets = _collect_array_offsets(circuit)
    array_sizes   = _collect_array_sizes(circuit, array_offsets)
    # Store offsets on the circuit so IR2NoirVisitor can apply them during emission.
    circuit.array_offsets = array_offsets

    # Unify sizes across arrays whose SMT sorts are compatible at any nesting depth.
    #
    # Case 1 — same sort, different outer sizes: two (Array Int Bool) arrays sized
    # [bool;60] and [bool;8] cannot be compared with == in Noir.  Take the max.
    #
    # Case 2 — cross-level: an (Array Int (Array Int Bool)) outer array and a flat
    # (Array Int Bool) inner array can appear in the same store, e.g.:
    #   arr1 : (Array Int Bool)              -> [bool; 60]  (accessed at index 59)
    #   arr2 : (Array Int (Array Int Bool))  -> [[bool; 8]; N]  (inner defaults to 8)
    #   (store arr2 i arr1)  =>  assign [bool;60] into element slot typed [bool;8] -> error
    # Fix: unify the outer size of arr1 with the inner (depth-1) size of arr2.
    #
    # Build a map: sort_key -> max depth-0 size seen across all arrays of that sort.
    sort_max_size: dict[str, int] = {}
    for var in circuit.inputs:
        if var.variable_type == IRNodes.VariableType.ARRAY:
            smt_sort = var.meta_info.get('smt_sort', ["Array", "Int", "Int"])
            if smt_sort[1] == "Bool":
                continue
            key = str(smt_sort)
            cur = array_sizes.get(var.name, [8])[0]
            sort_max_size[key] = max(sort_max_size.get(key, 8), cur)

    # Apply: for each array, unify every depth with the max size for that depth's sort.
    for var in circuit.inputs:
        if var.variable_type == IRNodes.VariableType.ARRAY:
            smt_sort = var.meta_info.get('smt_sort', ["Array", "Int", "Int"])
            if smt_sort[1] == "Bool":
                continue
            depths = list(array_sizes.get(var.name, [8]))
            # depth 0: unify with all same-sort arrays
            key0 = str(smt_sort)
            depths[0] = sort_max_size.get(key0, depths[0])
            # depth 1: if element sort is itself an array, unify with that sort's max
            val_sort = smt_sort[2]
            if isinstance(val_sort, list) and val_sort[0] == "Array" and val_sort[1] != "Bool":
                key1 = str(val_sort)
                if key1 in sort_max_size and len(depths) > 1:
                    depths[1] = sort_max_size[key1]
                elif key1 in sort_max_size:
                    depths.append(sort_max_size[key1])
            array_sizes[var.name] = depths

    # Build and store the full Noir ArrayType for each array variable.
    array_type_map: dict[str, ArrayType] = {}
    for var in circuit.inputs:
        if var.variable_type == IRNodes.VariableType.ARRAY:
            smt_sort = var.meta_info.get('smt_sort', ["Array", "Int", "Int"])
            # Bool-indexed arrays have exactly 2 elements (false→0, true→1).
            depths = [2] if smt_sort[1] == "Bool" else array_sizes.get(var.name, [8])
            var.noir_type = _build_noir_array_type(smt_sort, depths)
            array_type_map[var.name] = var.noir_type

    # Pre-assign one type per variable name using circuit.inputs (one entry per name).
    # This ensures all occurrences of the same variable get the same type.
    var_type_map: dict[str, IntegerType] = {}
    for inp in circuit.inputs:
        if inp.variable_type == IRNodes.VariableType.INTEGER:
            var_type_map[inp.name] = _pick_var_type(inp.name, const_ranges, var_rng, sample_int_type, pool)

    _ELEM_TYPE = IntegerType(64, True)  # fixed element type for (Array Int Int)

    def _walk(expr: IRNodes.Expression) -> NoirType:
        # Post-order: children first.
        if isinstance(expr, IRNodes.UnaryExpression):
            _walk(expr.value)
        elif isinstance(expr, IRNodes.BinaryExpression):
            _walk(expr.lhs); _walk(expr.rhs)
        elif isinstance(expr, IRNodes.TernaryExpression):
            _walk(expr.condition); _walk(expr.if_expr); _walk(expr.else_expr)
        elif isinstance(expr, IRNodes.SelectExpression):
            _walk(expr.array); _walk(expr.index)
        elif isinstance(expr, IRNodes.StoreExpression):
            _walk(expr.array); _walk(expr.index); _walk(expr.value)

        if isinstance(expr, IRNodes.Variable):
            if expr.variable_type == IRNodes.VariableType.INTEGER:
                t: NoirType = var_type_map.get(expr.name, IntegerType(64, True))
            elif expr.variable_type == IRNodes.VariableType.ARRAY:
                # Use the authoritative map; expression-tree copies lack pre-built types.
                t = array_type_map.get(expr.name) or (expr.noir_type if isinstance(expr.noir_type, ArrayType) else ArrayType(_ELEM_TYPE, 8))
            else:
                t = BoolType()
        elif isinstance(expr, IRNodes.Integer):
            t = _pick_const_type(expr.value, const_rng, sample_int_type, pool)
        elif isinstance(expr, IRNodes.Boolean):
            t = BoolType()
        elif isinstance(expr, IRNodes.UnaryExpression):
            if expr.op == IRNodes.Operator.SUB:
                inner = expr.value.noir_type
                if isinstance(inner, IntegerType) and not inner.signed:
                    t = IntegerType(_UNSIGNED_NEG_PROMOTION.get(inner.bits, 64), True)
                else:
                    t = inner
            else:
                t = BoolType()
        elif isinstance(expr, IRNodes.BinaryExpression):
            if (expr.op in IRNodes.Operator.relation_connectives()
                    or expr.op in IRNodes.Operator.logic_connectives()):
                t = BoolType()
            else:
                lt, rt = expr.lhs.noir_type, expr.rhs.noir_type
                if isinstance(lt, IntegerType) and isinstance(rt, IntegerType):
                    t = _common_int_type(lt, rt)
                elif isinstance(lt, IntegerType):
                    t = lt
                elif isinstance(rt, IntegerType):
                    t = rt
                else:
                    t = BoolType()
                # For constant-only expressions, widen only if t cannot hold the folded value.
                if isinstance(t, IntegerType):
                    folded = _fold_expr_constant(expr)
                    if folded is not None and not (_type_bounds(t)[0] <= folded <= _type_bounds(t)[1]):
                        valid = [tp for tp in pool
                                 if _type_bounds(tp)[0] <= folded <= _type_bounds(tp)[1]]
                        if valid:
                            t = _common_int_type(t, min(valid, key=lambda tp: tp.bits))
        elif isinstance(expr, IRNodes.TernaryExpression):
            lt, rt = expr.if_expr.noir_type, expr.else_expr.noir_type
            if isinstance(lt, IntegerType) and isinstance(rt, IntegerType):
                t = _common_int_type(lt, rt)
            elif isinstance(lt, IntegerType):
                t = lt
            elif isinstance(rt, IntegerType):
                t = rt
            else:
                t = BoolType()
        elif isinstance(expr, IRNodes.SelectExpression):
            # Result type is the element type of the array.
            arr_type = expr.array.noir_type
            t = arr_type.element_type if isinstance(arr_type, ArrayType) else _ELEM_TYPE
        elif isinstance(expr, IRNodes.StoreExpression):
            # Result type is the same array type as the input.
            arr_type = expr.array.noir_type
            t = arr_type if isinstance(arr_type, ArrayType) else ArrayType(_ELEM_TYPE, 8)
        else:
            t = BoolType()

        expr.noir_type = t
        return t

    for var in circuit.inputs:
        _walk(var)
    for stmt in circuit.statements:
        if isinstance(stmt, (IRNodes.Assertion, IRNodes.Assume)):
            _walk(stmt.value if isinstance(stmt, IRNodes.Assertion) else stmt.condition)
        elif isinstance(stmt, IRNodes.Assignment):
            _walk(stmt.lhs); _walk(stmt.rhs)


def _common_int_type(t1: IntegerType, t2: IntegerType) -> IntegerType:
    """Promotion: max width, signed if either operand is signed.
    When both have equal width but opposite signedness, promote to the next
    wider signed type to avoid a lossy unsigned→signed cast (e.g. u16 → i16
    would overflow values > 32767; use i32 instead)."""
    bits = max(t1.bits, t2.bits)
    signed = t1.signed or t2.signed
    if signed and (t1.signed != t2.signed) and t1.bits == t2.bits:
        # Same width, mixed signedness: need one extra bit to stay lossless.
        bits = min(bits * 2, 64)
    return IntegerType(bits, signed)


def _sanitize_noir_ident(name: str) -> str:
    """Replace characters invalid in Noir identifiers (e.g. hyphens) with underscores."""
    return name.replace("-", "_")


class NameDispenser:
    def __init__(self):
        self._counter = 0

    def next(self, prefix: str) -> Identifier:
        ident = Identifier(f"{prefix}_{self._counter}")
        self._counter += 1
        return ident


class IR2NoirVisitor:
    def __init__(
        self,
        struct_map: dict[str, tuple[str, str]] | None = None,
        array_map: dict[str, tuple[str, int]] | None = None,
        tuple_map: dict[str, tuple[str, int]] | None = None,
        nesting_map: dict[str, tuple[str, str | int]] | None = None,
        rng: random.Random | None = None,
    ):
        self._name_dispenser = NameDispenser()
        # Cache: emitted-expression-string → Identifier for deduplication of mid_N vars.
        self._mid_cache: dict[str, tuple[Identifier, NoirType]] = {}
        # Populated by visit_circuit: maps original param name → actual Noir param name
        # (may add '_' prefix for unused params).
        self.param_name_map: dict[str, str] = {}
        self._struct_map: dict[str, tuple[str, str]] = struct_map or {}
        self._array_map: dict[str, tuple[str, int]] = array_map or {}
        self._tuple_map: dict[str, tuple[str, int]] = tuple_map or {}
        self._nesting_map: dict[str, tuple[str, str | int]] = nesting_map or {}
        # (op_name="abs", bits, signed) — abs functions needed; populated during visit
        self.safe_fns_needed: set[tuple[str, int, bool]] = set()
        self._rng = rng
        # Maps base IntegerType str → alias name (populated in visit_circuit).
        self._type_alias_map: dict[str, str] = {}
        # Maps root array variable name → index offset (to shift negative indices).
        self._array_offset_map: dict[str, int] = {}
        # Maps array variable name → authoritative ArrayType from circuit.inputs.
        self._array_type_map: dict[str, ArrayType] = {}

    def transform(self, system: IRNodes.Circuit) -> Document:
        return self.visit_circuit(system)

    def visit_operator(self, ir_op: IRNodes.Operator) -> Operator:
        mapping = {
            IRNodes.Operator.ADD: Operator.ADD,
            IRNodes.Operator.SUB: Operator.SUB,
            IRNodes.Operator.MUL: Operator.MUL,
            IRNodes.Operator.DIV: Operator.DIV,
            IRNodes.Operator.LAND: Operator.LAND,
            IRNodes.Operator.LOR: Operator.LOR,
            IRNodes.Operator.LXOR: Operator.LXOR,
            IRNodes.Operator.NOT: Operator.NOT,
            IRNodes.Operator.EQU: Operator.EQU,
            IRNodes.Operator.NEQ: Operator.NEQ,
            IRNodes.Operator.LTH: Operator.LTH,
            IRNodes.Operator.LEQ: Operator.LEQ,
            IRNodes.Operator.GTH: Operator.GTH,
            IRNodes.Operator.GEQ: Operator.GEQ,
        }
        ast_op = mapping.get(ir_op)
        if ast_op is None:
            raise NotImplementedError(f"unimplemented IR operator {ir_op.value}")
        return ast_op

    def visit_expression(self, node: IRNodes.IRNode) -> tuple[Expression, list[Statement]]:
        match node:
            case IRNodes.Variable():
                return self.visit_variable(node)
            case IRNodes.Boolean():
                return self.visit_boolean(node)
            case IRNodes.Integer():
                return self.visit_integer(node)
            case IRNodes.UnaryExpression():
                return self.visit_unary_expression(node)
            case IRNodes.BinaryExpression():
                return self.visit_binary_expression(node)
            case IRNodes.TernaryExpression():
                return self.visit_ternary_expression(node)
            case IRNodes.SelectExpression():
                return self.visit_select_expression(node)
            case IRNodes.StoreExpression():
                return self.visit_store_expression(node)
            case _:
                raise NotImplementedError(f"unsupported expression node {node.__class__.__name__}")

    def visit_statement(self, node: IRNodes.IRNode) -> list[Statement]:
        match node:
            case IRNodes.Assertion():
                return self.visit_assertion(node)
            case IRNodes.Assignment():
                return self.visit_assignment(node)
            case IRNodes.Assume():
                return self.visit_assume(node)
            case _:
                raise NotImplementedError(f"unsupported statement node {node.__class__.__name__}")

    def _apply_nesting(self, leaf_param: str, expr: Expression) -> Expression:
        """Recursively wrap expr with outer accesses for each nesting level."""
        if leaf_param not in self._nesting_map:
            return expr
        container, access_key = self._nesting_map[leaf_param]
        if isinstance(access_key, int):
            outer = TupleFieldAccess(Identifier(container), access_key)
        else:
            outer = FieldAccess(Identifier(container), access_key)
        composed = self._replace_root(expr, outer)
        # If the container is itself nested, recurse.
        return self._apply_nesting(container, composed)

    def _replace_root(self, expr: Expression, new_root: Expression) -> Expression:
        """Replace the Identifier root of an access chain with new_root."""
        if isinstance(expr, Identifier):
            return new_root
        if isinstance(expr, FieldAccess):
            return FieldAccess(self._replace_root(expr.obj, new_root), expr.field)
        if isinstance(expr, ArrayIndexExpression):
            return ArrayIndexExpression(self._replace_root(expr.array, new_root), expr.index)
        if isinstance(expr, TupleFieldAccess):
            return TupleFieldAccess(self._replace_root(expr.obj, new_root), expr.index)
        return expr

    def visit_variable(self, node: IRNodes.Variable) -> tuple[Expression, list[Statement]]:
        if node.name in self._struct_map:
            param, field = self._struct_map[node.name]
            expr = FieldAccess(Identifier(param), field)
            return self._apply_nesting(param, expr), []
        if node.name in self._array_map:
            param, idx = self._array_map[node.name]
            expr = ArrayIndexExpression(Identifier(param), idx)
            return self._apply_nesting(param, expr), []
        if node.name in self._tuple_map:
            param, idx = self._tuple_map[node.name]
            expr = TupleFieldAccess(Identifier(param), idx)
            return self._apply_nesting(param, expr), []
        return Identifier(_sanitize_noir_ident(node.name)), []

    @staticmethod
    def _root_array_name(node) -> str | None:
        """Walk through StoreExpression chains to find the root array Variable name."""
        if isinstance(node, IRNodes.Variable) and node.variable_type == IRNodes.VariableType.ARRAY:
            return node.name
        if isinstance(node, IRNodes.StoreExpression):
            return IR2NoirVisitor._root_array_name(node.array)
        return None

    def _apply_index_offset(self, index_node: IRNodes.Expression, idx_expr: Expression,
                            root_name: str | None) -> Expression:
        """Shift idx_expr by the per-array offset so all indices are non-negative."""
        offset = self._array_offset_map.get(root_name, 0) if root_name else 0
        if not offset:
            return idx_expr
        if isinstance(index_node, IRNodes.Integer):
            return IntegerLiteral(index_node.value + offset)
        # Variable index: cast to i64, add offset, result stays i64 for the later u32 cast.
        return BinaryExpression(
            Operator.ADD,
            CastExpression(idx_expr, IntegerType(64, True)),
            IntegerLiteral(offset),
        )

    def visit_select_expression(self, node: IRNodes.SelectExpression) -> tuple[Expression, list[Statement]]:
        arr_expr, arr_stmts = self.visit_expression(node.array)
        idx_expr, idx_stmts = self.visit_expression(node.index)
        root_name = self._root_array_name(node.array)
        idx_expr = self._apply_index_offset(node.index, idx_expr, root_name)
        stmts = arr_stmts + idx_stmts
        # Bounds check for non-literal indices: assert((idx as u64) < size) so that
        # the ACIR constraint system rejects out-of-range accesses the same way nargo does.
        if not isinstance(node.index, IRNodes.Integer):
            arr_type = (self._array_type_map.get(root_name) if root_name else None) or self._expr_type(node.array)
            if isinstance(arr_type, ArrayType):
                oob_check = BinaryExpression(
                    Operator.LTH,
                    CastExpression(idx_expr, IntegerType(64, False)),
                    CastExpression(IntegerLiteral(arr_type.size), IntegerType(64, False)),
                )
                stmts.append(AssertStatement(oob_check, StringLiteral("oob")))
        # Noir requires u32 for array indices; also cast bool (false→0, true→1).
        if isinstance(self._expr_type(node.index), (IntegerType, BoolType)):
            idx_expr = CastExpression(idx_expr, IntegerType(32, False))
        return ArrayIndexExpression(arr_expr, idx_expr), stmts

    def visit_store_expression(self, node: IRNodes.StoreExpression) -> tuple[Expression, list[Statement]]:
        arr_expr, arr_stmts = self.visit_expression(node.array)
        idx_expr, idx_stmts = self.visit_expression(node.index)
        val_expr, val_stmts = self.visit_expression(node.value)
        root_name = self._root_array_name(node.array)
        idx_expr = self._apply_index_offset(node.index, idx_expr, root_name)
        # Prefer the authoritative array type from circuit.inputs over the
        # _TypeAssignVisitor fallback (which may be [i64; 8] due to the
        # expression-tree Variable identity issue).
        arr_type = (self._array_type_map.get(root_name) if root_name else None) or self._expr_type(node)
        stmts = arr_stmts + idx_stmts + val_stmts
        # Bounds check for non-literal indices: assert((idx as u64) < size).
        if not isinstance(node.index, IRNodes.Integer) and isinstance(arr_type, ArrayType):
            oob_check = BinaryExpression(
                Operator.LTH,
                CastExpression(idx_expr, IntegerType(64, False)),
                CastExpression(IntegerLiteral(arr_type.size), IntegerType(64, False)),
            )
            stmts.append(AssertStatement(oob_check, StringLiteral("oob")))
        # Noir requires u32 for array indices; also cast bool (false→0, true→1).
        if isinstance(self._expr_type(node.index), (IntegerType, BoolType)):
            idx_expr = CastExpression(idx_expr, IntegerType(32, False))
        if isinstance(arr_type, ArrayType):
            elem_t = arr_type.element_type
            val_t = self._expr_type(node.value)
            if isinstance(elem_t, IntegerType) and isinstance(val_t, IntegerType) and elem_t != val_t:
                val_expr = CastExpression(val_expr, elem_t)
        tmp_name = self._name_dispenser.next("store")
        stmts.append(LetStatement(tmp_name.copy(), arr_expr, arr_type, is_mutable=True))
        stmts.append(AssignStatement(ArrayIndexExpression(tmp_name.copy(), idx_expr), val_expr))
        return tmp_name, stmts

    def visit_boolean(self, node: IRNodes.Boolean) -> tuple[Expression, list[Statement]]:
        return BooleanLiteral(node.value), []

    def visit_integer(self, node: IRNodes.Integer) -> tuple[Expression, list[Statement]]:
        t = self._expr_type(node)
        if isinstance(t, IntegerType):
            return CastExpression(IntegerLiteral(node.value), t), []
        return IntegerLiteral(node.value), []

    def _expr_type(self, expr: IRNodes.Expression) -> NoirType:
        """Return the Noir type of an expression (pre-computed by recompute_types)."""
        if expr.noir_type is None:
            raise RuntimeError(
                f"expr.noir_type is None on {expr.__class__.__name__}: {expr}. "
                "recompute_types() must be called before visiting the circuit."
            )
        return expr.noir_type

    def visit_unary_expression(self, node: IRNodes.UnaryExpression) -> tuple[Expression, list[Statement]]:
        value_expr, statements = self.visit_expression(node.value)
        op = self.visit_operator(node.op)
        # Unary negation of an unsigned type: promote to signed before negating.
        if node.op == IRNodes.Operator.SUB:
            t = self._expr_type(node.value)
            if isinstance(t, IntegerType) and not t.signed:
                promo = _UNSIGNED_NEG_PROMOTION.get(t.bits, 64)
                value_expr = CastExpression(value_expr, IntegerType(promo, signed=True))
        return UnaryExpression(op, value_expr), statements

    def visit_binary_expression(self, node: IRNodes.BinaryExpression) -> tuple[Expression, list[Statement]]:
        # Constant-fold comparisons where both sides are pure constants.
        # Avoids type-inference ambiguity for expressions like (-91 > -91).
        if node.op in IRNodes.Operator.relation_connectives():
            lv = _fold_expr_constant(node.lhs)
            rv = _fold_expr_constant(node.rhs)
            if lv is not None and rv is not None:
                result = {
                    IRNodes.Operator.EQU: lv == rv,
                    IRNodes.Operator.NEQ: lv != rv,
                    IRNodes.Operator.LTH: lv < rv,
                    IRNodes.Operator.LEQ: lv <= rv,
                    IRNodes.Operator.GTH: lv > rv,
                    IRNodes.Operator.GEQ: lv >= rv,
                }.get(node.op)
                if result is not None:
                    return BooleanLiteral(result), []
        statements: list[Statement] = []
        lhs, lhs_tail = self.visit_expression(node.lhs)
        statements += lhs_tail
        rhs, rhs_tail = self.visit_expression(node.rhs)
        statements += rhs_tail
        op = self.visit_operator(node.op)

        lt = self._expr_type(node.lhs)
        rt = self._expr_type(node.rhs)
        node_type = self._infer_type(node)

        # Insert casts for type-differing integer operands.
        # Also incorporate node_type so that when constant-folding widened the result
        # to a signed type (e.g., u32 operands but result folds to -28 → i32), we use
        # signed arithmetic instead of unsigned (which would panic on underflow).
        common: IntegerType | None = None
        if isinstance(lt, IntegerType) and isinstance(rt, IntegerType):
            common = _common_int_type(lt, rt)
            if isinstance(node_type, IntegerType):
                common = _common_int_type(common, node_type)
            if lt != common:
                lhs = CastExpression(lhs, common)
            if rt != common:
                rhs = CastExpression(rhs, common)
        elif lt is not None:
            common = lt
        elif rt is not None:
            common = rt

        final_expr: Expression = BinaryExpression(op, lhs, rhs)
        # If operands were promoted to a wider type (e.g., u32→i64 for signed arithmetic),
        # cast the result back to node_type so the let binding type matches.
        if isinstance(common, IntegerType) and isinstance(node_type, IntegerType) and common != node_type:
            final_expr = CastExpression(final_expr, node_type)

        return self._maybe_intermediate(node, final_expr, statements)

    @staticmethod
    def _expr_to_str(expr: Expression) -> str:
        """Emit a Noir AST expression to a string for cache-key comparison."""
        import io
        from .emitter import EmitVisitor
        buf = io.StringIO()
        ev = EmitVisitor()
        ev.buffer = buf
        ev.visit(expr)
        return buf.getvalue()

    def _maybe_intermediate(
        self,
        ir_node: IRNodes.BinaryExpression,
        noir_expr: Expression,
        stmts: list[Statement],
    ) -> tuple[Expression, list[Statement]]:
        """Randomly extract a sub-expression into a named intermediate let binding.
        Identical expressions are deduplicated — the same Identifier is reused."""
        if (self._rng
                and ir_node.node_size() >= 3
                and self._rng.random() < 0.3):
            key = self._expr_to_str(noir_expr)
            if key in self._mid_cache:
                cached_id, cached_type = self._mid_cache[key]
                result: Expression = cached_id.copy()
                current_type = self._infer_type(ir_node)
                if isinstance(cached_type, IntegerType) and isinstance(current_type, IntegerType) and cached_type != current_type:
                    result = CastExpression(result, current_type)
                return result, stmts
            mid_name = self._name_dispenser.next("mid")
            mid_type = self._infer_type(ir_node)
            stmts.append(LetStatement(mid_name.copy(), noir_expr, mid_type))
            self._mid_cache[key] = (mid_name.copy(), mid_type)
            return mid_name, stmts
        return noir_expr, stmts

    def _is_abs_pattern(self, node: IRNodes.TernaryExpression) -> IRNodes.Expression | None:
        """Return the inner expression if node is abs(inner), else None."""
        cond = node.condition
        if not (isinstance(cond, IRNodes.BinaryExpression)
                and cond.op == IRNodes.Operator.GEQ
                and isinstance(cond.rhs, IRNodes.Integer)
                and cond.rhs.value == 0):
            return None
        inner = cond.lhs
        if node.if_expr != inner:
            return None
        else_expr = node.else_expr
        if not (isinstance(else_expr, IRNodes.UnaryExpression)
                and else_expr.op == IRNodes.Operator.SUB
                and else_expr.value == inner):
            return None
        return inner

    def visit_ternary_expression(self, node: IRNodes.TernaryExpression) -> tuple[Expression, list[Statement]]:
        # Detect abs pattern and emit safe_math::safe_abs_T instead of tmp if/else.
        inner = self._is_abs_pattern(node)
        if inner is not None:
            t = self._expr_type(inner)
            if isinstance(t, IntegerType):
                inner_expr, inner_stmts = self.visit_expression(inner)
                if not t.signed:
                    # Unsigned values are always >= 0; abs is a no-op.
                    return inner_expr, inner_stmts
                self.safe_fns_needed.add(("abs", t.bits, t.signed))
                call = FunctionCall(Identifier(f"safe_math::abs_{t}"), [inner_expr])
                return call, inner_stmts

        statements: list[Statement] = []
        condition, cond_tail = self.visit_expression(node.condition)
        statements += cond_tail

        result_name = self._name_dispenser.next("tmp")

        # Determine result type from both branches so the declaration matches
        # the actual assigned type (e.g. if one branch needs promotion to i32).
        lt = self._expr_type(node.if_expr)
        rt = self._expr_type(node.else_expr)
        if isinstance(lt, IntegerType) and isinstance(rt, IntegerType) and lt != rt:
            result_type: NoirType = _common_int_type(lt, rt)
        elif isinstance(lt, IntegerType):
            result_type = lt
        elif isinstance(rt, IntegerType):
            result_type = rt
        else:
            result_type = lt  # both are BoolType

        default_value: Expression = BooleanLiteral(False) if isinstance(result_type, BoolType) else IntegerLiteral(0)
        statements.append(LetStatement(result_name.copy(), default_value, result_type, is_mutable=True))

        # Each branch is a new scope — mid_N defined inside must not leak out.
        self._mid_cache.clear()
        if_expr, if_tail = self.visit_expression(node.if_expr)
        self._mid_cache.clear()
        else_expr, else_tail = self.visit_expression(node.else_expr)
        self._mid_cache.clear()

        # Cast branches to the common type when they differ.
        if isinstance(lt, IntegerType) and isinstance(rt, IntegerType) and lt != rt:
            common = _common_int_type(lt, rt)
            if lt != common:
                if_expr = CastExpression(if_expr, common)
            if rt != common:
                else_expr = CastExpression(else_expr, common)

        true_block_stmts = if_tail + [AssignStatement(result_name.copy(), if_expr)]
        false_block_stmts = else_tail + [AssignStatement(result_name.copy(), else_expr)]

        statements.append(
            IfStatement(
                condition,
                BasicBlock(true_block_stmts),
                BasicBlock(false_block_stmts),
            )
        )
        return result_name, statements

    def visit_assertion(self, node: IRNodes.Assertion) -> list[Statement]:
        self._mid_cache.clear()   # reset per-statement so mid_N never escapes its scope
        expr, stmts = self.visit_expression(node.value)
        stmts.append(AssertStatement(expr, StringLiteral(node.identifier)))
        return stmts

    def visit_assignment(self, node: IRNodes.Assignment) -> list[Statement]:
        rhs, rhs_tail = self.visit_expression(node.rhs)
        lhs, lhs_tail = self.visit_expression(node.lhs)
        if not isinstance(lhs, Identifier):
            raise ValueError("Noir assignment target must be an identifier")
        return lhs_tail + rhs_tail + [AssignStatement(lhs, rhs)]

    def visit_assume(self, node: IRNodes.Assume) -> list[Statement]:
        cond, stmts = self.visit_expression(node.condition)
        stmts.append(AssertStatement(cond, StringLiteral(node.identifier)))
        return stmts

    def visit_circuit(self, node: IRNodes.Circuit) -> Document:
        output_names = {o.name for o in node.outputs}
        statements: list[Statement] = []

        # Pick up the per-array index offsets computed by recompute_types (or recompute
        # here as a fallback for callers that skipped recompute_types).
        self._array_offset_map = getattr(node, 'array_offsets', None) or _collect_array_offsets(node)

        # Build an authoritative name→ArrayType map from circuit.inputs so that
        # expression-tree Variable copies (which lack noir_type) resolve correctly.
        self._array_type_map = {
            v.name: v.noir_type
            for v in node.inputs
            if isinstance(v.noir_type, ArrayType)
        }

        for out in node.outputs:
            out_type = self._type_from_var(out)
            default_value: Expression = BooleanLiteral(False) if isinstance(out_type, BoolType) else IntegerLiteral(0)
            statements.append(LetStatement(Identifier(out.name), default_value, out_type, is_mutable=True))

        # Randomly assign each assertion to: inline / group_0 / group_1.
        # Single-member groups → solo bool helper.
        # Multi-member groups → void helper with multiple assert statements.
        # Number of helper groups: 0–5, sampled once per circuit.
        n_groups = self._rng.randint(0, 5) if self._rng else 0
        GROUP_NAMES = tuple(f"group_{i}" for i in range(n_groups))
        SLOTS = ["inline"] + list(GROUP_NAMES)

        assertion_stmts = [s for s in node.statements if isinstance(s, IRNodes.Assertion)]
        non_assertion_stmts = [s for s in node.statements if not isinstance(s, IRNodes.Assertion)]

        slot_map: dict[str, list[IRNodes.Assertion]] = {s: [] for s in SLOTS}
        for stmt in assertion_stmts:
            slot = self._rng.choice(SLOTS) if self._rng else "inline"
            slot_map[slot].append(stmt)

        for stmt in non_assertion_stmts:
            statements += self.visit_statement(stmt)
        for stmt in slot_map["inline"]:
            statements += self.visit_statement(stmt)

        helper_fns: list[FunctionDefinition] = []
        flat_visitor = IR2NoirVisitor(rng=self._rng)
        # Share the parent's name dispenser and global defs so:
        #  - generated names (K_N, tmp_N) are globally unique across main + helpers
        #  - globals extracted inside helpers end up in the Document
        flat_visitor._name_dispenser = self._name_dispenser
        flat_visitor.safe_fns_needed = self.safe_fns_needed
        flat_visitor._array_offset_map = self._array_offset_map
        flat_visitor._array_type_map = self._array_type_map
        # Each visitor has its own _mid_cache — scopes must not cross visitor boundaries.
        # (no global defs to share — global constant extraction removed)

        group_helpers: dict[str, tuple] = {}

        for group_name in GROUP_NAMES:
            group = slot_map.get(group_name, [])
            if not group:
                continue

            helper_name = f"helper_{group_name}"
            seen: set[str] = set()
            all_vars: list[IRNodes.Variable] = []
            for stmt in group:
                for v in _collect_used_variables(stmt.value):
                    if v.name not in seen:
                        seen.add(v.name)
                        all_vars.append(v)

            helper_params = [VariableDefinition(Identifier(_sanitize_noir_ident(v.name)), self._type_from_var(v)) for v in all_vars]
            call_args = [self.visit_variable(v)[0] for v in all_vars]

            if len(group) == 1:
                # Solo: returns bool.
                stmt = group[0]
                flat_visitor._mid_cache.clear()
                helper_expr, helper_pre = flat_visitor.visit_expression(stmt.value)
                helper_fns.append(FunctionDefinition(
                    name=Identifier(helper_name),
                    arguments=helper_params,
                    body=BasicBlock(helper_pre + [ExpressionStatement(helper_expr, with_semicolon=False)]),
                    return_type=BoolType(),
                    is_public=False, is_public_return=False,
                ))
                group_helpers[group_name] = (helper_fns[-1], call_args, all_vars, True)
            else:
                # Multi: void, multiple assert statements inside.
                body_stmts: list[Statement] = []
                for stmt in group:
                    flat_visitor._mid_cache.clear()
                    helper_expr, helper_pre = flat_visitor.visit_expression(stmt.value)
                    body_stmts += helper_pre
                    body_stmts.append(AssertStatement(helper_expr, StringLiteral(stmt.identifier)))
                helper_fns.append(FunctionDefinition(
                    name=Identifier(helper_name),
                    arguments=helper_params,
                    body=BasicBlock(body_stmts),
                    return_type=None,
                    is_public=False, is_public_return=False,
                ))
                group_helpers[group_name] = (helper_fns[-1], call_args, all_vars, False)

        return_type: NoirType | None = None
        if len(node.outputs) == 0:
            pass
        elif len(node.outputs) == 1:
            return_expr: Expression = Identifier(node.outputs[0].name)
            return_type = self._type_from_var(node.outputs[0])
            statements.append(ExpressionStatement(return_expr, with_semicolon=False))
        else:
            return_expr = TupleLiteral([Identifier(o.name) for o in node.outputs])
            return_type = TupleType([self._type_from_var(o) for o in node.outputs])
            statements.append(ExpressionStatement(return_expr, with_semicolon=False))

        # --- Type aliases: randomly alias integer types used in this circuit ---
        used_int_types: dict[str, IntegerType] = {}
        for inp in node.inputs:
            if inp.variable_type == IRNodes.VariableType.INTEGER and isinstance(inp.noir_type, IntegerType):
                used_int_types[str(inp.noir_type)] = inp.noir_type
        alias_counter = 0
        for type_str, base_type in sorted(used_int_types.items()):
            if self._rng and self._rng.choice([True, False]):
                self._type_alias_map[type_str] = f"T{alias_counter}"
                alias_counter += 1
        type_alias_defs = [
            TypeAliasDefinition(alias, used_int_types[ts])
            for ts, alias in self._type_alias_map.items()
        ]

        # --- Struct definitions (flat fields + nested leaf types as fields) ---
        struct_fields: dict[str, list[tuple[str, IRNodes.Variable]]] = {}
        for inp in node.inputs:
            if inp.name in self._struct_map:
                param, field = self._struct_map[inp.name]
                struct_fields.setdefault(param, []).append((field, inp))

        # Helper: return the Noir type for a leaf group used as a struct field.
        def _leaf_type(leaf: str) -> NoirType:
            if leaf.startswith("bools_"):
                # array leaf
                members = [v for v in node.inputs if v.name in self._array_map
                           and self._array_map[v.name][0] == leaf]
                return ArrayType(BoolType(), len(members))
            if leaf.startswith("g"):
                # struct leaf
                return StructType(leaf[0].upper() + leaf[1:])
            if leaf.startswith("tup_"):
                # tuple leaf
                members = sorted(
                    [(idx, v) for v in node.inputs if v.name in self._tuple_map
                     and self._tuple_map[v.name][0] == leaf
                     for _, idx in [self._tuple_map[v.name]]],
                    key=lambda x: x[0]
                )
                return TupleType([self._type_from_var(v) for _, v in members])
            return FieldType()

        struct_defs: list[StructDefinition] = []
        for param_name, fields in struct_fields.items():
            if param_name in self._nesting_map:
                continue  # this struct IS nested — emit its StructDef but not as top-level param
            struct_name = param_name[0].upper() + param_name[1:]
            field_defs = [VariableDefinition(Identifier(f), self._type_from_var(v)) for f, v in fields]
            # Add any nested leaf groups as additional fields.
            for leaf, (container, access_key) in self._nesting_map.items():
                if container == param_name and isinstance(access_key, str):
                    field_defs.append(VariableDefinition(Identifier(access_key), _leaf_type(leaf)))
            struct_defs.append(StructDefinition(struct_name, field_defs))

        # Emit StructDefinitions for nested struct groups too (they're referenced as field types).
        for leaf, (container, _) in self._nesting_map.items():
            if leaf.startswith("g") and leaf in struct_fields:
                leaf_name = leaf[0].upper() + leaf[1:]
                leaf_field_defs = [VariableDefinition(Identifier(f), self._type_from_var(v))
                                   for f, v in struct_fields[leaf]]
                struct_defs.append(StructDefinition(leaf_name, leaf_field_defs))

        # --- Array groups ---
        array_groups: dict[str, list[tuple[int, IRNodes.Variable]]] = {}
        for inp in node.inputs:
            if inp.name in self._array_map:
                param, idx = self._array_map[inp.name]
                array_groups.setdefault(param, []).append((idx, inp))

        # --- Tuple groups ---
        tuple_groups: dict[str, list[tuple[int, IRNodes.Variable]]] = {}
        for inp in node.inputs:
            if inp.name in self._tuple_map:
                param, idx = self._tuple_map[inp.name]
                tuple_groups.setdefault(param, []).append((idx, inp))

        # --- Main function parameter list ---
        # Leaf groups that are nested inside something else are NOT top-level params.
        nested_leaves: set[str] = set(self._nesting_map.keys())

        # Collect IR variable names actually referenced in statements so that
        # unused parameters can be prefixed with '_' to silence Noir warnings.
        used_ir_vars: set[str] = set()
        for stmt in node.statements:
            if isinstance(stmt, IRNodes.Assertion):
                for v in _collect_used_variables(stmt.value):
                    used_ir_vars.add(v.name)
            elif isinstance(stmt, IRNodes.Assume):
                for v in _collect_used_variables(stmt.condition):
                    used_ir_vars.add(v.name)
            elif isinstance(stmt, IRNodes.Assignment):
                for v in _collect_used_variables(stmt.rhs):
                    used_ir_vars.add(v.name)

        # Map each Noir param name → the IR variable names that feed into it.
        param_to_ir_vars: dict[str, set[str]] = {}
        for inp in node.inputs:
            if inp.name in output_names:
                continue
            if inp.name in self._struct_map:
                p, _ = self._struct_map[inp.name]
                param_to_ir_vars.setdefault(p, set()).add(inp.name)
            elif inp.name in self._array_map:
                p, _ = self._array_map[inp.name]
                param_to_ir_vars.setdefault(p, set()).add(inp.name)
            elif inp.name in self._tuple_map:
                p, _ = self._tuple_map[inp.name]
                param_to_ir_vars.setdefault(p, set()).add(inp.name)
            else:
                param_to_ir_vars.setdefault(inp.name, set()).add(inp.name)

        def _noir_param_name(p: str) -> str:
            used = bool(param_to_ir_vars.get(p, set()) & used_ir_vars)
            sanitized = _sanitize_noir_ident(p)
            name = sanitized if (used or sanitized.startswith("_")) else f"_{sanitized}"
            self.param_name_map[p] = name
            return name

        seen_params: set[str] = set()
        params: list[VariableDefinition] = []
        for inp in node.inputs:
            if inp.name in output_names:
                continue
            if inp.name in self._struct_map:
                param, _ = self._struct_map[inp.name]
                if param in nested_leaves:
                    continue  # handled via its container
                if param not in seen_params:
                    struct_name = param[0].upper() + param[1:]
                    params.append(VariableDefinition(Identifier(_noir_param_name(param)), StructType(struct_name)))
                    seen_params.add(param)
            elif inp.name in self._array_map:
                param, _ = self._array_map[inp.name]
                if param in nested_leaves:
                    continue
                if param not in seen_params:
                    members = sorted(array_groups[param], key=lambda x: x[0])
                    elem_type = self._type_from_var(members[0][1])
                    params.append(VariableDefinition(Identifier(_noir_param_name(param)), ArrayType(elem_type, len(members))))
                    seen_params.add(param)
            elif inp.name in self._tuple_map:
                param, _ = self._tuple_map[inp.name]
                if param in nested_leaves:
                    continue
                if param not in seen_params:
                    members = sorted(tuple_groups[param], key=lambda x: x[0])
                    elem_types = [self._type_from_var(v) for _, v in members]
                    # Prepend any struct types nested into this tuple.
                    nested_struct_types = [
                        _leaf_type(leaf) for leaf, (cont, idx) in self._nesting_map.items()
                        if cont == param and isinstance(idx, int)
                    ]
                    all_types = nested_struct_types + elem_types
                    params.append(VariableDefinition(Identifier(_noir_param_name(param)), TupleType(all_types)))
                    seen_params.add(param)
            else:
                params.append(VariableDefinition(Identifier(_sanitize_noir_ident(_noir_param_name(inp.name))), self._type_from_var(inp)))

        # --- Nested helper call chains ---
        # Each non-empty group is called exactly once — either from main or from one earlier group.
        # A parent can only adopt one child to guarantee each group is called at most once.
        active = [g for g in GROUP_NAMES if slot_map.get(g)]
        call_parent: dict[str, str] = {}
        already_a_parent: set[str] = set()
        for i, g in enumerate(active):
            if i == 0 or self._rng is None:
                call_parent[g] = "main"
            else:
                # Eligible parents: "main" or earlier active groups that haven't taken a child yet.
                eligible = ["main"] + [active[j] for j in range(i) if active[j] not in already_a_parent]
                parent = self._rng.choice(eligible)
                call_parent[g] = parent
                if parent != "main":
                    already_a_parent.add(parent)
        # Empty groups always point to main (harmless, they won't be in group_helpers).
        for g in GROUP_NAMES:
            if g not in call_parent:
                call_parent[g] = "main"

        # For each non-empty group, build helper params/body, then propagate nested calls upward.
        # Helper metadata: {group_name: (helper_fn, call_args_from_main_perspective)}
        # We process in reverse so we can expand parent params.
        # group_helpers already initialised before the first pass.
        flat_visitor = IR2NoirVisitor(rng=self._rng)
        # Share the parent's name dispenser and global defs so:
        #  - generated names (K_N, tmp_N) are globally unique across main + helpers
        #  - globals extracted inside helpers end up in the Document
        flat_visitor._name_dispenser = self._name_dispenser
        flat_visitor.safe_fns_needed = self.safe_fns_needed
        flat_visitor._array_offset_map = self._array_offset_map
        flat_visitor._array_type_map = self._array_type_map
        # Each visitor has its own _mid_cache — scopes must not cross visitor boundaries.
        # (no global defs to share — global constant extraction removed)

        # First pass: build each helper independently.
        for group_name in GROUP_NAMES:
            group = slot_map.get(group_name, [])
            if not group:
                continue
            helper_name = f"helper_{group_name}"
            seen: set[str] = set()
            all_vars: list[IRNodes.Variable] = []
            for stmt in group:
                for v in _collect_used_variables(stmt.value):
                    if v.name not in seen:
                        seen.add(v.name)
                        all_vars.append(v)

            helper_params = [VariableDefinition(Identifier(_sanitize_noir_ident(v.name)), self._type_from_var(v)) for v in all_vars]
            call_args = [self.visit_variable(v)[0] for v in all_vars]

            if len(group) == 1:
                stmt = group[0]
                flat_visitor._mid_cache.clear()
                helper_expr, helper_pre = flat_visitor.visit_expression(stmt.value)
                fn = FunctionDefinition(
                    name=Identifier(helper_name),
                    arguments=helper_params,
                    body=BasicBlock(helper_pre + [ExpressionStatement(helper_expr, with_semicolon=False)]),
                    return_type=BoolType(),
                    is_public=False, is_public_return=False,
                )
                group_helpers[group_name] = (fn, call_args, all_vars, True)  # True = returns bool
            else:
                body_stmts: list[Statement] = []
                for stmt in group:
                    flat_visitor._mid_cache.clear()
                    helper_expr, helper_pre = flat_visitor.visit_expression(stmt.value)
                    body_stmts += helper_pre
                    body_stmts.append(AssertStatement(helper_expr, StringLiteral(stmt.identifier)))
                fn = FunctionDefinition(
                    name=Identifier(helper_name),
                    arguments=helper_params,
                    body=BasicBlock(body_stmts),
                    return_type=None,
                    is_public=False, is_public_return=False,
                )
                group_helpers[group_name] = (fn, call_args, all_vars, False)  # False = void

        # Second pass: wire nested calls — process in REVERSE order so a group's params
        # are fully expanded before its call is embedded in its grandparent.
        for group_name in reversed(GROUP_NAMES):
            if group_name not in group_helpers:
                continue
            parent = call_parent[group_name]
            if parent == "main":
                continue
            # Inject this group's call into the parent's body and expand parent params.
            if parent not in group_helpers:
                call_parent[group_name] = "main"
                continue
            p_fn, p_call_args, p_vars, p_returns_bool = group_helpers[parent]
            c_fn, c_call_args, c_vars, c_returns_bool = group_helpers[group_name]
            # Add child's vars to parent's params (deduplicated).
            parent_var_names = {v.name for v in p_vars}
            extra_vars = [v for v in c_vars if v.name not in parent_var_names]
            new_p_vars = p_vars + extra_vars
            new_p_params = [VariableDefinition(Identifier(_sanitize_noir_ident(v.name)), self._type_from_var(v)) for v in new_p_vars]
            new_p_call_args = p_call_args + [self.visit_variable(v)[0] for v in extra_vars]
            # Build call to child inside parent's body.
            child_inner_args = [Identifier(_sanitize_noir_ident(v.name)) for v in c_vars]
            child_call = FunctionCall(Identifier(c_fn.name.name), child_inner_args)
            child_stmt = (AssertStatement(child_call, StringLiteral(group_name))
                          if c_returns_bool
                          else ExpressionStatement(child_call, with_semicolon=True))
            if p_returns_bool and p_fn.body.statements:
                # Solo helper: insert before the final return expression.
                pre = p_fn.body.statements[:-1]
                ret = p_fn.body.statements[-1]
                new_body = BasicBlock(pre + [child_stmt] + [ret])
            else:
                new_body = BasicBlock(p_fn.body.statements + [child_stmt])
            new_p_fn = FunctionDefinition(
                name=p_fn.name,
                arguments=new_p_params,
                body=new_body,
                return_type=p_fn.return_type,
                is_public=False, is_public_return=False,
            )
            group_helpers[parent] = (new_p_fn, new_p_call_args, new_p_vars, p_returns_bool)

        # Third pass: add calls to main's statements and collect final helper list.
        for group_name in GROUP_NAMES:
            if group_name not in group_helpers:
                continue
            if call_parent[group_name] != "main":
                continue
            fn, _, all_vars_final, returns_bool = group_helpers[group_name]
            # Recompute call args from the final (post-expansion) var list so they always
            # match the function's current parameter count, even if all_vars grew in pass 2.
            call_args = [self.visit_variable(v)[0] for v in all_vars_final]
            call = FunctionCall(Identifier(fn.name.name), call_args)
            if returns_bool:
                statements.append(AssertStatement(call, StringLiteral(group_name)))
            else:
                statements.append(ExpressionStatement(call, with_semicolon=True))

        # Emit helpers in dependency order (callees before callers).
        helper_fns = []
        for group_name in reversed(GROUP_NAMES):
            if group_name in group_helpers:
                helper_fns.append(group_helpers[group_name][0])
        helper_fns.reverse()

        global_defs: list[GlobalDefinition] = []  # global constant extraction removed

        main_fn = FunctionDefinition(
            name=Identifier("main"),
            arguments=params,
            body=BasicBlock(statements),
            return_type=return_type,
            is_public=True,
            is_public_return=True,
        )
        # Add safe_math submodule if any safe functions were needed.
        submodules = ["safe_math"] if self.safe_fns_needed else []

        return Document(
            main_fn,
            type_alias_defs=type_alias_defs,
            global_defs=global_defs,
            struct_definitions=struct_defs,
            helper_functions=helper_fns,
            submodules=submodules,
        )

    def _type_from_var(self, var: IRNodes.Variable) -> NoirType:
        if var.variable_type == IRNodes.VariableType.BOOLEAN:
            return BoolType()
        if var.variable_type == IRNodes.VariableType.INTEGER:
            base = var.noir_type if isinstance(var.noir_type, IntegerType) else IntegerType(64, True)
            alias = self._type_alias_map.get(str(base))
            return AliasType(alias) if alias else base
        if var.variable_type == IRNodes.VariableType.ARRAY:
            if var.name in self._array_type_map:
                return self._array_type_map[var.name]
            if isinstance(var.noir_type, ArrayType):
                return var.noir_type
            return ArrayType(IntegerType(64, True), 8)
        return FieldType()

    def _infer_type(self, expr: IRNodes.Expression) -> NoirType:
        return self._expr_type(expr)
