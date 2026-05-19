import src.ir.nodes as IRNodes

from .nodes import *
from .operators import Operator
from .types import BoolType, FieldType, IntegerType, TupleType, NoirType

# Promotion for unsigned unary negation: u{N} → i{2N} (min i16, max i64).
_UNSIGNED_NEG_PROMOTION = {8: 16, 16: 32, 32: 64, 64: 64}


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
    """Smallest signed IntegerType whose range contains val."""
    a = abs(val)
    if a <= 127:       return IntegerType(8,  True)
    if a <= 32_767:    return IntegerType(16, True)
    if a <= 2**31 - 1: return IntegerType(32, True)
    return IntegerType(64, True)


def _common_int_type(t1: IntegerType, t2: IntegerType) -> IntegerType:
    """Promotion: max width, signed if either operand is signed."""
    return IntegerType(max(t1.bits, t2.bits), t1.signed or t2.signed)


class NameDispenser:
    def __init__(self):
        self._counter = 0

    def next(self, prefix: str) -> Identifier:
        ident = Identifier(f"{prefix}_{self._counter}")
        self._counter += 1
        return ident


class IR2NoirVisitor:
    def __init__(self, int_type_map: dict[str, IntegerType] | None = None):
        self._name_dispenser = NameDispenser()
        self._int_type_map: dict[str, IntegerType] = int_type_map or {}

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

    def visit_variable(self, node: IRNodes.Variable) -> tuple[Expression, list[Statement]]:
        return Identifier(node.name), []

    def visit_boolean(self, node: IRNodes.Boolean) -> tuple[Expression, list[Statement]]:
        return BooleanLiteral(node.value), []

    def visit_integer(self, node: IRNodes.Integer) -> tuple[Expression, list[Statement]]:
        return IntegerLiteral(node.value), []

    def _expr_int_type(self, expr: IRNodes.Expression) -> IntegerType | None:
        """Return the concrete Noir integer type of an IR expression, or None."""
        match expr:
            case IRNodes.Variable() if expr.variable_type == IRNodes.VariableType.INTEGER:
                return self._int_type_map.get(expr.name, IntegerType(64, True))
            case IRNodes.Integer() if expr.value < 0:
                # Negative literals can't be unsigned; anchor them to i64 so the
                # surrounding expression is promoted to a signed type.
                return IntegerType(64, True)
            case IRNodes.UnaryExpression() if expr.op == IRNodes.Operator.SUB:
                t = self._expr_int_type(expr.value)
                if t is None:
                    return None
                if not t.signed:
                    promo = _UNSIGNED_NEG_PROMOTION.get(t.bits, 64)
                    return IntegerType(promo, signed=True)
                return t
            case IRNodes.BinaryExpression() if expr.op in IRNodes.Operator.arithmetic_connectives():
                lt = self._expr_int_type(expr.lhs)
                rt = self._expr_int_type(expr.rhs)
                if lt is None and rt is None:
                    val = _fold_expr_constant(expr)
                    return _min_signed_type(val) if val is not None else None
                if lt is None:
                    return rt
                if rt is None:
                    return lt
                return _common_int_type(lt, rt)
            case IRNodes.TernaryExpression():
                lt = self._expr_int_type(expr.if_expr)
                rt = self._expr_int_type(expr.else_expr)
                if lt is None:
                    return rt
                if rt is None:
                    return lt
                return _common_int_type(lt, rt)
            case _:
                return None

    def visit_unary_expression(self, node: IRNodes.UnaryExpression) -> tuple[Expression, list[Statement]]:
        value_expr, statements = self.visit_expression(node.value)
        op = self.visit_operator(node.op)
        # Unary negation of an unsigned type: promote to signed before negating.
        if node.op == IRNodes.Operator.SUB:
            t = self._expr_int_type(node.value)
            if t is not None and not t.signed:
                promo = _UNSIGNED_NEG_PROMOTION.get(t.bits, 64)
                value_expr = CastExpression(value_expr, IntegerType(promo, signed=True))
        return UnaryExpression(op, value_expr), statements

    def visit_binary_expression(self, node: IRNodes.BinaryExpression) -> tuple[Expression, list[Statement]]:
        # Constant-fold arithmetic expressions that have no variables.
        if node.op in (IRNodes.Operator.ADD, IRNodes.Operator.SUB, IRNodes.Operator.MUL):
            val = _fold_expr_constant(node)
            if val is not None:
                return IntegerLiteral(val), []
        statements: list[Statement] = []
        lhs, lhs_tail = self.visit_expression(node.lhs)
        statements += lhs_tail
        rhs, rhs_tail = self.visit_expression(node.rhs)
        statements += rhs_tail
        op = self.visit_operator(node.op)
        # Insert casts when integer operands have different types.
        lt = self._expr_int_type(node.lhs)
        rt = self._expr_int_type(node.rhs)
        if lt is not None and rt is not None and lt != rt:
            common = _common_int_type(lt, rt)
            if lt != common:
                lhs = CastExpression(lhs, common)
            if rt != common:
                rhs = CastExpression(rhs, common)
        return BinaryExpression(op, lhs, rhs), statements

    def visit_ternary_expression(self, node: IRNodes.TernaryExpression) -> tuple[Expression, list[Statement]]:
        statements: list[Statement] = []
        condition, cond_tail = self.visit_expression(node.condition)
        statements += cond_tail

        result_name = self._name_dispenser.next("tmp")
        result_type = self._infer_type(node.if_expr)
        default_value: Expression = BooleanLiteral(False) if isinstance(result_type, BoolType) else IntegerLiteral(0)
        statements.append(LetStatement(result_name.copy(), default_value, result_type, is_mutable=True))

        if_expr, if_tail = self.visit_expression(node.if_expr)
        else_expr, else_tail = self.visit_expression(node.else_expr)

        # Cast branches to common integer type when they differ.
        lt = self._expr_int_type(node.if_expr)
        rt = self._expr_int_type(node.else_expr)
        if lt is not None and rt is not None and lt != rt:
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

        # Declare output locals as mutable variables.
        for out in node.outputs:
            out_type = self._type_from_var(out)
            default_value: Expression = BooleanLiteral(False) if isinstance(out_type, BoolType) else IntegerLiteral(0)
            statements.append(
                LetStatement(
                    Identifier(out.name),
                    default_value,
                    out_type,
                    is_mutable=True,
                )
            )

        for stmt in node.statements:
            statements += self.visit_statement(stmt)

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

        # Deduplicate: don't re-declare inputs that share a name with an output.
        deduped_parameters = []
        for inp in node.inputs:
            if inp.name in output_names:
                continue
            deduped_parameters.append(VariableDefinition(Identifier(inp.name), self._type_from_var(inp)))

        main_fn = FunctionDefinition(
            name=Identifier("main"),
            arguments=deduped_parameters,
            body=BasicBlock(statements),
            return_type=return_type,
            is_public=True,
            is_public_return=True,
        )
        return Document(main_fn)

    def _type_from_var(self, var: IRNodes.Variable) -> NoirType:
        if var.variable_type == IRNodes.VariableType.BOOLEAN:
            return BoolType()
        if var.variable_type == IRNodes.VariableType.INTEGER:
            return self._int_type_map.get(var.name, IntegerType(64, True))
        return FieldType()

    def _infer_type(self, expr: IRNodes.Expression) -> NoirType:
        if expr.is_boolean_expression():
            return BoolType()
        t = self._expr_int_type(expr)
        if t is not None:
            return t
        return FieldType()
