import src.ir.nodes as IRNodes

from .nodes import *
from .operators import Operator
from .types import BoolType, FieldType, TupleType, NoirType


class NameDispenser:
    def __init__(self):
        self._counter = 0

    def next(self, prefix: str) -> Identifier:
        ident = Identifier(f"{prefix}_{self._counter}")
        self._counter += 1
        return ident


class IR2NoirVisitor:
    def __init__(self):
        self._name_dispenser = NameDispenser()

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

    def visit_unary_expression(self, node: IRNodes.UnaryExpression) -> tuple[Expression, list[Statement]]:
        value_expr, statements = self.visit_expression(node.value)
        op = self.visit_operator(node.op)
        return UnaryExpression(op, value_expr), statements

    def visit_binary_expression(self, node: IRNodes.BinaryExpression) -> tuple[Expression, list[Statement]]:
        statements: list[Statement] = []
        lhs, lhs_tail = self.visit_expression(node.lhs)
        statements += lhs_tail
        rhs, rhs_tail = self.visit_expression(node.rhs)
        statements += rhs_tail
        op = self.visit_operator(node.op)
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
        true_block_stmts = if_tail + [AssignStatement(result_name.copy(), if_expr)]

        else_expr, else_tail = self.visit_expression(node.else_expr)
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
        return FieldType()

    def _infer_type(self, expr: IRNodes.Expression) -> NoirType:
        if expr.is_boolean_expression():
            return BoolType()
        return FieldType()
