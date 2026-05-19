import io

from .nodes import *


class EmitVisitor:
    def __init__(self):
        self.indent = 0
        self.buffer = io.StringIO()

    def emit(self, node: ASTNode) -> str:
        self.indent = 0
        self.buffer = io.StringIO()
        self.visit(node)
        return self.buffer.getvalue()

    def visit(self, node: ASTNode):
        match node:
            case Identifier():
                self.visit_identifier(node)
            case BinaryExpression():
                self.visit_binary_expression(node)
            case UnaryExpression():
                self.visit_unary_expression(node)
            case BooleanLiteral():
                self.visit_boolean_literal(node)
            case IntegerLiteral():
                self.visit_integer_literal(node)
            case StringLiteral():
                self.visit_string_literal(node)
            case CastExpression():
                self.visit_cast_expression(node)
            case TupleLiteral():
                self.visit_tuple_literal(node)
            case FunctionCall():
                self.visit_function_call(node)
            case ExpressionStatement():
                self.visit_expression_statement(node)
            case LetStatement():
                self.visit_let_statement(node)
            case AssignStatement():
                self.visit_assign_statement(node)
            case AssertStatement():
                self.visit_assert_statement(node)
            case BasicBlock():
                self.visit_basic_block(node)
            case IfStatement():
                self.visit_if_statement(node)
            case VariableDefinition():
                self.visit_variable_definition(node)
            case FunctionDefinition():
                self.visit_function_definition(node)
            case Document():
                self.visit_document(node)
            case _:
                raise NotImplementedError(f"unsupported node {node.__class__.__name__}")

    def visit_identifier(self, node: Identifier):
        self.buffer.write(node.name)

    def visit_binary_expression(self, node: BinaryExpression):
        self.buffer.write("(")
        self.visit(node.lhs)
        self.buffer.write(f" {node.operator.value} ")
        self.visit(node.rhs)
        self.buffer.write(")")

    def visit_unary_expression(self, node: UnaryExpression):
        self.buffer.write(f"({node.operator.value}")
        self.visit(node.value)
        self.buffer.write(")")

    def visit_boolean_literal(self, node: BooleanLiteral):
        self.buffer.write("true" if node.value else "false")

    def visit_integer_literal(self, node: IntegerLiteral):
        self.buffer.write(str(node.value))

    def visit_string_literal(self, node: StringLiteral):
        self.buffer.write(f"\"{node.value}\"")

    def visit_cast_expression(self, node: CastExpression):
        self.buffer.write("(")
        self.visit(node.expr)
        self.buffer.write(f" as {node.type_})")

    def visit_tuple_literal(self, node: TupleLiteral):
        self.buffer.write("(")
        for idx, value in enumerate(node.values):
            self.visit(value)
            if idx + 1 < len(node.values):
                self.buffer.write(", ")
        self.buffer.write(")")

    def visit_function_call(self, node: FunctionCall):
        self.visit(node.function)
        self.buffer.write("(")
        for idx, arg in enumerate(node.arguments):
            self.visit(arg)
            if idx + 1 < len(node.arguments):
                self.buffer.write(", ")
        self.buffer.write(")")

    def visit_expression_statement(self, node: ExpressionStatement):
        self.buffer.write(self.current_indent)
        self.visit(node.expr)
        if node.with_semicolon:
            self.buffer.write(";")

    def visit_let_statement(self, node: LetStatement):
        self.buffer.write(self.current_indent + "let ")
        if node.is_mutable:
            self.buffer.write("mut ")
        self.visit(node.name)
        if node.type_:
            self.buffer.write(f" : {node.type_}")
        self.buffer.write(" = ")
        self.visit(node.expr)
        self.buffer.write(";")

    def visit_assign_statement(self, node: AssignStatement):
        self.buffer.write(self.current_indent)
        self.visit(node.lhs)
        self.buffer.write(" = ")
        self.visit(node.rhs)
        self.buffer.write(";")

    def visit_assert_statement(self, node: AssertStatement):
        self.buffer.write(self.current_indent + "assert(")
        self.visit(node.condition)
        if node.message:
            self.buffer.write(", ")
            self.visit(node.message)
        self.buffer.write(");")

    def visit_basic_block(self, node: BasicBlock):
        self.buffer.write("{\n")
        self.indent += 4
        for stmt in node.statements:
            self.visit(stmt)
            self.buffer.write("\n")
        self.indent -= 4
        self.buffer.write(self.current_indent + "}")

    def visit_if_statement(self, node: IfStatement):
        self.buffer.write(self.current_indent + "if ")
        self.visit(node.condition)
        self.buffer.write(" ")
        self.visit(node.true_block)
        if node.false_block:
            self.buffer.write(" else ")
            self.visit(node.false_block)

    def visit_variable_definition(self, node: VariableDefinition):
        self.visit(node.name)
        self.buffer.write(f" : {node.type_}")

    def visit_function_definition(self, node: FunctionDefinition):
        if node.is_public:
            self.buffer.write("pub ")
        if node.is_unconstrained:
            self.buffer.write("unconstrained ")
        self.buffer.write("fn ")
        self.visit(node.name)
        self.buffer.write("(")
        for idx, arg in enumerate(node.arguments):
            self.visit(arg)
            if idx + 1 < len(node.arguments):
                self.buffer.write(", ")
        self.buffer.write(")")
        if node.return_type is not None:
            if node.is_public_return:
                self.buffer.write(f" -> pub {node.return_type} ")
            else:
                self.buffer.write(f" -> {node.return_type} ")
        else:
            self.buffer.write(" ")
        self.visit(node.body)

    def visit_document(self, node: Document):
        for idx, imp in enumerate(node.imports):
            self.buffer.write(f"use {imp};\n")
            if idx + 1 == len(node.imports):
                self.buffer.write("\n")
        for helper in node.helper_functions:
            self.visit(helper)
            self.buffer.write("\n\n")
        self.visit(node.main)
        self.buffer.write("\n")

    @property
    def current_indent(self) -> str:
        return " " * self.indent
