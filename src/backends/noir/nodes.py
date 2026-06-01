from dataclasses import dataclass
from dataclasses import field

from .operators import Operator
from .types import NoirType, IntegerType, StructType, AliasType


@dataclass
class ASTNode:
    def copy(self) -> "ASTNode":
        raise NotImplementedError()


@dataclass
class Expression(ASTNode):
    def copy(self) -> "Expression":
        raise NotImplementedError()


@dataclass
class Statement(ASTNode):
    def copy(self) -> "Statement":
        raise NotImplementedError()


@dataclass
class Identifier(Expression):
    name: str

    def copy(self) -> "Identifier":
        return Identifier(self.name)


@dataclass
class BinaryExpression(Expression):
    operator: Operator
    lhs: Expression
    rhs: Expression

    def copy(self) -> "BinaryExpression":
        return BinaryExpression(self.operator, self.lhs.copy(), self.rhs.copy())


@dataclass
class UnaryExpression(Expression):
    operator: Operator
    value: Expression

    def copy(self) -> "UnaryExpression":
        return UnaryExpression(self.operator, self.value.copy())


@dataclass
class BooleanLiteral(Expression):
    value: bool

    def copy(self) -> "BooleanLiteral":
        return BooleanLiteral(self.value)


@dataclass
class IntegerLiteral(Expression):
    value: int

    def copy(self) -> "IntegerLiteral":
        return IntegerLiteral(self.value)


@dataclass
class StringLiteral(Expression):
    value: str

    def copy(self) -> "StringLiteral":
        return StringLiteral(self.value)


@dataclass
class CastExpression(Expression):
    expr: Expression
    type_: IntegerType

    def copy(self) -> "CastExpression":
        return CastExpression(self.expr.copy(), self.type_.copy())


@dataclass
class TupleLiteral(Expression):
    values: list[Expression]

    def copy(self) -> "TupleLiteral":
        return TupleLiteral([v.copy() for v in self.values])


@dataclass
class FunctionCall(Expression):
    function: Identifier
    arguments: list[Expression]

    def copy(self) -> "FunctionCall":
        return FunctionCall(self.function.copy(), [arg.copy() for arg in self.arguments])


@dataclass
class ExpressionStatement(Statement):
    expr: Expression
    with_semicolon: bool = True

    def copy(self) -> "ExpressionStatement":
        return ExpressionStatement(self.expr.copy(), self.with_semicolon)


@dataclass
class LetStatement(Statement):
    name: Identifier
    expr: Expression
    type_: NoirType | None = None
    is_mutable: bool = False
    is_comptime: bool = False

    def copy(self) -> "LetStatement":
        return LetStatement(
            self.name.copy(),
            self.expr.copy(),
            self.type_.copy() if self.type_ else None,
            self.is_mutable,
            self.is_comptime,
        )


@dataclass
class AssignStatement(Statement):
    lhs: Identifier
    rhs: Expression

    def copy(self) -> "AssignStatement":
        return AssignStatement(self.lhs.copy(), self.rhs.copy())


@dataclass
class AssertStatement(Statement):
    condition: Expression
    message: StringLiteral | None = None

    def copy(self) -> "AssertStatement":
        return AssertStatement(self.condition.copy(), self.message.copy() if self.message else None)


@dataclass
class BasicBlock(Statement):
    statements: list[Statement]

    def copy(self) -> "BasicBlock":
        return BasicBlock([s.copy() for s in self.statements])


@dataclass
class IfStatement(Statement):
    condition: Expression
    true_block: BasicBlock
    false_block: BasicBlock | None = None

    def copy(self) -> "IfStatement":
        return IfStatement(
            self.condition.copy(),
            self.true_block.copy(),
            self.false_block.copy() if self.false_block else None,
        )


@dataclass
class VariableDefinition(ASTNode):
    name: Identifier
    type_: NoirType

    def copy(self) -> "VariableDefinition":
        return VariableDefinition(self.name.copy(), self.type_.copy())


@dataclass
class FunctionDefinition(ASTNode):
    name: Identifier
    arguments: list[VariableDefinition]
    body: BasicBlock
    return_type: NoirType | None = None
    is_public: bool = True
    is_public_return: bool = True
    is_unconstrained: bool = False

    def copy(self) -> "FunctionDefinition":
        return FunctionDefinition(
            self.name.copy(),
            [a.copy() for a in self.arguments],
            self.body.copy(),
            self.return_type.copy() if self.return_type else None,
            self.is_public,
            self.is_public_return,
            self.is_unconstrained,
        )


@dataclass
class TypeAliasDefinition(ASTNode):
    alias: str
    target: NoirType

    def copy(self) -> "TypeAliasDefinition":
        return TypeAliasDefinition(self.alias, self.target.copy())


@dataclass
class GlobalDefinition(ASTNode):
    name: str
    type_: NoirType
    value: Expression

    def copy(self) -> "GlobalDefinition":
        return GlobalDefinition(self.name, self.type_.copy(), self.value.copy())


@dataclass
class TupleFieldAccess(Expression):
    obj: Expression
    index: int

    def copy(self) -> "TupleFieldAccess":
        return TupleFieldAccess(self.obj.copy(), self.index)


@dataclass
class StructDefinition(ASTNode):
    name: str
    fields: list[VariableDefinition]

    def copy(self) -> "StructDefinition":
        return StructDefinition(self.name, [f.copy() for f in self.fields])


@dataclass
class FieldAccess(Expression):
    obj: Expression
    field: str

    def copy(self) -> "FieldAccess":
        return FieldAccess(self.obj.copy(), self.field)


@dataclass
class ArrayIndexExpression(Expression):
    array: Expression
    index: "int | Expression"

    def copy(self) -> "ArrayIndexExpression":
        idx_copy = self.index.copy() if isinstance(self.index, ASTNode) else self.index
        return ArrayIndexExpression(self.array.copy(), idx_copy)


@dataclass
class Document(ASTNode):
    main: FunctionDefinition
    type_alias_defs: list[TypeAliasDefinition] = field(default_factory=list)
    global_defs: list[GlobalDefinition] = field(default_factory=list)
    struct_definitions: list[StructDefinition] = field(default_factory=list)
    helper_functions: list[FunctionDefinition] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    submodules: list[str] = field(default_factory=list)

    def copy(self) -> "Document":
        return Document(
            self.main.copy(),
            [t.copy() for t in self.type_alias_defs],
            [g.copy() for g in self.global_defs],
            [s.copy() for s in self.struct_definitions],
            [helper.copy() for helper in self.helper_functions],
            [i for i in self.imports],
            [s for s in self.submodules],
        )
