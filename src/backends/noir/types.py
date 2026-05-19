from dataclasses import dataclass


@dataclass
class NoirType:
    def copy(self) -> "NoirType":
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


@dataclass
class BoolType(NoirType):
    def copy(self) -> "BoolType":
        return BoolType()

    def __str__(self) -> str:
        return "bool"


@dataclass
class FieldType(NoirType):
    def copy(self) -> "FieldType":
        return FieldType()

    def __str__(self) -> str:
        return "Field"


@dataclass
class IntegerType(NoirType):
    bits: int = 64
    signed: bool = True

    def copy(self) -> "IntegerType":
        return IntegerType(self.bits, self.signed)

    def __str__(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.bits}"


@dataclass
class TupleType(NoirType):
    elems: list[NoirType]

    def copy(self) -> "TupleType":
        return TupleType([e.copy() for e in self.elems])

    def __str__(self) -> str:
        return "(" + ", ".join(str(e) for e in self.elems) + ")"
