try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class Operator(StrEnum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"

    # Noir uses bitwise-style boolean operators.
    LAND = "&"
    LOR = "|"
    LXOR = "^"
    NOT = "!"

    EQU = "=="
    NEQ = "!="
    LTH = "<"
    LEQ = "<="
    GTH = ">"
    GEQ = ">="
