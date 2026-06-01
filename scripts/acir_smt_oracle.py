#!/usr/bin/env python3
"""
ACIR SMT Oracle — translates ACIR constraints to QF_ANIA SMT with modular
arithmetic over the BN254 prime field, then checks satisfiability with Z3.

Handles: BLACKBOX::RANGE, ASSERT, INIT, READ, WRITE.
Skips: BRILLIG CALL (unconstrained hints).

Usage:
    python scripts/acir_smt_oracle.py <project_dir> [options]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acir_inspector import find_nargo_toml, compile_acir, parse_acir

BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# ---------------------------------------------------------------------------
# Expression parser: ACIR infix polynomial  →  list of (coeff, [witnesses])
# ---------------------------------------------------------------------------

def tokenize_expr(s: str) -> list[str]:
    return re.findall(r"w\d+|\d+|[+\-*]", s)


def parse_poly(s: str) -> list[tuple[int, list[str]]]:
    """
    Parse an ACIR polynomial expression like '2*w9 - 65536*w10 + 32768'
    into a list of (coefficient, [witness, ...]) monomials.
    Constants are represented as (value, []).
    """
    tokens = tokenize_expr(s.strip())
    terms: list[tuple[int, list[str]]] = []
    sign = 1
    coeff: int | None = None
    witnesses: list[str] = []

    def flush():
        nonlocal coeff, witnesses
        if coeff is None and not witnesses:
            return
        c = sign if coeff is None else coeff
        terms.append((c, list(witnesses)))
        coeff = None
        witnesses = []

    for tok in tokens:
        if tok == "+":
            flush()
            sign = 1
        elif tok == "-":
            flush()
            sign = -1
        elif tok == "*":
            pass
        elif re.match(r"w\d+$", tok):
            if coeff is None:
                coeff = sign
            witnesses.append(tok)
        else:
            val = int(tok)
            coeff = (sign * val) if coeff is None else (coeff * val)

    flush()
    return terms


# ---------------------------------------------------------------------------
# SMT s-expression builder
# ---------------------------------------------------------------------------

def term_to_smt(coeff: int, witnesses: list[str]) -> str:
    if coeff == 0:
        return "0"
    factors = list(witnesses)
    if abs(coeff) != 1 or not factors:
        factors = [str(abs(coeff))] + factors
    product = factors[0] if len(factors) == 1 else f"(* {' '.join(factors)})"
    return f"(- {product})" if coeff < 0 else product


def poly_to_smt(terms: list[tuple[int, list[str]]]) -> str:
    parts = [term_to_smt(c, ws) for c, ws in terms if c != 0]
    if not parts:
        return "0"
    if len(parts) == 1:
        return parts[0]
    return f"(+ {' '.join(parts)})"


# ---------------------------------------------------------------------------
# ASSERT line parser
# ---------------------------------------------------------------------------

def parse_assert(line: str) -> tuple[str, str] | None:
    """'ASSERT lhs = rhs [// comment]' → (lhs_str, rhs_str), or None."""
    body = line[len("ASSERT "):].split("//")[0].strip()
    if " = " not in body:
        return None
    lhs, _, rhs = body.partition(" = ")
    return lhs.strip(), rhs.strip()


# ---------------------------------------------------------------------------
# SMT query builder
# ---------------------------------------------------------------------------

def build_smt(funcs) -> str:
    range_bits: dict[str, int] = {}
    assert_pairs: list[tuple[str, str]] = []
    # Memory: block_name -> list of witness names (in order)
    mem_inits: dict[str, list[str]] = {}
    # Reads: (result_witness, block_name, index_witness)
    mem_reads: list[tuple[str, str, str]] = []
    # Writes: (block_name, index_witness, value_witness) -> new block version name
    mem_writes: list[tuple[str, str, str, str]] = []  # (old_block, idx, val, new_block)

    # Track the current "live" version of each block for sequential writes.
    block_version: dict[str, str] = {}

    for func in funcs:
        if func.is_brillig:
            continue
        for kind, raw in func.opcodes:
            if kind == "BLACKBOX":
                m = re.match(r"BLACKBOX::RANGE input: (w\d+), bits: (\d+)", raw)
                if m:
                    range_bits[m.group(1)] = int(m.group(2))
            elif kind == "ASSERT":
                pair = parse_assert(raw)
                if pair:
                    assert_pairs.append(pair)
            elif kind == "MEM_INIT":
                # INIT b0 = [w0, w1, w2, ...]
                m = re.match(r"INIT (b\d+) = \[([^\]]*)\]", raw)
                if m:
                    block = m.group(1)
                    witnesses = [w.strip() for w in m.group(2).split(",") if w.strip()]
                    mem_inits[block] = witnesses
                    block_version[block] = block
            elif kind == "MEM_READ":
                # READ wN = b0[wM]
                m = re.match(r"READ (w\d+) = (b\d+)\[(w\d+)\]", raw)
                if m:
                    result, block, idx = m.group(1), m.group(2), m.group(3)
                    live = block_version.get(block, block)
                    mem_reads.append((result, live, idx))
            elif kind == "MEM_WRITE":
                # WRITE b0[wM] = wN
                m = re.match(r"WRITE (b\d+)\[(w\d+)\] = (w\d+)", raw)
                if m:
                    block, idx, val = m.group(1), m.group(2), m.group(3)
                    old = block_version.get(block, block)
                    new = f"{block}_v{len(mem_writes)}"
                    mem_writes.append((old, idx, val, new))
                    block_version[block] = new

    # Collect all witnesses
    all_witnesses: set[str] = set(range_bits.keys())
    for lhs, rhs in assert_pairs:
        all_witnesses |= set(re.findall(r"w\d+", lhs))
        all_witnesses |= set(re.findall(r"w\d+", rhs))
    for witnesses in mem_inits.values():
        all_witnesses |= set(witnesses)
    for result, block, idx in mem_reads:
        all_witnesses |= {result, idx}
    for old, idx, val, new in mem_writes:
        all_witnesses |= {idx, val}

    # All array block names (including write versions)
    all_blocks: set[str] = set(mem_inits.keys())
    for old, idx, val, new in mem_writes:
        all_blocks.add(new)

    lines = [
        "; ACIR SMT Oracle — QF_ANIA mod BN254",
        f"; prime = {BN254_PRIME}",
        "(set-logic QF_ANIA)",
        "",
    ]

    # Declare witnesses and add bounds
    for w in sorted(all_witnesses, key=lambda x: int(x[1:])):
        upper = (2 ** range_bits[w]) if w in range_bits else BN254_PRIME
        lines.append(f"(declare-fun {w} () Int)")
        lines.append(f"(assert (>= {w} 0))")
        lines.append(f"(assert (< {w} {upper}))")

    # Declare array blocks
    if all_blocks:
        lines.append("")
        lines.append("; Array blocks")
        for b in sorted(all_blocks):
            lines.append(f"(declare-fun {b} () (Array Int Int))")

    # INIT: assert each element equals its witness
    if mem_inits:
        lines.append("")
        lines.append("; INIT constraints")
        for block, witnesses in mem_inits.items():
            for i, w in enumerate(witnesses):
                lines.append(f"(assert (= (select {block} {i}) {w}))")

    # WRITE: each write produces a new array version via store
    if mem_writes:
        lines.append("")
        lines.append("; WRITE constraints")
        for old, idx, val, new in mem_writes:
            lines.append(f"(assert (= {new} (store {old} {idx} {val})))")

    # READ: result = select(block, index)
    if mem_reads:
        lines.append("")
        lines.append("; READ constraints")
        for result, block, idx in mem_reads:
            lines.append(f"(assert (= {result} (select {block} {idx})))")

    lines.append("")
    lines.append("; ASSERT constraints (mod prime)")
    for lhs, rhs in assert_pairs:
        lhs_smt = poly_to_smt(parse_poly(lhs))
        rhs_smt = poly_to_smt(parse_poly(rhs))
        diff = f"(- {lhs_smt} {rhs_smt})"
        lines.append(f"(assert (= (mod {diff} {BN254_PRIME}) 0))")

    lines += ["", "(check-sat)", "(get-model)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Z3 runner
# ---------------------------------------------------------------------------

def run_z3(smt_content: str, timeout: int) -> tuple[str, str]:
    result = subprocess.run(
        ["z3", "-in"],
        input=smt_content,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (result.stdout + result.stderr).strip()
    verdict = output.split("\n")[0].strip() if output else "unknown"
    return verdict, output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Check ACIR circuit satisfiability with Z3 (QF_NIA mod BN254).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("project", help="Nargo project directory or a file inside it")
    ap.add_argument("--save-smt", action="store_true", help="Write query to acir_oracle.smt2 in the project dir")
    ap.add_argument("--timeout", type=int, default=60, help="Z3 timeout in seconds (default: 60)")
    ap.add_argument("--no-model", action="store_true", help="Suppress model output on SAT")
    ap.add_argument("--force", action="store_true", help="Force nargo recompilation")
    args = ap.parse_args()

    start = Path(args.project).resolve()
    if start.is_file():
        start = start.parent
    project_dir = find_nargo_toml(start)
    if project_dir is None:
        print(f"No Nargo.toml found at or above {start}", file=sys.stderr)
        sys.exit(1)

    print(f"Project : {project_dir}")

    raw_acir = compile_acir(project_dir, args.force)
    funcs = parse_acir(raw_acir)

    smt = build_smt(funcs)

    if args.save_smt:
        out_path = project_dir / "acir_oracle.smt2"
        out_path.write_text(smt)
        print(f"SMT query : {out_path}")

    print(f"Running Z3 (timeout: {args.timeout}s) ...")
    try:
        verdict, full_output = run_z3(smt, args.timeout)
    except subprocess.TimeoutExpired:
        print("Result  : TIMEOUT")
        sys.exit(2)

    print(f"Result  : {verdict.upper()}")

    if verdict == "sat" and not args.no_model:
        print("\n--- model ---")
        print(full_output)
    elif verdict == "unsat":
        print("No satisfying witness exists under BN254 field constraints.")


if __name__ == "__main__":
    main()
