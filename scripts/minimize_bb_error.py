#!/usr/bin/env python3
"""
Minimize a bb_prove_error case via SMT clause delta-debugging.

Algorithm (1-minimization):
  Each pass tries removing one (assert ...) clause at a time.
  If the full pipeline still triggers a bb prove crash, keep the removal.
  Repeat until no single clause can be removed without losing the bug.

Usage:
    python scripts/minimize_bb_error.py <error_dir> [--output DIR]
                                        [--z3-timeout N] [--nargo-timeout N]
                                        [--bb-timeout N]

<error_dir> should be one of the per-case folders inside errors/bb_prove_error/.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.smt_to_ir.parser import parse_smtlib2
from src.backends.noir.ir2noir import IR2NoirVisitor, recompute_types
from src.backends.noir.emitter import EmitVisitor
from src.witnesses.generator import find_witness
from smt_to_noir_witnesses import (
    build_safe_math_source,
    model_to_prover_toml,
    create_noir_project,
    run_nargo_execute,
)
from smt_to_noir_bb_proofs import run_bb_write_vk, run_bb_prove

_EMITTER = EmitVisitor()


# ---------------------------------------------------------------------------
# SMT helpers
# ---------------------------------------------------------------------------

def _extract_parts(smt_content: str) -> tuple[list[str], list[str]]:
    """Split the formula into declaration lines and (assert ...) lines."""
    declarations, asserts = [], []
    for line in smt_content.splitlines():
        s = line.strip()
        if s.startswith(("(set-logic", "(declare-fun")):
            declarations.append(line)
        elif s.startswith("(assert"):
            asserts.append(line)
    return declarations, asserts


def _rebuild(declarations: list[str], asserts: list[str]) -> str:
    return "\n".join(declarations + asserts + ["(check-sat)"])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _test(smt_content: str, work_dir: Path, z3_timeout: int,
          nargo_timeout: int, bb_timeout: int) -> str:
    """
    Run parse → emit → Z3 → nargo execute → bb prove.
    Returns one of: 'bb_crash', 'sat_ok', 'unsat', 'nargo_fail', 'error'.
    """
    project_dir = work_dir / "project"
    if project_dir.exists():
        shutil.rmtree(project_dir)

    # Parse
    try:
        circuit = parse_smtlib2(smt_content)
    except Exception:
        return "error"

    recompute_types(circuit, smt_content)

    # Emit Noir (flat bundling — simplest for minimization)
    try:
        visitor = IR2NoirVisitor()
        noir_source = _EMITTER.emit(visitor.transform(circuit))
        safe_math_src = (
            build_safe_math_source(visitor.safe_fns_needed)
            if visitor.safe_fns_needed else ""
        )
        param_name_map = visitor.param_name_map
    except Exception:
        return "error"

    # Find witness
    try:
        model, _ = find_witness(smt_content, circuit, solver="z3", timeout=z3_timeout)
    except Exception:
        return "error"
    if model is None:
        return "unsat"

    prover_toml = model_to_prover_toml(model, circuit, {}, {}, {}, {}, param_name_map)
    create_noir_project(project_dir, noir_source, prover_toml, "min_circuit",
                        safe_math_source=safe_math_src)

    # nargo execute
    try:
        nargo_ok, _ = run_nargo_execute(project_dir, timeout=nargo_timeout)
    except subprocess.TimeoutExpired:
        return "error"
    if not nargo_ok:
        return "nargo_fail"

    # bb write_vk + prove
    try:
        vk_ok, _ = run_bb_write_vk(project_dir, "min_circuit", bb_timeout)
        if not vk_ok:
            return "bb_crash"
        prove_ok, _ = run_bb_prove(project_dir, "min_circuit", bb_timeout)  # uses vk/vk internally
        return "sat_ok" if prove_ok else "bb_crash"
    except subprocess.TimeoutExpired:
        return "error"


# ---------------------------------------------------------------------------
# Minimizer
# ---------------------------------------------------------------------------

def minimize(smt_content: str, work_dir: Path,
             z3_timeout: int, nargo_timeout: int, bb_timeout: int) -> str:
    declarations, asserts = _extract_parts(smt_content)
    print(f"Initial formula: {len(asserts)} assert clause(s)")

    # Sanity-check that the original triggers the bug.
    print("Verifying original triggers bb_crash...", end=" ", flush=True)
    result = _test(smt_content, work_dir, z3_timeout, nargo_timeout, bb_timeout)
    print(result)
    if result != "bb_crash":
        sys.exit(f"ERROR: original formula does not trigger bb_crash (got: {result})")

    iteration = 0
    changed = True
    while changed:
        changed = False
        iteration += 1
        print(f"\nPass {iteration}  ({len(asserts)} asserts remaining)")
        for i in range(len(asserts)):
            candidate_asserts = asserts[:i] + asserts[i + 1:]
            candidate_smt = _rebuild(declarations, candidate_asserts)
            result = _test(candidate_smt, work_dir, z3_timeout, nargo_timeout, bb_timeout)
            marker = "KEEP" if result == "bb_crash" else f"skip ({result})"
            print(f"  drop [{i:2d}]: {marker}")
            if result == "bb_crash":
                asserts = candidate_asserts
                changed = True
                break  # restart with the shortened list

    print(f"\nMinimized to {len(asserts)} assert clause(s)  (from {len(_extract_parts(smt_content)[1])} in original)")
    return _rebuild(declarations, asserts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimize a bb_prove_error SMT formula by delta-debugging assert clauses."
    )
    parser.add_argument(
        "error_dir",
        help="Path to a single case folder inside errors/bb_prove_error/.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Directory for minimized output (default: <error_dir>/minimized/).",
    )
    parser.add_argument("--z3-timeout",    type=int, default=30,  metavar="N")
    parser.add_argument("--nargo-timeout", type=int, default=60,  metavar="N")
    parser.add_argument("--bb-timeout",    type=int, default=300, metavar="N")
    args = parser.parse_args()

    error_dir = Path(args.error_dir)
    if not error_dir.is_dir():
        sys.exit(f"Error: {error_dir} is not a directory.")

    # Find the SMT formula (exclude z3_query.smt2)
    smt_files = [f for f in error_dir.glob("*.smt2") if "z3_query" not in f.name]
    if not smt_files:
        sys.exit(f"No .smt2 file found in {error_dir}")
    smt_file = smt_files[0]
    smt_content = smt_file.read_text()
    print(f"Input: {smt_file}\n")

    output_dir = Path(args.output) if args.output else error_dir / "minimized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp dir outside any existing Nargo workspace to avoid nargo
    # walking up and picking up a parent Nargo.toml from the error directory.
    with tempfile.TemporaryDirectory(prefix="bb_min_") as _tmp:
        work_dir = Path(_tmp)

        minimized_smt = minimize(
            smt_content, work_dir,
            args.z3_timeout, args.nargo_timeout, args.bb_timeout,
        )

        # Save minimized formula
        out_smt = output_dir / f"minimized_{smt_file.name}"
        out_smt.write_text(minimized_smt)
        print(f"\nMinimized formula  → {out_smt}")

        # Final run to capture the minimized Noir project
        print("Building final minimized Noir project...", end=" ", flush=True)
        final_result = _test(minimized_smt, work_dir,
                             args.z3_timeout, args.nargo_timeout, args.bb_timeout)
        print(final_result)

        project_dir = work_dir / "project"
        if project_dir.exists():
            final_project = output_dir / "project"
            if final_project.exists():
                shutil.rmtree(final_project)
            shutil.copytree(project_dir, final_project,
                            ignore=shutil.ignore_patterns("target"))
            print(f"Minimized project  → {final_project}")

    print("\nDone.")


if __name__ == "__main__":
    main()
