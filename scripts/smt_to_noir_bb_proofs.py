#!/usr/bin/env python3
"""
Extends the witness pipeline with Barretenberg proof generation and verification.

Two thread pools:
  Witness pool: parse → emit Noir → Z3/cvc5 → nargo execute
  bb pool:      bb write_vk → bb prove → bb verify  (SAT cases only)

Bug oracles:
  nargo execute OK → bb prove FAIL   → errors/bb_prove_error/
  bb prove OK      → bb verify FAIL  → errors/bb_verify_error/

Usage (same CLI as smt_to_noir_witnesses.py, plus three new flags):
  python scripts/smt_to_noir_bb_proofs.py [input_dir] [options]
  --witness-workers N   parallel nargo workers (default: 4)
  --bb-workers M        parallel bb workers    (default: 2)
  --bb-timeout SECS     per-step bb timeout    (default: 300)
"""

import argparse
import hashlib
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smt_to_noir_witnesses import (
    build_safe_math_source,
    model_to_prover_toml,
    _sanitize_name,
    create_noir_project,
    save_error,
    _is_assertion_failure,
    run_nargo_execute,
    compute_groupings,
    generate_smt,
    build_boundary_prover_toml,
    find_sat_oracle_input,
)

from src.smt_to_ir.parser import parse_smtlib2
from src.backends.noir.ir2noir import IR2NoirVisitor, recompute_types
from src.backends.noir.emitter import EmitVisitor
from src.ir.nodes import VariableType
from src.witnesses.generator import find_witness, build_type_bounds, _strip_check_commands

BB_PATH = os.path.expanduser("~/.bb/bb")
_STATS_INTERVAL = 100


def _write_recent_stats(path: Path, stats: dict, start: int, end: int) -> None:
    lines = [
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Iterations: {start}-{end}",
        "",
        f"sat         : {stats.get('sat_ok',0)} ok  |  {stats.get('sat_error',0)} error  |  {stats.get('sat_overflow',0)} overflow",
        f"unsat       : {stats.get('unsat_ok',0)} ok  |  {stats.get('unsat_error',0)} bug  |  {stats.get('unsat_pipeline_error',0)} pipeline",
        f"sat_to_unsat: {stats.get('oracle_ok',0)} ok  |  {stats.get('oracle_bug',0)} bug  |  {stats.get('oracle_skip',0)} no-flip  |  {stats.get('oracle_pipeline_error',0)} pipeline",
        f"bb_prove    : {stats.get('bb_prove_ok',0)} ok  |  {stats.get('bb_prove_error',0)} error",
        f"bb_verify   : {stats.get('bb_verify_ok',0)} ok  |  {stats.get('bb_verify_bug',0)} bug",
        f"timeout     : {stats.get('n_timeout',0)}",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# bb helpers
# ---------------------------------------------------------------------------

def run_bb_write_vk(project_dir: Path, pkg_name: str, timeout: int) -> tuple[bool, str]:
    proc = subprocess.run(
        [BB_PATH, "write_vk",
         "-b", str(project_dir / "target" / f"{pkg_name}.json"),
         "-o", str(project_dir / "target" / "vk")],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_bb_prove(project_dir: Path, pkg_name: str, timeout: int) -> tuple[bool, str]:
    proc = subprocess.run(
        [BB_PATH, "prove",
         "-b", str(project_dir / "target" / f"{pkg_name}.json"),
         "-w", str(project_dir / "target" / f"{pkg_name}.gz"),
         "-k", str(project_dir / "target" / "vk" / "vk"),
         "-o", str(project_dir / "target" / "proof")],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_bb_verify(project_dir: Path, timeout: int) -> tuple[bool, str]:
    proc = subprocess.run(
        [BB_PATH, "verify",
         "-p", str(project_dir / "target" / "proof" / "proof"),
         "-i", str(project_dir / "target" / "proof" / "public_inputs"),
         "-k", str(project_dir / "target" / "vk" / "vk")],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def save_bb_error(
    errors_dir: Path,
    category: str,
    folder_name: str,
    noir_source: str,
    prover_toml: str,
    package_name: str,
    smt_content: str,
    smt_filename: str,
    bb_output: str,
    safe_math_source: str = "",
    z3_query: str = "",
) -> None:
    dest = errors_dir / category / folder_name
    tmp = dest.parent / (folder_name + ".__tmp__")
    if tmp.exists():
        shutil.rmtree(tmp)
    create_noir_project(tmp, noir_source, prover_toml, package_name,
                        smt_source=smt_content, smt_filename=smt_filename,
                        safe_math_source=safe_math_source, z3_query=z3_query)
    (tmp / "bb_output.txt").write_text(bb_output)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def bb_pipeline(
    project_dir: Path,
    pkg_name: str,
    errors_dir: Path,
    folder_name: str,
    noir_source: str,
    prover_toml: str,
    smt_content: str,
    smt_filename: str,
    safe_math_src: str,
    z3_query: str,
    bb_timeout: int,
    print_lock: threading.Lock,
) -> dict:
    """Run write_vk → prove → verify on a project that passed nargo execute."""
    stats = {
        "bb_prove_ok": 0, "bb_prove_error": 0,
        "bb_verify_ok": 0, "bb_verify_bug": 0,
        "bb_timeout": 0,
    }
    bb_output = ""
    pkg = _sanitize_name(folder_name)

    # Step 1: write_vk
    try:
        vk_ok, vk_out = run_bb_write_vk(project_dir, pkg_name, bb_timeout)
        bb_output += f"=== write_vk ===\n{vk_out}\n"
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"  [{folder_name}] bb write_vk     ... TIMEOUT")
        stats["bb_timeout"] += 1
        return stats

    if not vk_ok:
        with print_lock:
            print(f"  [{folder_name}] bb write_vk     ... ERROR")
        stats["bb_prove_error"] += 1
        save_bb_error(errors_dir, "bb_prove_error", folder_name,
                      noir_source, prover_toml, pkg,
                      smt_content, smt_filename, bb_output, safe_math_src, z3_query)
        return stats

    # Step 2: prove
    try:
        prove_ok, prove_out = run_bb_prove(project_dir, pkg_name, bb_timeout)
        bb_output += f"=== prove ===\n{prove_out}\n"
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"  [{folder_name}] bb prove        ... TIMEOUT")
        stats["bb_timeout"] += 1
        return stats

    if not prove_ok:
        with print_lock:
            print(f"  [{folder_name}] bb prove        ... ERROR")
        stats["bb_prove_error"] += 1
        save_bb_error(errors_dir, "bb_prove_error", folder_name,
                      noir_source, prover_toml, pkg,
                      smt_content, smt_filename, bb_output, safe_math_src, z3_query)
        return stats

    stats["bb_prove_ok"] += 1

    # Step 3: verify
    try:
        verify_ok, verify_out = run_bb_verify(project_dir, bb_timeout)
        bb_output += f"=== verify ===\n{verify_out}\n"
    except subprocess.TimeoutExpired:
        with print_lock:
            print(f"  [{folder_name}] bb verify       ... TIMEOUT")
        stats["bb_timeout"] += 1
        return stats

    if not verify_ok:
        with print_lock:
            print(f"  [{folder_name}] bb verify       ... BUG")
        stats["bb_verify_bug"] += 1
        save_bb_error(errors_dir, "bb_verify_error", folder_name,
                      noir_source, prover_toml, pkg,
                      smt_content, smt_filename, bb_output, safe_math_src, z3_query)
    else:
        with print_lock:
            print(f"  [{folder_name}] bb verify       ... ok")
        stats["bb_verify_ok"] += 1

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMT-LIB → Noir + Barretenberg proof generation & verification."
    )
    parser.add_argument(
        "input_dir", nargs="?", default=None,
        help="Directory with .smt/.smt2 files. Omit for smtfuzz generate mode.",
    )
    parser.add_argument("--smtfuzz-config", default="smtfuzz_config.json", metavar="FILE")
    parser.add_argument("--generator-config", default="generator_config.json", metavar="FILE")
    parser.add_argument("--output-dir", default="obj")
    parser.add_argument("--max-witnesses", type=int, default=1, metavar="N")
    parser.add_argument("--max-programs", type=int, default=None, metavar="N")
    parser.add_argument("--z3-timeout", type=int, default=30, metavar="SECS")
    parser.add_argument("--nargo-timeout", type=int, default=120, metavar="SECS")
    parser.add_argument("--keep-runs", type=int, default=10, metavar="N")
    parser.add_argument("--max-noir-lines", type=int, default=20000, metavar="N")
    parser.add_argument(
        "--witness-workers", type=int, default=4, metavar="N",
        help="Witness thread-pool size (default: 4).",
    )
    parser.add_argument(
        "--bb-workers", type=int, default=2, metavar="M",
        help="bb thread-pool size (default: 2).",
    )
    parser.add_argument(
        "--bb-timeout", type=int, default=300, metavar="SECS",
        help="Timeout per bb subcommand in seconds (default: 300).",
    )
    args = parser.parse_args()

    # Load generator config
    gen_config: dict = {}
    gen_config_path = Path(args.generator_config)
    if gen_config_path.exists():
        with gen_config_path.open() as f:
            gen_config = json.load(f)
    variables_bundling = gen_config.get("variables_bundling", "simple")
    if variables_bundling not in ("flat", "simple", "recursive"):
        sys.exit(f"Error: variables_bundling must be flat, simple or recursive, got '{variables_bundling}'")
    solver = gen_config.get("solver", "z3")
    if solver not in ("z3", "cvc5", "z3_to_cvc5"):
        sys.exit(f"Error: solver must be z3, cvc5, or z3_to_cvc5, got '{solver}'")
    if args.z3_timeout == 30:
        args.z3_timeout = gen_config.get("solver_timeout", 30)
    if args.nargo_timeout == 120:
        args.nargo_timeout = gen_config.get("nargo_timeout", 120)

    output_dir = Path(args.output_dir)
    generate_mode = args.input_dir is None

    if not generate_mode:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            sys.exit(f"Error: {input_dir} is not a directory.")
        smt_files = sorted(input_dir.glob("*.smt")) + sorted(input_dir.glob("*.smt2"))
        if not smt_files:
            sys.exit(f"No .smt / .smt2 files found in {input_dir}.")
        if args.max_programs is not None:
            smt_files = smt_files[: args.max_programs]

    smtfuzz_config: dict = {}
    n_generate = 0
    if generate_mode:
        config_path = Path(args.smtfuzz_config)
        if not config_path.exists():
            sys.exit(f"Error: smtfuzz config not found at {config_path}.")
        with config_path.open() as f:
            smtfuzz_config = json.load(f)
        n_generate = args.max_programs or 100
        print(f"Generate mode: smtfuzz config={config_path}, target={n_generate} formula(s)")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = secrets.token_hex(4)
    run_dir = output_dir / f"run-{timestamp}-{run_id}"
    sat_dir          = run_dir / "out" / "sat"
    unsat_dir        = run_dir / "out" / "unsat"
    sat_to_unsat_dir = run_dir / "out" / "sat_to_unsat"
    errors_dir       = run_dir / "errors"
    sat_dir.mkdir(parents=True, exist_ok=True)
    unsat_dir.mkdir(parents=True, exist_ok=True)
    sat_to_unsat_dir.mkdir(parents=True, exist_ok=True)
    for cat in ("error", "overflow", "timeout", "smt_solver_errors",
                "z3_error_cvc5_succ", "sat_to_unsat_pipeline", "unsat_pipeline",
                "bb_prove_error", "bb_verify_error"):
        (errors_dir / cat).mkdir(parents=True, exist_ok=True)

    # Enforce --keep-runs: clear out/ from old runs, never touch errors/
    all_runs = sorted(output_dir.glob("run-*"))
    for old_run in all_runs[:max(0, len(all_runs) - args.keep_runs)]:
        out_to_delete = old_run / "out"
        if out_to_delete.exists():
            shutil.rmtree(out_to_delete, ignore_errors=True)
            print(f"[cleanup] cleared out/ from: {old_run.name}")

    print(f"Run ID: {run_id}  →  {run_dir}")
    print(f"Workers: {args.witness_workers} witness / {args.bb_workers} bb")

    recent_stats_path = run_dir / "recent_stats.txt"
    _window_snap: dict[str, int] = {}
    _window_start = 1

    # Thread-safety primitives
    print_lock  = threading.Lock()
    stats_lock  = threading.Lock()
    counters: dict[str, int] = {}

    def locked_print(*a, **kw) -> None:
        with print_lock:
            print(*a, **kw)

    def inc(key: str, n: int = 1) -> None:
        with stats_lock:
            counters[key] = counters.get(key, 0) + n

    def _iter_formulas():
        if not generate_mode:
            for smt_file in smt_files:
                yield smt_file.stem, smt_file.read_text()
        else:
            idx = 0
            while idx < n_generate:
                try:
                    content = generate_smt(smtfuzz_config)
                except Exception as exc:
                    locked_print(f"  [smtfuzz] generation error: {exc}")
                    continue
                yield f"gen-{idx:04d}", content
                idx += 1

    def process_formula(stem: str, smt_content: str, bb_pool: ThreadPoolExecutor) -> None:
        """Full witness pipeline for one formula (runs in a witness worker thread)."""
        smt_filename = f"{stem}.smt2"
        locked_print(f"\n=== {stem} ===")

        # 1. Parse → Circuit IR
        try:
            circuit = parse_smtlib2(smt_content)
        except Exception as exc:
            locked_print(f"  ERROR (parse error): {exc}")
            inc("sat_error")
            dest = errors_dir / "error" / f"obj.{stem}-parse"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(f"PARSE ERROR: {exc}\n")
            return

        int_var_names = [v.name for v in circuit.inputs if v.variable_type == VariableType.INTEGER]
        recompute_types(circuit, smt_content, sample_int_type=gen_config.get("sample_int_type", True))
        _type_bounds_clauses = build_type_bounds(circuit)
        if int_var_names:
            types_summary = ", ".join(
                f"{v.name}:{v.noir_type}"
                for v in circuit.inputs if v.variable_type == VariableType.INTEGER
            )
            locked_print(f"  Types: {types_summary}")

        if variables_bundling == "flat":
            struct_map, array_map, tuple_map, nesting_map = {}, {}, {}, {}
        else:
            struct_map, array_map, tuple_map, nesting_map = compute_groupings(
                circuit, smt_content,
                allow_nesting=(variables_bundling == "recursive"),
            )
        ir_rng = random.Random(int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 3)

        # 2. Circuit IR → Noir source
        try:
            visitor = IR2NoirVisitor(
                struct_map=struct_map, array_map=array_map,
                tuple_map=tuple_map, nesting_map=nesting_map, rng=ir_rng,
            )
            noir_doc = visitor.transform(circuit)
            noir_source = EmitVisitor().emit(noir_doc)
            safe_math_src = build_safe_math_source(visitor.safe_fns_needed) if visitor.safe_fns_needed else ""
            param_name_map = visitor.param_name_map
        except Exception as exc:
            import traceback
            locked_print(f"  ERROR (Noir translation error): {exc}")
            inc("sat_error")
            dest = errors_dir / "error" / f"obj.{stem}-translation"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(
                f"TRANSLATION ERROR: {exc}\n\n{traceback.format_exc()}"
            )
            return

        noir_lines = noir_source.count("\n")
        if args.max_noir_lines > 0 and noir_lines > args.max_noir_lines:
            locked_print(f"  [{stem}] SKIP (main.nr too large: {noir_lines} lines > {args.max_noir_lines})")
            inc("skipped")
            return

        # 3. Find a satisfying witness
        z3_failed_error: Exception | None = None
        active_solver = solver
        models: list[dict] = []
        _z3_sat_query = ""

        def _run_solver(s: str) -> tuple[list[dict], str]:
            model, query = find_witness(smt_content, circuit, solver=s, timeout=args.z3_timeout)
            return ([model] if model is not None else []), query

        try:
            if solver == "z3_to_cvc5":
                try:
                    models, _z3_sat_query = _run_solver("z3")
                    active_solver = "z3"
                except Exception as z3_exc:
                    z3_failed_error = z3_exc
                    locked_print(f"  Z3 failed ({z3_exc}), retrying with cvc5 ...")
                    models, _z3_sat_query = _run_solver("cvc5")
                    active_solver = "cvc5"
            else:
                models, _z3_sat_query = _run_solver(solver)
                active_solver = solver
        except subprocess.TimeoutExpired:
            locked_print(f"  ERROR (solver timed out)")
            inc("n_timeout")
            return
        except FileNotFoundError as exc:
            sys.exit(f"Error: solver not found in PATH: {exc}")
        except Exception as exc:
            label = "z3_then_cvc5" if solver == "z3_to_cvc5" else active_solver
            locked_print(f"  ERROR ({label} error): {exc}")
            inc("sat_error")
            dest = errors_dir / "smt_solver_errors" / f"obj.{stem}-{label}error"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "src").mkdir(exist_ok=True)
            (dest / "src" / "main.nr").write_text(noir_source)
            (dest / "nargo_output.txt").write_text(f"SOLVER ERROR: {exc}\n")
            return

        if z3_failed_error is not None:
            dest = errors_dir / "z3_error_cvc5_succ" / f"obj.{stem}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(
                f"Z3 ERROR: {z3_failed_error}\ncvc5 found {len(models)} model(s)\n"
            )

        if not models:
            # UNSAT oracle: boundary-value samples should all be rejected by nargo.
            seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 1
            rng_unsat = random.Random(seed)
            _z3_unsat_query = (
                _strip_check_commands(smt_content)
                + "\n" + "\n".join(_type_bounds_clauses)
                + "\n(check-sat)\n(get-model)\n"
            )
            for idx in range(args.max_witnesses):
                folder_name = f"obj.{stem}-{idx}"
                pkg_name    = _sanitize_name(folder_name)
                project_dir = unsat_dir / folder_name
                boundary_toml = build_boundary_prover_toml(
                    circuit, rng_unsat, struct_map, array_map, tuple_map, nesting_map, param_name_map,
                )
                create_noir_project(project_dir, noir_source, boundary_toml, pkg_name,
                                    smt_source=smt_content, smt_filename=smt_filename,
                                    safe_math_source=safe_math_src, z3_query=_z3_unsat_query)
                timed_out = False
                try:
                    u_ok, u_out = run_nargo_execute(project_dir, timeout=args.nargo_timeout)
                except subprocess.TimeoutExpired:
                    timed_out, u_ok, u_out = True, False, ""
                except FileNotFoundError:
                    sys.exit("Error: 'nargo' not found in PATH.")

                if u_ok:
                    locked_print(f"  [{folder_name}] unsat boundary  ... BUG")
                    inc("unsat_error")
                    save_error(errors_dir, "error", folder_name + "_unsat_oracle",
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT ORACLE VIOLATION: nargo accepted boundary input\n" + u_out,
                               safe_math_src, _z3_unsat_query)
                elif timed_out:
                    locked_print(f"  [{folder_name}] unsat boundary  ... TIMEOUT")
                    inc("n_timeout")
                elif _is_assertion_failure(u_out):
                    locked_print(f"  [{folder_name}] unsat boundary  ... ok (correctly rejected)")
                    inc("unsat_ok")
                else:
                    locked_print(f"  [{folder_name}] unsat boundary  ... PIPELINE ERROR")
                    inc("unsat_pipeline_error")
                    save_error(errors_dir, "unsat_pipeline", folder_name,
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT PIPELINE ERROR: nargo failed but not via assertion\n" + u_out,
                               safe_math_src, _z3_unsat_query)
            return

        locked_print(f"  {len(models)} satisfying model(s) found")

        # 4. For each model: create project → nargo execute → (submit bb pipeline)
        for idx, model in enumerate(models):
            folder_name = f"obj.{stem}-{idx}"
            pkg_name = _sanitize_name(folder_name)
            project_dir = sat_dir / folder_name

            prover_toml = model_to_prover_toml(
                model, circuit, struct_map, array_map, tuple_map, nesting_map, param_name_map,
            )
            create_noir_project(
                project_dir, noir_source, prover_toml, pkg_name,
                smt_source=smt_content, smt_filename=smt_filename,
                safe_math_source=safe_math_src, z3_query=_z3_sat_query,
            )

            timed_out = False
            try:
                success, output = run_nargo_execute(project_dir, timeout=args.nargo_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                success = False
                output = ""
            except FileNotFoundError:
                sys.exit("Error: 'nargo' not found in PATH.")

            if success:
                witness_path = project_dir / "target" / f"{pkg_name}.gz"
                locked_print(f"  [{folder_name}] nargo execute   ... OK  →  {witness_path}")
                inc("sat_ok")

                # Submit bb pipeline; accumulate its stats via callback as soon as it finishes.
                def _on_bb_done(fut):
                    try:
                        for k, v in fut.result().items():
                            inc(k, v)
                    except Exception as exc:
                        locked_print(f"  [BB WORKER ERROR] {exc}")
                bb_pool.submit(
                    bb_pipeline,
                    project_dir, pkg_name, errors_dir, folder_name,
                    noir_source, prover_toml,
                    smt_content, smt_filename, safe_math_src, _z3_sat_query,
                    args.bb_timeout, print_lock,
                ).add_done_callback(_on_bb_done)

                # SAT oracle: find a minimal UNSAT modification and assert nargo rejects it.
                oracle_result = find_sat_oracle_input(smt_content, circuit, model)
                if oracle_result is not None:
                    oracle_model, oracle_z3_query = oracle_result
                    oracle_toml = model_to_prover_toml(
                        oracle_model, circuit, struct_map, array_map, tuple_map, nesting_map, param_name_map,
                    )
                    oracle_dir = sat_to_unsat_dir / folder_name
                    create_noir_project(oracle_dir, noir_source, oracle_toml, pkg_name,
                                        smt_source=smt_content, smt_filename=smt_filename,
                                        safe_math_source=safe_math_src, z3_query=oracle_z3_query)
                    timed_out_oracle = False
                    try:
                        o_ok, o_out = run_nargo_execute(oracle_dir, timeout=args.nargo_timeout)
                    except subprocess.TimeoutExpired:
                        timed_out_oracle = True
                        o_ok, o_out = False, ""

                    if o_ok:
                        locked_print(f"  [{folder_name}] sat oracle      ... BUG")
                        inc("oracle_bug")
                        save_error(errors_dir, "error", folder_name + "_sat_oracle",
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT ORACLE VIOLATION: nargo accepted a Z3-UNSAT input\n" + o_out,
                                   safe_math_src, oracle_z3_query)
                    elif _is_assertion_failure(o_out):
                        locked_print(f"  [{folder_name}] sat oracle      ... ok (correctly rejected)")
                        inc("oracle_ok")
                    elif timed_out_oracle:
                        locked_print(f"  [{folder_name}] sat oracle      ... TIMEOUT")
                        inc("n_timeout")
                    else:
                        locked_print(f"  [{folder_name}] sat oracle      ... PIPELINE ERROR")
                        inc("oracle_pipeline_error")
                        save_error(errors_dir, "sat_to_unsat_pipeline", folder_name,
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT_TO_UNSAT PIPELINE ERROR: nargo failed but not via assertion\n" + o_out,
                                   safe_math_src, oracle_z3_query)
                else:
                    locked_print(f"  [{folder_name}] sat oracle      ... skipped (no UNSAT mutation found)")
                    inc("oracle_skip")

            elif timed_out:
                locked_print(f"  [{folder_name}] nargo execute   ... TIMEOUT")
                inc("n_timeout")
                save_error(errors_dir, "timeout", folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, "",
                           safe_math_src, _z3_sat_query)
            else:
                is_overflow = "overflow" in output.lower()
                category = "overflow" if is_overflow else "error"
                label = "OVERFLOW" if is_overflow else "ERROR"
                first_error = next(
                    (ln for ln in output.splitlines() if "error" in ln.lower()),
                    output.splitlines()[0] if output.strip() else "(no output)",
                )
                locked_print(f"  [{folder_name}] nargo execute   ... {label}")
                locked_print(f"    {first_error}")
                inc("sat_overflow" if is_overflow else "sat_error")
                save_error(errors_dir, category, folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, output,
                           safe_math_src, _z3_sat_query)

    # --- Two-pool execution ---
    # bb_pool stays open (outer) until all bb tasks finish via add_done_callback.
    # witness_pool (inner) finishes first, guaranteeing all bb tasks are submitted.
    with ThreadPoolExecutor(max_workers=args.bb_workers) as bb_pool:
        with ThreadPoolExecutor(max_workers=args.witness_workers) as witness_pool:
            pending_witness = []
            for stem, smt_content in _iter_formulas():
                pending_witness.append(
                    witness_pool.submit(process_formula, stem, smt_content, bb_pool)
                )
            _n_witness_done = 0
            for f in as_completed(pending_witness):
                try:
                    f.result()
                except Exception as exc:
                    locked_print(f"  [WITNESS WORKER ERROR] {exc}")
                _n_witness_done += 1
                if _n_witness_done % _STATS_INTERVAL == 0:
                    with stats_lock:
                        _snap = dict(counters)
                    _delta = {k: _snap.get(k, 0) - _window_snap.get(k, 0) for k in _snap}
                    _write_recent_stats(recent_stats_path, _delta, _window_start, _n_witness_done)
                    _window_snap = _snap
                    _window_start = _n_witness_done + 1
        # bb_pool.__exit__ waits for all bb tasks; their stats land via add_done_callback.

    # --- Summary ---
    c = counters
    total_bugs = (c.get("sat_error", 0) + c.get("unsat_error", 0)
                  + c.get("oracle_bug", 0) + c.get("bb_verify_bug", 0))
    print(f"\n{'='*60}")
    print(f"sat         : {c.get('sat_ok',0)} ok  |  {c.get('sat_error',0)} error  |  {c.get('sat_overflow',0)} overflow")
    print(f"unsat       : {c.get('unsat_ok',0)} ok  |  {c.get('unsat_error',0)} bug  |  {c.get('unsat_pipeline_error',0)} pipeline")
    print(f"sat_to_unsat: {c.get('oracle_ok',0)} ok  |  {c.get('oracle_bug',0)} bug  |  {c.get('oracle_skip',0)} no-flip  |  {c.get('oracle_pipeline_error',0)} pipeline")
    print(f"bb_prove    : {c.get('bb_prove_ok',0)} ok  |  {c.get('bb_prove_error',0)} error")
    print(f"bb_verify   : {c.get('bb_verify_ok',0)} ok  |  {c.get('bb_verify_bug',0)} bug")
    print(f"timeout     : {c.get('n_timeout',0)}")
    print(f"{'─'*40}")
    print(f"total bugs  : {total_bugs}  (overflows not counted as bugs)")


if __name__ == "__main__":
    main()
