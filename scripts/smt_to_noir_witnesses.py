#!/usr/bin/env python3
"""
Translate SMT-LIB formulas to Noir circuits, enumerate satisfying models
from Z3, and run `nargo execute` to generate witness files.

Usage:
    python scripts/smt_to_noir_witnesses.py <input_dir> [options]

For each .smt / .smt2 file in <input_dir>:
  1. Parse to Circuit IR using the project's parser.
  2. Emit Noir source via IR2NoirVisitor + EmitVisitor.
  3. Use Z3 subprocess to enumerate up to --max-witnesses satisfying models
     (blocking clause enumeration).
  4. For each model, write a self-contained Noir project and run nargo execute.

Supported logics:
  - Boolean core (declare-fun + bool operators)
  - QF_FF (finite-field; requires a Z3 build with FiniteField support)
"""

import argparse
import hashlib
import json
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root so that `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.smt_to_ir.parser import parse_smtlib2
from src.backends.noir.ir2noir import IR2NoirVisitor, recompute_types, _type_bounds
from src.backends.noir.emitter import EmitVisitor
from src.backends.noir.types import IntegerType, ArrayType, BoolType
from src.ir.nodes import VariableType
from src.witnesses.generator import find_witness, build_type_bounds, _strip_check_commands, _eval_z3_lambda_body




def build_safe_math_source(fns_needed: set[tuple[str, int, bool]]) -> str:
    """Generate safe_math.nr source for safe_add/sub/mul/abs per type."""
    lines: list[str] = []
    for op_name, bits, signed in sorted(fns_needed):
        type_str = f"{'i' if signed else 'u'}{bits}"
        fn_name = f"{op_name}_{type_str}"
        if op_name == "abs":
            if not signed:
                # Unsigned values are always non-negative; abs is the identity.
                lines += [
                    f"pub fn {fn_name}(a: {type_str}) -> {type_str} {{",
                    f"    a",
                    f"}}",
                    "",
                ]
            else:
                lines += [
                    f"pub fn {fn_name}(a: {type_str}) -> {type_str} {{",
                    f"    if a >= 0 {{ a }} else {{ -a }}",
                    f"}}",
                    "",
                ]
        else:
            noir_op = {"add": "+", "sub": "-", "mul": "*"}[op_name]
            lo = -(2 ** (bits - 1)) if signed else 0
            hi = (2 ** (bits - 1) - 1) if signed else (2 ** min(bits, 63)) - 1
            lines += [
                f"pub fn {fn_name}(a: {type_str}, b: {type_str}) -> {type_str} {{",
                f"    let c: {type_str} = a {noir_op} b;",
                f"    assert(c >= {lo}, \"{fn_name}_lb\");",
                f"    assert(c <= {hi}, \"{fn_name}_ub\");",
                f"    c",
                f"}}",
                "",
            ]
    return "\n".join(lines)




# ---------------------------------------------------------------------------
# Prover.toml generation
# ---------------------------------------------------------------------------

def _var_toml_value(var, model: dict | None, boundary_val: str | None = None) -> str:
    """Return the TOML value string for a single scalar variable (non-array)."""
    name = var.name
    if var.variable_type == VariableType.BOOLEAN:
        if model and name in model:
            return model[name][1]
        return boundary_val or "false"
    else:
        if model and name in model:
            entry = model[name]
            raw = entry[1] if isinstance(entry[1], str) else "0"
            if raw.startswith("ff"):
                raw = raw[2:]
            return f'"{raw}"'
        return f'"{boundary_val}"' if boundary_val else '"0"'


def _materialize_array_toml(array_val, noir_type: ArrayType) -> str:
    """Recursively materialize a Z3 array model value as a Prover.toml string."""
    size = noir_type.size
    elem_type = noir_type.element_type

    # Lambda: Z3 returned (lambda ((x!1 Int)) body) — evaluate body for each index.
    if isinstance(array_val, tuple) and len(array_val) == 3 and array_val[0] == "lambda":
        _, var, body = array_val
        vals = []
        for i in range(size):
            v = _eval_z3_lambda_body(body, var, i)
            if isinstance(elem_type, ArrayType):
                vals.append(_materialize_array_toml({-1: v}, elem_type))
            elif isinstance(elem_type, BoolType):
                vals.append("true" if v else "false")
            else:
                vals.append(f'"{v}"')
        return "[" + ", ".join(vals) + "]"

    default = array_val.get(-1, 0) if isinstance(array_val, dict) else 0
    vals = []
    for i in range(size):
        v = array_val.get(i, default) if isinstance(array_val, dict) else default
        if isinstance(elem_type, ArrayType):
            inner_dict = v if isinstance(v, dict) else ({-1: v} if isinstance(v, int) else {-1: 0})
            vals.append(_materialize_array_toml(inner_dict, elem_type))
        elif isinstance(elem_type, BoolType):
            vals.append("true" if v else "false")
        else:
            vals.append(f'"{v if isinstance(v, int) else 0}"')
    return "[" + ", ".join(vals) + "]"


def _materialize_boundary_array(noir_type: ArrayType, rng: random.Random) -> str:
    """Recursively generate boundary-value array as a Prover.toml string."""
    size = noir_type.size
    elem_type = noir_type.element_type
    vals = []
    for _ in range(size):
        if isinstance(elem_type, ArrayType):
            vals.append(_materialize_boundary_array(elem_type, rng))
        elif isinstance(elem_type, BoolType):
            vals.append(rng.choice(["true", "false"]))
        elif isinstance(elem_type, IntegerType):
            lo, hi = _type_bounds(elem_type)
            pool = sorted(set(v for v in [lo, lo + 1, -1, 0, 1, hi - 1, hi] if lo <= v <= hi))
            vals.append(f'"{rng.choice(pool)}"')
        else:
            vals.append('"0"')
    return "[" + ", ".join(vals) + "]"


def _array_toml_value(var, model: dict | None) -> str:
    """Return the TOML value string for an array variable."""
    name = var.name
    noir_type = var.noir_type if isinstance(var.noir_type, ArrayType) else None
    if model and name in model:
        entry = model[name]
        if entry[0] == "Array" and isinstance(entry[1], dict) and noir_type is not None:
            return _materialize_array_toml(entry[1], noir_type)
    if noir_type is not None:
        return _materialize_array_toml({-1: 0}, noir_type)
    size = var.meta_info.get('array_size', 8)
    return "[" + ", ".join(['"0"'] * size) + "]"


def model_to_prover_toml(
    model: dict[str, tuple[str, str]],
    circuit,
    struct_map: dict[str, tuple[str, str]] | None = None,
    array_map: dict[str, tuple[str, int]] | None = None,
    tuple_map: dict[str, tuple[str, int]] | None = None,
    nesting_map: dict[str, tuple[str, str | int]] | None = None,
    param_name_map: dict[str, str] | None = None,
) -> str:
    """Convert a Z3 model to Noir's Prover.toml, respecting struct/array/tuple groupings."""
    sm = struct_map or {}
    am = array_map or {}
    tm = tuple_map or {}
    nm = nesting_map or {}
    pnm = param_name_map or {}  # original param → Noir param (may have _ prefix)

    def _pname(p: str) -> str:
        return pnm.get(p, p)

    array_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in am:
            param, idx = am[var.name]
            array_contents.setdefault(param, []).append((idx, var))

    tuple_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in tm:
            param, idx = tm[var.name]
            tuple_contents.setdefault(param, []).append((idx, var))

    # Params that are containers (not leaves) — must NOT be emitted in the flat loop.
    container_params: set[str] = {cont for _, (cont, _) in nm.items()}

    # Helper: render a leaf group as a TOML value string (for nesting into tuples).
    def _leaf_toml_value(leaf: str) -> str:
        if leaf in array_contents:
            members = sorted(array_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_var_toml_value(v, model) for _, v in members) + "]"
        if leaf in tuple_contents:
            members = sorted(tuple_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_var_toml_value(v, model) for _, v in members) + "]"
        # Struct rendered as TOML inline table.
        fields = [(f, v) for v in circuit.inputs
                  if v.name in sm and sm[v.name][0] == leaf
                  for _, f in [sm[v.name]]]
        if fields:
            inner = ", ".join(f"{f} = {_var_toml_value(v, model)}" for f, v in fields)
            return "{" + inner + "}"
        return '""'

    flat_lines: list[str] = []
    struct_lines: list[str] = []
    emitted: set[str] = set()

    # Emit flat/array/tuple top-level params (only those NOT nested inside something else).
    for var in circuit.inputs:
        name = var.name
        if name in am:
            param, _ = am[name]
            if param not in emitted and param not in nm and param not in container_params:
                emitted.add(param)
                members = sorted(array_contents[param], key=lambda x: x[0])
                vals = ", ".join(_var_toml_value(v, model) for _, v in members)
                flat_lines.append(f"{_pname(param)} = [{vals}]")
        elif name in tm:
            param, _ = tm[name]
            if param not in emitted and param not in nm and param not in container_params:
                emitted.add(param)
                members = sorted(tuple_contents[param], key=lambda x: x[0])
                vals = ", ".join(_var_toml_value(v, model) for _, v in members)
                flat_lines.append(f"{_pname(param)} = [{vals}]")
        elif name not in sm:
            if var.variable_type == VariableType.ARRAY:
                flat_lines.append(f"{_pname(name)} = {_array_toml_value(var, model)}")
            else:
                flat_lines.append(f"{_pname(name)} = {_var_toml_value(var, model)}")

    # Build struct blocks, including nested leaves as struct fields.
    struct_blocks: dict[str, list[str]] = {}
    for var in circuit.inputs:
        name = var.name
        if name in sm:
            param, field = sm[name]
            if param not in nm:  # only top-level structs here
                struct_blocks.setdefault(_pname(param), []).append(f"{field} = {_var_toml_value(var, model)}")

    # Inject nested leaves into their struct containers.
    for leaf, (container, access_key) in nm.items():
        if not container.startswith("g"):
            continue  # tuple container handled below
        field_name = access_key  # str for struct containers
        leaf_val = _leaf_toml_value(leaf)
        struct_blocks.setdefault(_pname(container), []).append(f"{field_name} = {leaf_val}")

    for pkey, field_lines in struct_blocks.items():
        struct_lines.append(f"[{pkey}]")
        struct_lines.extend(field_lines)

    # Handle tuple containers: tuples that have structs nested inside.
    # Collect per-tuple: list of (index, value_str) to build the full tuple array.
    tuple_nest_extra: dict[str, dict[int, str]] = {}
    for leaf, (container, access_key) in nm.items():
        if not container.startswith("tup_"):
            continue
        idx = access_key  # int for tuple containers
        leaf_val = _leaf_toml_value(leaf)
        tuple_nest_extra.setdefault(container, {})[idx] = leaf_val

    for container, extra in tuple_nest_extra.items():
        if container in emitted:
            continue
        emitted.add(container)
        # Build tuple array: regular members + nested struct elements.
        base_members = sorted(tuple_contents.get(container, []), key=lambda x: x[0])
        # Combine: each position is either a nested leaf or a regular var.
        all_positions: list[str] = []
        base_idx = 0
        struct_idx = 0
        positions: dict[int, str] = {i: _var_toml_value(v, model) for i, v in base_members}
        # Renumber: nested structs occupy positions 0..len(extra)-1, regular vars follow.
        # Actually, emit struct elements first in the tuple, then vars.
        struct_vals = [extra[k] for k in sorted(extra.keys())]
        var_vals = [_var_toml_value(v, model) for _, v in base_members]
        all_vals = struct_vals + var_vals
        flat_lines.append(f"{_pname(container)} = [{', '.join(all_vals)}]")

    return "\n".join(flat_lines + struct_lines) + "\n"


# ---------------------------------------------------------------------------
# Noir project helpers
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convert an arbitrary string to a valid Noir package name."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "circuit"


def create_noir_project(
    project_dir: Path,
    noir_source: str,
    prover_toml: str,
    package_name: str,
    smt_source: str = "",
    smt_filename: str = "",
    safe_math_source: str = "",
    z3_query: str = "",
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "main.nr").write_text(noir_source)
    if safe_math_source:
        (src_dir / "safe_math.nr").write_text(safe_math_source)
    (project_dir / "Prover.toml").write_text(prover_toml)
    (project_dir / "Nargo.toml").write_text(
        f'[package]\nname = "{package_name}"\ntype = "bin"\nauthors = [""]\n\n[dependencies]\n'
    )
    if smt_source and smt_filename:
        (project_dir / smt_filename).write_text(smt_source)
    if z3_query:
        (project_dir / "z3_query.smt2").write_text(z3_query)


def save_error(
    errors_dir: Path,
    category: str,
    folder_name: str,
    noir_source: str,
    prover_toml: str,
    package_name: str,
    smt_content: str,
    smt_filename: str,
    nargo_output: str,
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
    (tmp / "nargo_output.txt").write_text(nargo_output)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def _is_assertion_failure(output: str) -> bool:
    """Return True when nargo correctly rejected a witness — via runtime assertion
    failure, static always-false detection, or backend constraint unsatisfiability.
    Compilation/linking failures (missing module, type error, etc.) are NOT valid
    oracle rejections and must not be counted as one."""
    lower = output.lower()
    return (
        "assertion failed" in lower
        or "failed assertion" in lower
        or "assertion is always false" in lower        # nargo static analysis
        or "cannot satisfy constraint" in lower        # bb backend constraint failure
        or "failed constraint" in lower                # bb backend constraint failure
    )


def run_nargo_execute(project_dir: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run `nargo execute` in project_dir.  Returns (success, combined output)."""
    proc = subprocess.run(
        ["nargo", "execute"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Variable grouping (structs + arrays)
# ---------------------------------------------------------------------------

def compute_groupings(
    circuit,
    smt_content: str,
    allow_nesting: bool = False,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, int]], dict[str, tuple[str, int]]]:
    """
    Randomly assign each input variable to one of:
      flat / struct_N / array_N (booleans only) / tuple_N (any type)

    Groups with < 2 members are demoted to flat.
    Seed = int(md5(smt_content), 16).

    Returns:
      struct_map : {var_name → (param_name, field_name)}
      array_map  : {var_name → (param_name, index)}
      tuple_map  : {var_name → (param_name, index)}
    """
    if len(circuit.inputs) < 4:
        return {}, {}, {}, {}

    seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16)
    rng = random.Random(seed)

    buckets = ["flat", "struct_0", "struct_1", "array_0", "array_1", "tuple_0", "tuple_1"]
    assignments: dict[str, str] = {}
    for var in circuit.inputs:
        if var.variable_type == VariableType.ARRAY:
            # SMT array variables are already arrays — keep them flat (top-level params)
            assignments[var.name] = "flat"
            continue
        choice = rng.choice(buckets)
        if choice.startswith("array") and var.variable_type != VariableType.BOOLEAN:
            choice = "flat"
        assignments[var.name] = choice

    # Demote singleton groups to flat.
    from collections import Counter
    counts = Counter(assignments.values())
    for name, bucket in list(assignments.items()):
        if bucket != "flat" and counts[bucket] < 2:
            assignments[name] = "flat"

    struct_map: dict[str, tuple[str, str]] = {}
    array_counters: dict[str, int] = {}
    array_map: dict[str, tuple[str, int]] = {}

    for var in circuit.inputs:
        bucket = assignments[var.name]
        if bucket.startswith("struct"):
            param = bucket.replace("_", "")   # struct_0 → struct0 → g0
            param = "g" + bucket[-1]          # g0 / g1
            struct_map[var.name] = (param, var.name)
        elif bucket.startswith("array"):
            param = "bools_" + bucket[-1]     # bools_0 / bools_1
            idx = array_counters.get(param, 0)
            array_counters[param] = idx + 1
            array_map[var.name] = (param, idx)

    tuple_counters: dict[str, int] = {}
    tuple_map: dict[str, tuple[str, int]] = {}
    for var in circuit.inputs:
        bucket = assignments[var.name]
        if bucket.startswith("tuple"):
            param = "tup_" + bucket[-1]
            idx = tuple_counters.get(param, 0)
            tuple_counters[param] = idx + 1
            tuple_map[var.name] = (param, idx)

    # Demote singleton tuple groups to flat.
    from collections import Counter
    tuple_counts = Counter(p for p, _ in tuple_map.values())
    tuple_map = {k: v for k, v in tuple_map.items() if tuple_counts[v[0]] >= 2}

    if not allow_nesting:
        return struct_map, array_map, tuple_map, {}

    # --- Second-level nesting (recursive mode only) ---
    # Randomly assign leaf groups as fields/elements of container groups.
    # nesting_map: {leaf_group → (container_group, access_key)}
    #   access_key is a field name (str) for structs, or an int index for tuples.
    # Rules:
    #   - Only struct_0 and struct_1 can be containers (structs are the only composite type)
    #   - OR tuple_0 / tuple_1 as containers (struct can be a tuple element)
    #   - A container cannot itself be nested
    #   - Each container holds at most one nested group per nesting run
    nesting_map: dict[str, tuple[str, str | int]] = {}

    # Collect which leaf groups exist (non-empty).
    leaf_groups = (
        [f"bools_{i}" for i in range(2) if any(v[0] == f"bools_{i}" for v in array_map.values())] +
        [f"g{i}"    for i in range(2) if any(v[0] == f"g{i}"    for v in struct_map.values())] +
        [f"tup_{i}" for i in range(2) if any(v[0] == f"tup_{i}" for v in tuple_map.values())]
    )
    struct_containers = [f"g{i}" for i in range(2) if any(v[0] == f"g{i}" for v in struct_map.values())]
    tuple_containers  = [f"tup_{i}" for i in range(2) if any(v[0] == f"tup_{i}" for v in tuple_map.values())]
    all_containers = struct_containers + tuple_containers

    used_as_container: set[str] = set()
    used_as_nested: set[str] = set()
    field_counters: dict[str, int] = {}  # how many fields added to each container
    tuple_nest_counters: dict[str, int] = {}

    for leaf in rng.sample(leaf_groups, len(leaf_groups)):  # random order
        if leaf in used_as_nested:
            continue
        eligible_containers = [c for c in all_containers
                                if c != leaf and c not in used_as_nested
                                and field_counters.get(c, 0) < 2]  # max 2 nested per container
        if not eligible_containers:
            continue
        if not rng.choice([True, False]):  # 50% chance of nesting
            continue
        container = rng.choice(eligible_containers)
        used_as_container.add(container)
        used_as_nested.add(leaf)
        field_counters[container] = field_counters.get(container, 0) + 1
        if container.startswith("tup_"):
            idx = tuple_nest_counters.get(container, 0)
            tuple_nest_counters[container] = idx + 1
            nesting_map[leaf] = (container, idx)
        else:
            # struct container: use a field name based on the leaf group
            field_name = leaf.replace("_", "").replace("bools", "arr").replace("tup", "tup")
            nesting_map[leaf] = (container, field_name)

    return struct_map, array_map, tuple_map, nesting_map


# ---------------------------------------------------------------------------
# smtfuzz generation
# ---------------------------------------------------------------------------

def generate_smt(config: dict, timeout: int = 30) -> str:
    """Call smtfuzz with flags from config dict and return the generated formula.
    List values are resolved by picking one element at random (allows e.g. multiple logics).
    Re-rolls up to 20 times if the formula contains array sorts our parser cannot handle
    (non-Int index, Bool-indexed, or nested-array-indexed arrays)."""
    import re
    # Matches any (Array X ...) where X is not 'Int' — catches Bool-indexed,
    # nested-array-indexed, and other unsupported index sorts.
    _BAD_ARRAY_SORT = re.compile(r'\(Array\s+(?!Int[\s)])' )

    def _build_cmd() -> list[str]:
        cmd = ["smtfuzz"]
        for key, value in config.items():
            if value is None:
                continue
            if isinstance(value, dict):
                value = random.choices(list(value.keys()), weights=list(value.values()), k=1)[0]
            elif isinstance(value, list):
                value = random.choice(value)
            cmd += [f"--{key}", str(value)]
        return cmd

    for _ in range(20):
        proc = subprocess.run(_build_cmd(), capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"smtfuzz failed: {proc.stderr.strip()}")
        formula = proc.stdout
        if not _BAD_ARRAY_SORT.search(formula):
            return formula
    # If every attempt had an unsupported sort, return the last one and let the
    # parser reject it normally (avoids an infinite loop when the config always
    # produces unsupported sorts).
    return formula


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------



def build_boundary_prover_toml(
    circuit,
    rng: random.Random,
    struct_map: dict[str, tuple[str, str]] | None = None,
    array_map: dict[str, tuple[str, int]] | None = None,
    tuple_map: dict[str, tuple[str, int]] | None = None,
    nesting_map: dict[str, tuple[str, str | int]] | None = None,
    param_name_map: dict[str, str] | None = None,
) -> str:
    """Build a boundary-value Prover.toml respecting struct/array/tuple/nesting groupings."""
    sm = struct_map or {}
    am = array_map or {}
    tm = tuple_map or {}
    nm = nesting_map or {}
    pnm = param_name_map or {}

    def _pname(p: str) -> str:
        return pnm.get(p, p)

    def _bval(var) -> str:
        if var.variable_type == VariableType.BOOLEAN:
            return rng.choice(["true", "false"])
        t = var.noir_type if isinstance(var.noir_type, IntegerType) else IntegerType(64, True)
        lo, hi = _type_bounds(t)
        pool = sorted(set(v for v in [lo, lo+1, -1, 0, 1, hi-1, hi] if lo <= v <= hi))
        return str(rng.choice(pool))

    def _bfmt(var) -> str:
        v = _bval(var)
        return v if var.variable_type == VariableType.BOOLEAN else f'"{v}"'

    array_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in am:
            param, idx = am[var.name]
            array_contents.setdefault(param, []).append((idx, var))

    tuple_contents: dict[str, list] = {}
    for var in circuit.inputs:
        if var.name in tm:
            param, idx = tm[var.name]
            tuple_contents.setdefault(param, []).append((idx, var))

    def _leaf_boundary(leaf: str) -> str:
        if leaf in array_contents:
            members = sorted(array_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_bval(v) for _, v in members) + "]"
        if leaf in tuple_contents:
            members = sorted(tuple_contents[leaf], key=lambda x: x[0])
            return "[" + ", ".join(_bfmt(v) for _, v in members) + "]"
        fields = [(f, v) for v in circuit.inputs
                  if v.name in sm and sm[v.name][0] == leaf
                  for _, f in [sm[v.name]]]
        if fields:
            inner = ", ".join(f"{f} = {_bfmt(v)}" for f, v in fields)
            return "{" + inner + "}"
        return '""'

    container_params_b: set[str] = {cont for _, (cont, _) in nm.items()}

    flat_lines: list[str] = []
    struct_lines: list[str] = []
    emitted: set[str] = set()

    for var in circuit.inputs:
        name = var.name
        if name in am:
            param, _ = am[name]
            if param not in emitted and param not in nm and param not in container_params_b:
                emitted.add(param)
                members = sorted(array_contents[param], key=lambda x: x[0])
                flat_lines.append(f"{_pname(param)} = [{', '.join(_bval(v) for _, v in members)}]")
        elif name in tm:
            param, _ = tm[name]
            if param not in emitted and param not in nm and param not in container_params_b:
                emitted.add(param)
                members = sorted(tuple_contents[param], key=lambda x: x[0])
                flat_lines.append(f"{_pname(param)} = [{', '.join(_bfmt(v) for _, v in members)}]")
        elif name not in sm:
            if var.variable_type == VariableType.ARRAY:
                if isinstance(var.noir_type, ArrayType):
                    flat_lines.append(f"{_pname(name)} = {_materialize_boundary_array(var.noir_type, rng)}")
                else:
                    size = var.meta_info.get('array_size', 8)
                    vals = ', '.join(f'"{rng.randint(-(2**62), 2**62 - 1)}"' for _ in range(size))
                    flat_lines.append(f"{_pname(name)} = [{vals}]")
            else:
                flat_lines.append(f"{_pname(name)} = {_bfmt(var)}")

    struct_blocks: dict[str, list[str]] = {}
    for var in circuit.inputs:
        name = var.name
        if name in sm:
            param, field = sm[name]
            if param not in nm:
                struct_blocks.setdefault(_pname(param), []).append(f"{field} = {_bfmt(var)}")

    for leaf, (container, access_key) in nm.items():
        if container.startswith("g"):
            struct_blocks.setdefault(_pname(container), []).append(f"{access_key} = {_leaf_boundary(leaf)}")

    for pkey, field_lines in struct_blocks.items():
        struct_lines.append(f"[{pkey}]")
        struct_lines.extend(field_lines)

    for leaf, (container, access_key) in nm.items():
        if container.startswith("tup_") and container not in emitted:
            emitted.add(container)
            struct_vals = [_leaf_boundary(leaf)]
            var_vals = [_bfmt(v) for _, v in sorted(tuple_contents.get(container, []), key=lambda x: x[0])]
            flat_lines.append(f"{_pname(container)} = [{', '.join(struct_vals + var_vals)}]")

    return "\n".join(flat_lines + struct_lines) + "\n"


def _is_unsat_with_fix(
    smt_content: str,
    var_name: str,
    fixed_value: str,
    circuit,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Return (is_unsat, z3_query) — query saved for reproduction if oracle fires."""
    base = _strip_check_commands(smt_content)
    bounds = build_type_bounds(circuit)
    fix = f"(assert (= {var_name} {fixed_value}))"
    query = base + "\n" + "\n".join(bounds) + "\n" + fix + "\n(check-sat)\n"
    proc = subprocess.run(
        ["z3", "-in"], input=query,
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout.strip().startswith("unsat"), query


_INT_FLIP_DELTAS = [1, -1]


def find_sat_oracle_input(
    smt_content: str,
    circuit,
    model: dict,
) -> tuple[dict, str] | None:
    """
    Try minimal single-variable modifications that Z3 confirms make the formula UNSAT.
    Returns (modified_model, z3_query) or None.
    """
    # --- Boolean flips ---
    for var in circuit.inputs:
        if var.variable_type != VariableType.BOOLEAN:
            continue
        current = model.get(var.name, ('Bool', 'false'))[1]
        flipped = 'false' if current == 'true' else 'true'
        try:
            is_unsat, query = _is_unsat_with_fix(smt_content, var.name, flipped, circuit)
            if is_unsat:
                return {**model, var.name: ('Bool', flipped)}, query
        except subprocess.TimeoutExpired:
            continue

    # --- Integer mutations (±deltas) ---
    for var in circuit.inputs:
        if var.variable_type != VariableType.INTEGER:
            continue
        raw = model.get(var.name, ('Int', '0'))[1]
        try:
            current_val = int(raw)
        except ValueError:
            continue
        t = var.noir_type if isinstance(var.noir_type, IntegerType) else IntegerType(64, True)
        lo, hi = _type_bounds(t)
        for delta in _INT_FLIP_DELTAS:
            new_val = current_val + delta
            if not (lo <= new_val <= hi):
                continue
            try:
                is_unsat, query = _is_unsat_with_fix(smt_content, var.name, str(new_val), circuit)
                if is_unsat:
                    return {**model, var.name: ('Int', str(new_val))}, query
            except subprocess.TimeoutExpired:
                continue

    return None


_STATS_INTERVAL = 100


def _write_recent_stats(path: Path, stats: dict, start: int, end: int) -> None:
    lines = [
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Iterations: {start}-{end}",
        "",
        f"sat         : {stats.get('sat_ok',0)} ok  |  {stats.get('sat_error',0)} error  |  {stats.get('sat_overflow',0)} overflow",
        f"unsat       : {stats.get('unsat_ok',0)} ok  |  {stats.get('unsat_error',0)} bug  |  {stats.get('unsat_pipeline_error',0)} pipeline",
        f"sat_to_unsat: {stats.get('oracle_ok',0)} ok  |  {stats.get('oracle_bug',0)} bug  |  {stats.get('oracle_skip',0)} no-flip  |  {stats.get('oracle_pipeline_error',0)} pipeline",
        f"timeout     : {stats.get('n_timeout',0)}",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMT-LIB → Noir circuit translation + witness generation."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Directory containing .smt / .smt2 files. "
             "If omitted, formulas are generated on-the-fly using smtfuzz.",
    )
    parser.add_argument(
        "--smtfuzz-config",
        default="smtfuzz_config.json",
        metavar="FILE",
        help="JSON file with smtfuzz flags (default: smtfuzz_config.json). "
             "Used only in generate mode.",
    )
    parser.add_argument(
        "--generator-config",
        default="generator_config.json",
        metavar="FILE",
        help="JSON file with generator options (default: generator_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="obj",
        help="Root directory for all runs (default: ./obj). Each run is stored under <output-dir>/run-<ID>/.",
    )
    parser.add_argument(
        "--max-witnesses",
        type=int,
        default=1,
        metavar="N",
        help="Maximum satisfying witnesses to generate per formula (default: 1).",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of input formulas to process (default: unlimited).",
    )
    parser.add_argument(
        "--z3-timeout",
        type=int,
        default=30,
        metavar="SECS",
        help="Seconds to allow Z3 per query (default: 30).",
    )
    parser.add_argument(
        "--nargo-timeout",
        type=int,
        default=120,
        metavar="SECS",
        help="Seconds to allow nargo execute per witness (default: 120).",
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=10,
        metavar="N",
        help="Keep only the N most recent run directories, deleting older ones (default: 10).",
    )
    parser.add_argument(
        "--cleanup-every",
        type=int,
        default=500,
        metavar="N",
        help="Delete out/ of the current run every N programs to free disk space (default: 500).",
    )
    parser.add_argument(
        "--max-noir-lines",
        type=int,
        default=20000,
        metavar="N",
        help="Skip programs whose generated main.nr exceeds N lines to avoid OOM (default: 20000).",
    )
    args = parser.parse_args()

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
    # CLI args override config values; config values override hardcoded defaults.
    if args.z3_timeout == 30:      # still at default → use config
        args.z3_timeout = gen_config.get("solver_timeout", 30)
    if args.nargo_timeout == 120:  # still at default → use config
        args.nargo_timeout = gen_config.get("nargo_timeout", 120)


    output_dir = Path(args.output_dir)
    generate_mode = args.input_dir is None

    # --- folder mode ---
    if not generate_mode:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            sys.exit(f"Error: {input_dir} is not a directory.")
        smt_files = sorted(input_dir.glob("*.smt")) + sorted(input_dir.glob("*.smt2"))
        if not smt_files:
            sys.exit(f"No .smt / .smt2 files found in {input_dir}.")
        if args.max_programs is not None:
            smt_files = smt_files[: args.max_programs]

    # --- generate mode: load smtfuzz config ---
    smtfuzz_config: dict = {}
    if generate_mode:
        config_path = Path(args.smtfuzz_config)
        if not config_path.exists():
            sys.exit(f"Error: smtfuzz config not found at {config_path}. "
                     f"Create it or pass --smtfuzz-config <file>.")
        with config_path.open() as f:
            smtfuzz_config = json.load(f)
        n_generate = args.max_programs or 100   # default 100 in generate mode
        print(f"Generate mode: smtfuzz config={config_path}, target={n_generate} formula(s)")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id  = secrets.token_hex(4)
    run_dir = output_dir / f"run-{timestamp}-{run_id}"
    sat_dir          = run_dir / "out" / "sat"
    unsat_dir        = run_dir / "out" / "unsat"
    sat_to_unsat_dir = run_dir / "out" / "sat_to_unsat"
    errors_dir       = run_dir / "errors"
    sat_dir.mkdir(parents=True, exist_ok=True)
    unsat_dir.mkdir(parents=True, exist_ok=True)
    sat_to_unsat_dir.mkdir(parents=True, exist_ok=True)
    (errors_dir / "error").mkdir(parents=True, exist_ok=True)
    (errors_dir / "overflow").mkdir(parents=True, exist_ok=True)
    (errors_dir / "timeout").mkdir(parents=True, exist_ok=True)
    (errors_dir / "parse-errors").mkdir(parents=True, exist_ok=True)
    (errors_dir / "smt_solver_errors").mkdir(parents=True, exist_ok=True)
    (errors_dir / "z3_error_cvc5_succ").mkdir(parents=True, exist_ok=True)
    (errors_dir / "sat_to_unsat_pipeline").mkdir(parents=True, exist_ok=True)
    (errors_dir / "unsat_pipeline").mkdir(parents=True, exist_ok=True)

    # Enforce --keep-runs: delete only the out/ of old runs — errors/ is never touched.
    all_runs = sorted(output_dir.glob("run-*"))   # lexicographic = chronological
    excess = len(all_runs) - args.keep_runs
    for old_run in all_runs[:excess]:
        out_to_delete = old_run / "out"
        if out_to_delete.exists():
            shutil.rmtree(out_to_delete, ignore_errors=True)
            print(f"[cleanup] cleared out/ from: {old_run.name}")

    if generate_mode:
        print(f"Run ID: {run_id}  →  {run_dir}")
    else:
        print(f"Found {len(smt_files)} formula(s) in {input_dir} (cap: {args.max_programs or 'none'})")
        print(f"Run ID: {run_id}  →  {run_dir}")

    emitter = EmitVisitor()
    sat_ok = sat_error = sat_overflow = 0
    unsat_ok = unsat_error = unsat_pipeline_error = 0
    oracle_ok = oracle_bug = oracle_skip = oracle_pipeline_error = 0
    n_timeout = skipped = 0

    recent_stats_path = run_dir / "recent_stats.txt"
    _window_snap: dict[str, int] = {}
    _window_start = 1

    def _iter_formulas():
        """Yield (stem, smt_content) for either folder or generate mode."""
        if not generate_mode:
            for smt_file in smt_files:
                yield smt_file.stem, smt_file.read_text()
        else:
            idx = 0
            while idx < n_generate:
                try:
                    content = generate_smt(smtfuzz_config)
                except Exception as exc:
                    print(f"  [smtfuzz] generation error: {exc}")
                    continue
                try:
                    parse_smtlib2(content)
                except Exception:
                    continue  # silently retry; parse failures don't count
                yield f"gen-{idx:04d}", content
                idx += 1

    n_processed = 0
    for stem, smt_content in _iter_formulas():
        if args.cleanup_every > 0 and n_processed > 0 and n_processed % args.cleanup_every == 0:
            for sub in (sat_dir, unsat_dir, sat_to_unsat_dir):
                if sub.exists():
                    shutil.rmtree(sub, ignore_errors=True)
                    sub.mkdir(parents=True, exist_ok=True)
            print(f"[cleanup] cleared out/ after {n_processed} programs")
        if n_processed > 0 and n_processed % _STATS_INTERVAL == 0:
            _current = {
                "sat_ok": sat_ok, "sat_error": sat_error, "sat_overflow": sat_overflow,
                "unsat_ok": unsat_ok, "unsat_error": unsat_error,
                "unsat_pipeline_error": unsat_pipeline_error,
                "oracle_ok": oracle_ok, "oracle_bug": oracle_bug,
                "oracle_skip": oracle_skip, "oracle_pipeline_error": oracle_pipeline_error,
                "n_timeout": n_timeout,
            }
            _delta = {k: _current[k] - _window_snap.get(k, 0) for k in _current}
            _write_recent_stats(recent_stats_path, _delta, _window_start, n_processed)
            _window_snap = _current
            _window_start = n_processed + 1
        n_processed += 1

        smt_filename = f"{stem}.smt2"
        print(f"\n=== {stem} ===")

        # 1. Parse → Circuit IR
        try:
            circuit = parse_smtlib2(smt_content)
        except Exception as exc:
            # In generate mode this path is unreachable (_iter_formulas pre-validates).
            # In folder mode, save to parse-errors and count as an error.
            print(f"  ERROR (parse error): {exc}")
            sat_error += 1
            dest = errors_dir / "parse-errors" / f"obj.{stem}-parse"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(f"PARSE ERROR: {exc}\n")
            continue

        declared = {v.name for v in circuit.inputs}
        int_var_names = [v.name for v in circuit.inputs if v.variable_type == VariableType.INTEGER]

        recompute_types(circuit, smt_content, sample_int_type=gen_config.get("sample_int_type", True),
                        integer_signedness=gen_config.get("integer_signedness", "signed"))
        _type_bounds_clauses = build_type_bounds(circuit)
        if int_var_names:
            types_summary = ", ".join(
                f"{v.name}:{v.noir_type}"
                for v in circuit.inputs if v.variable_type == VariableType.INTEGER
            )
            print(f"  Types: {types_summary}")

        if variables_bundling == "flat":
            struct_map, array_map, tuple_map, nesting_map = {}, {}, {}, {}
        else:
            struct_map, array_map, tuple_map, nesting_map = compute_groupings(
                circuit, smt_content,
                allow_nesting=(variables_bundling == "recursive"),
            )
        # Seeded RNG for per-expression randomization (comptime, helpers).
        # Offset by 3 to be independent of the grouping seed (offset 0) and
        # the UNSAT boundary seed (offset 1).
        ir_rng = random.Random(int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 3)

        # 2. Circuit IR → Noir source
        try:
            visitor = IR2NoirVisitor(
                struct_map=struct_map,
                array_map=array_map,
                tuple_map=tuple_map,
                nesting_map=nesting_map,
                rng=ir_rng,
            )
            noir_doc = visitor.transform(circuit)
            noir_source = emitter.emit(noir_doc)
            safe_math_src = build_safe_math_source(visitor.safe_fns_needed) if visitor.safe_fns_needed else ""
            param_name_map = visitor.param_name_map
        except Exception as exc:
            import traceback
            print(f"  ERROR (Noir translation error): {exc}")
            sat_error += 1
            dest = errors_dir / "error" / f"obj.{stem}-translation"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(
                f"TRANSLATION ERROR: {exc}\n\n{traceback.format_exc()}"
            )
            continue

        # Guard against programs so large they would OOM the compiler.
        noir_lines = noir_source.count("\n")
        if args.max_noir_lines > 0 and noir_lines > args.max_noir_lines:
            print(f"  [{stem}] SKIP (main.nr too large: {noir_lines} lines > {args.max_noir_lines})")
            skipped += 1
            n_processed += 1
            continue

        # 3. Find a satisfying witness using the witness generator module.
        z3_failed_error: Exception | None = None
        active_solver = solver
        models: list[dict] = []

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
                    print(f"  Z3 failed ({z3_exc}), retrying with cvc5 ...")
                    models, _z3_sat_query = _run_solver("cvc5")
                    active_solver = "cvc5"
            else:
                models, _z3_sat_query = _run_solver(solver)
                active_solver = solver
        except subprocess.TimeoutExpired:
            print(f"  ERROR (solver timed out)")
            n_timeout += 1
            continue
        except FileNotFoundError as exc:
            sys.exit(f"Error: solver not found in PATH: {exc}")
        except Exception as exc:
            # Both solvers failed (or single solver failed).
            label = f"z3_then_cvc5" if solver == "z3_to_cvc5" else active_solver
            print(f"  ERROR ({label} error): {exc}")
            sat_error += 1
            dest = errors_dir / "smt_solver_errors" / f"obj.{stem}-{label}error"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "src").mkdir(exist_ok=True)
            (dest / "src" / "main.nr").write_text(noir_source)
            (dest / "nargo_output.txt").write_text(f"SOLVER ERROR: {exc}\n")
            continue

        # If Z3 failed but cvc5 succeeded — save to dedicated folder for analysis.
        if z3_failed_error is not None:
            dest = errors_dir / "z3_error_cvc5_succ" / f"obj.{stem}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / smt_filename).write_text(smt_content)
            (dest / "nargo_output.txt").write_text(
                f"Z3 ERROR: {z3_failed_error}\ncvc5 found {len(models)} model(s)\n"
            )
            # Continue processing — use the cvc5 models for witness generation.

        if not models:
            # UNSAT oracle: run N boundary-value samples — all must be rejected.
            seed = int(hashlib.md5(smt_content.encode()).hexdigest(), 16) + 1
            rng_unsat = random.Random(seed)
            # Build the exact Z3 query that declared this formula UNSAT so it can
            # be reproduced independently (saved alongside any oracle violations).
            _z3_unsat_query = (
                _strip_check_commands(smt_content)
                + "\n"
                + "\n".join(_type_bounds_clauses)
                + "\n(check-sat)\n(get-model)\n"
            )
            for idx in range(args.max_witnesses):
                folder_name = f"obj.{stem}-{idx}"
                pkg_name    = _sanitize_name(folder_name)
                project_dir = unsat_dir / folder_name
                boundary_toml = build_boundary_prover_toml(circuit, rng_unsat, struct_map, array_map, tuple_map, nesting_map, param_name_map)
                create_noir_project(project_dir, noir_source, boundary_toml, pkg_name,
                                    smt_source=smt_content, smt_filename=smt_filename,
                                    safe_math_source=safe_math_src, z3_query=_z3_unsat_query)
                print(f"  [{folder_name}] unsat boundary  ... ", end="", flush=True)
                timed_out = False
                try:
                    u_ok, u_out = run_nargo_execute(project_dir, timeout=args.nargo_timeout)
                except subprocess.TimeoutExpired:
                    timed_out, u_ok, u_out = True, False, ""
                except FileNotFoundError:
                    sys.exit("Error: 'nargo' not found in PATH.")
                if u_ok:
                    print("BUG")
                    unsat_error += 1
                    dest_path = errors_dir / "error" / (folder_name + "_unsat_oracle")
                    save_error(errors_dir, "error", folder_name + "_unsat_oracle",
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT ORACLE VIOLATION: nargo accepted boundary input\n" + u_out,
                               safe_math_src, _z3_unsat_query)
                elif timed_out:
                    print("TIMEOUT")
                    n_timeout += 1
                elif _is_assertion_failure(u_out):
                    print("ok (correctly rejected)")
                    unsat_ok += 1
                else:
                    print("PIPELINE ERROR")
                    unsat_pipeline_error += 1
                    save_error(errors_dir, "unsat_pipeline", folder_name,
                               noir_source, boundary_toml, pkg_name,
                               smt_content, smt_filename,
                               "UNSAT PIPELINE ERROR: nargo failed but not via assertion\n" + u_out,
                               safe_math_src, _z3_unsat_query)
            continue

        print(f"  {len(models)} satisfying model(s) found")

        # 4. For each model: create Noir project + run nargo execute
        for idx, model in enumerate(models):
            folder_name = f"obj.{stem}-{idx}"
            pkg_name = _sanitize_name(folder_name)
            project_dir = sat_dir / folder_name

            prover_toml = model_to_prover_toml(model, circuit, struct_map, array_map, tuple_map, nesting_map, param_name_map)
            create_noir_project(
                project_dir, noir_source, prover_toml, pkg_name,
                smt_source=smt_content, smt_filename=smt_filename,
                safe_math_source=safe_math_src, z3_query=_z3_sat_query,
            )

            print(f"  [{folder_name}] nargo execute   ... ", end="", flush=True)
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
                print(f"OK  →  {witness_path}")
                sat_ok += 1

                # SAT oracle: find a minimal UNSAT modification and assert nargo rejects it.
                oracle_result = find_sat_oracle_input(smt_content, circuit, model)
                if oracle_result is not None:
                    oracle_model, oracle_z3_query = oracle_result
                    oracle_toml = model_to_prover_toml(oracle_model, circuit, struct_map, array_map, tuple_map, nesting_map, param_name_map)
                    oracle_dir = sat_to_unsat_dir / folder_name
                    create_noir_project(oracle_dir, noir_source, oracle_toml, pkg_name,
                                        smt_source=smt_content, smt_filename=smt_filename,
                                        safe_math_source=safe_math_src, z3_query=oracle_z3_query)
                    print(f"  [{folder_name}] sat oracle      ... ", end="", flush=True)
                    try:
                        o_ok, o_out = run_nargo_execute(oracle_dir, timeout=args.nargo_timeout)
                    except subprocess.TimeoutExpired:
                        o_ok, o_out = False, ""
                    if o_ok:
                        print("BUG")
                        oracle_bug += 1
                        save_error(errors_dir, "error", folder_name + "_sat_oracle",
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT ORACLE VIOLATION: nargo accepted a Z3-UNSAT input\n" + o_out,
                                   safe_math_src, oracle_z3_query)
                    elif _is_assertion_failure(o_out):
                        print("ok (correctly rejected)")
                        oracle_ok += 1
                    else:
                        print("PIPELINE ERROR")
                        oracle_pipeline_error += 1
                        save_error(errors_dir, "sat_to_unsat_pipeline", folder_name,
                                   noir_source, oracle_toml, pkg_name,
                                   smt_content, smt_filename,
                                   "SAT_TO_UNSAT PIPELINE ERROR: nargo failed but not via assertion\n" + o_out,
                                   safe_math_src, oracle_z3_query)
                else:
                    print(f"  [{folder_name}] sat oracle      ... skipped (no UNSAT mutation found)")
                    oracle_skip += 1

            elif timed_out:
                print("TIMEOUT")
                n_timeout += 1
                save_error(errors_dir, "timeout", folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, "",
                           safe_math_src, _z3_sat_query)
            else:
                is_overflow = "overflow" in output.lower()
                category = "overflow" if is_overflow else "error"
                print("OVERFLOW" if is_overflow else "ERROR")
                first_error = next(
                    (l for l in output.splitlines() if "error" in l.lower()),
                    output.splitlines()[0] if output.strip() else "(no output)"
                )
                print(f"    {first_error}")
                if is_overflow:
                    sat_overflow += 1
                else:
                    sat_error += 1
                save_error(errors_dir, category, folder_name,
                           noir_source, prover_toml, pkg_name,
                           smt_content, smt_filename, output,
                           safe_math_src, _z3_sat_query)

    total_bugs = sat_error + unsat_error + oracle_bug
    print(f"\n{'='*60}")
    print(f"sat         : {sat_ok} ok  |  {sat_error} error  |  {sat_overflow} overflow")
    print(f"unsat       : {unsat_ok} ok  |  {unsat_error} bug  |  {unsat_pipeline_error} pipeline")
    print(f"sat_to_unsat: {oracle_ok} ok  |  {oracle_bug} bug  |  {oracle_skip} no-flip  |  {oracle_pipeline_error} pipeline")
    print(f"timeout     : {n_timeout}")
    print(f"{'─'*40}")
    print(f"total bugs  : {total_bugs}  (overflows not counted as bugs)")


if __name__ == "__main__":
    main()
