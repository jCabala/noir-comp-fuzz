# noir-comp-fuzz

A compiler fuzzer for [Noir](https://noir-lang.org/). It generates SMT-LIB formulas (or reads existing ones), translates them to Noir circuits, and checks for bugs in `nargo` and the Barretenberg (`bb`) proving backend.

## Prerequisites

- Python 3.11+
- [`nargo`](https://noir-lang.org/docs/getting_started/installation/) on `$PATH`
- [`bb`](https://github.com/AztecProtocol/aztec-packages) installed at `~/.bb/bb` (only needed for proof scripts)
- [`z3`](https://github.com/Z3Prover/z3) and/or [`cvc5`](https://cvc5.github.io/) on `$PATH`
- [smtfuzz](https://github.com/testsmt/smtfuzz) on `$PATH` (only needed in generate mode)

## Scripts

All scripts are run from the repository root.

### `scripts/smt_to_noir_witnesses.py` — witness generation

Translates SMT-LIB formulas to Noir circuits and checks them with `nargo execute`.

```bash
# Generate mode: smtfuzz produces formulas on-the-fly
python scripts/smt_to_noir_witnesses.py

# File mode: process a directory of .smt / .smt2 files
python scripts/smt_to_noir_witnesses.py path/to/smt_files/
```

| Flag | Default | Description |
|---|---|---|
| `--smtfuzz-config FILE` | `smtfuzz_config.json` | smtfuzz options (generate mode only) |
| `--generator-config FILE` | `generator_config.json` | Noir generator options |
| `--output-dir DIR` | `obj` | Root for run output; each run goes in `<dir>/run-<ID>/` |
| `--max-witnesses N` | `1` | Satisfying models to enumerate per formula |
| `--max-programs N` | unlimited | Cap on total formulas processed |
| `--z3-timeout SECS` | `30` | Z3 time limit per query |
| `--nargo-timeout SECS` | `120` | `nargo execute` time limit |
| `--keep-runs N` | `10` | Keep only the N most recent run directories |
| `--cleanup-every N` | `500` | Delete `out/` every N programs to free disk |
| `--max-noir-lines N` | `20000` | Skip programs whose `main.nr` exceeds N lines |

Bugs are written to `<output-dir>/run-<ID>/errors/`.

---

### `scripts/smt_to_noir_bb_proofs.py` — proof generation

Extends the witness pipeline with Barretenberg proof generation and verification. Adds two more bug oracles: `bb_prove_error` and `bb_verify_error`.

```bash
python scripts/smt_to_noir_bb_proofs.py [input_dir] [options]
```

Inherits all flags from the witness script, plus:

| Flag | Default | Description |
|---|---|---|
| `--witness-workers N` | `4` | Parallel `nargo` workers |
| `--bb-workers M` | `2` | Parallel `bb` workers |
| `--bb-timeout SECS` | `300` | Per-step `bb` time limit |

---

### `scripts/smt_to_noir_acir_oracle.py` — ACIR oracle

Extends the proof pipeline with an ACIR-level SMT oracle: compiled ACIR constraints are translated back to SMT and re-checked with Z3 to detect over- or under-constrained circuits.

```bash
python scripts/smt_to_noir_acir_oracle.py [input_dir] [options]
```

Inherits all flags from the proof script, plus:

| Flag | Default | Description |
|---|---|---|
| `--acir-oracle-timeout SECS` | `60` | Z3 timeout for the ACIR oracle |

---

### `scripts/acir_smt_oracle.py` — standalone ACIR oracle

Run the ACIR SMT oracle on a single compiled Nargo project.

```bash
python scripts/acir_smt_oracle.py path/to/nargo_project/ [--timeout 60] [--save-smt] [--no-model] [--force]
```

---

### `scripts/acir_inspector.py` — ACIR inspector

Compile a Nargo project and explore its ACIR opcodes.

```bash
python scripts/acir_inspector.py path/to/nargo_project/ [options]
```

| Flag | Description |
|---|---|
| `--stats` | Print opcode-type breakdown and exit |
| `--filter-type TYPE` | Show only opcodes matching TYPE (e.g. `ASSERT`, `BLACKBOX`) |
| `--witness WN` | Show only opcodes referencing witness `WN` (e.g. `w42`) |
| `--func N` | Show only ACIR function N (0-indexed) |
| `--no-brillig` | Suppress unconstrained Brillig listings |
| `--no-color` | Disable ANSI colour output |
| `--no-source` | Do not print Noir source above ACIR |
| `--force` | Pass `--force` to nargo (recompile from scratch) |

---

### `scripts/minimize_bb_error.py` — error minimizer

Delta-debug a `bb_prove_error` case by removing SMT clauses one at a time until the bug can no longer be reproduced without them.

```bash
python scripts/minimize_bb_error.py errors/bb_prove_error/<case>/ [--output DIR] \
    [--z3-timeout N] [--nargo-timeout N] [--bb-timeout N]
```

## Configuration files

### `generator_config.json`

Controls how SMT formulas are translated to Noir.

| Key | Values | Description |
|---|---|---|
| `variables_bundling` | `"flat"` \| `"simple"` \| `"recursive"` | How SMT variables are grouped into Noir structs |
| `solver` | `"z3"` \| `"cvc5"` \| `"z3_to_cvc5"` | Solver used for witness enumeration; `z3_to_cvc5` tries Z3 first, falls back to cvc5 |
| `solver_timeout` | integer (seconds) | Per-query solver timeout |
| `nargo_timeout` | integer (seconds) | `nargo execute` timeout |
| `sample_int_type` | `true` \| `false` | Randomly sample integer bit-widths |
| `integer_signedness` | `"signed"` \| `"unsigned"` \| `"mixed"` | Signedness of generated integer types |

### `smtfuzz_config.json`

Controls the smtfuzz formula generator (generate mode only).

| Key | Values | Description |
|---|---|---|
| `strategy` | `"noinc"` \| … | smtfuzz mutation strategy |
| `logic` | `{ "QF_NIA": w, … }` | Weighted distribution over SMT logics |
| `seed` | integer or `null` | RNG seed (`null` = random) |
| `cntsize` | integer | Formula size (number of constraints) |

## Output layout

```
obj/
  run-<timestamp>-<hash>/
    out/          # per-formula Nargo projects (cleaned periodically)
    errors/
      nargo_error/          # nargo execute failures
      bb_prove_error/       # bb prove failures
      bb_verify_error/      # bb verify failures
      acir_oracle_violation/ # SMT ↔ ACIR satisfiability mismatch
    stats.json
```
