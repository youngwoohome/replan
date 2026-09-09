# RePLAN

Reference implementation of **RePLAN: Reallocating Released Parallelism for
Large-N Adaptive Best-of-N Serving**.

Adaptive sampling decides when a request has enough evidence to stop. RePLAN
coordinates the branch capacity released by those stopping decisions. Every
active request starts with the same speculative lead. Capacity released by
completed requests is then divided among requests that remain unresolved after
half of their maximum sampling budget.

RePLAN changes only parent-level branch admission. Stock vLLM remains
responsible for token batching, KV-cache allocation, preemption, and GPU
execution. It requires no learned difficulty predictor, output-length model,
dataset label, or modification to vLLM.

## Policies

All evaluated policies use the same serving implementation and stopping rule.

| Policy | Description |
|---|---|
| `adaptive` | Adaptive-Consistency stopping without capacity reallocation |
| `naive_refill` | Reallocates capacity across every unresolved request |
| `tail_refill` | RePLAN: reallocates capacity only across persistent tails |
| `largest_evidence_refill` | Progress-aware reviewer baseline |
| `remaining_budget_refill` | Remaining-budget reviewer baseline |
| `confidence_weighted_refill` | Appendix: posterior marginal-need allocation |
| `exact_tail_refill` | Appendix: cumulative-fair assignment of the remainder |

The default RePLAN boundary is `ceil(N/2)`. Alternative boundaries are exposed
only for sensitivity analysis through `--tail-activation-evidence`.

The reported stopping rule is Beta-based Adaptive-Consistency with confidence
threshold `0.95`, evaluated after at least four sampling-ordered candidates.
The per-parent speculative lead bounds `launched - evidence`, so a completed
higher-index candidate remains within the lead until missing lower indices have
completed. Admissions are coalesced with the common quantum `ceil(sqrt(B))`.

## Repository structure

```text
replan/                       CLI, policy, runtime, replay, reporting, trace collection
tests/                        deterministic unit tests
experiments/
  reproduce/                  all fixed-cohort experiments reported in thesis
  open_loop_strict/           Poisson and bursty online arrivals
  model_generalization/       model-matched trace collection and paired runs
  figures/                    regenerate the quantitative results figures
  analysis/                   stopping-depth and runtime diagnostics
artifact/runtime-lock.json    evaluated software configuration
docs/method.md                policy definition and midpoint model
docs/results.md               result summary
```

Generated model outputs and raw runtime logs are intentionally excluded because
they are large. The repository retains the scripts needed to reproduce and
analyse the reported experiments.

## Environment

The reported experiments use one GPU per serving arm and stock vLLM `v0.22.0`
at commit `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`. Exact versions are recorded
in `artifact/runtime-lock.json`.

Install this package and the test dependencies:

```bash
python3 -m pip install -e '.[dev]'
```

Install the pinned vLLM environment separately. The exact package versions and
vLLM commit used for the reported experiments are recorded in
`artifact/runtime-lock.json`. Unit tests do not require a GPU or vLLM.

## Running RePLAN

The serving command consumes a workload directory and a model-matched frozen
candidate trace. Paths can be local directories or shared storage locations;
no cluster path is assumed by the core runner.

```bash
replan-serve \
  --input-root /path/to/workload-matrix \
  --dataset math500 \
  --parents-per-dataset 32 \
  --n-max 128 \
  --policy tail_refill \
  --confidence-rule beta \
  --quality-trace math500=/path/to/candidates.jsonl \
  --model /path/to/model \
  --max-outstanding-branches 256 \
  --gpu-memory-utilization 0.78 \
  --enable-prefix-caching \
  --enforce-eager \
  --output result.json
```

For the primary eligibility ablation, change only `--policy` among `adaptive`,
`naive_refill`, and `tail_refill`. See `experiments/README.md` for the canonical
multi-run commands and required environment variables.

Open-loop serving must pass `--reference-concurrency`, fixed before the run.
The reported online matrix uses `P_base=36`, deriving `L0=floor(256/36)=7`.

## Verification

```bash
python3 -m pytest -q
ruff check .
python3 -m compileall -q replan
```
