# Strict-cap open-loop evaluation

`run_pair.sh` compares Adaptive-Consistency-only with RePLAN under Poisson or
bursty arrivals. Both arms share the same arrival trace, model-matched candidate
trace, FIFO parent activation, fixed base lead, and strict outstanding-branch
cap. Execution order alternates with the arrival seed.

Required environment variables:

```bash
export VLLM_ROOT=/path/to/pinned/vllm
export INPUT_ROOT=/path/to/workload-matrix
export MODEL=/path/to/model
export QUALITY_AMC=/path/to/amc-candidates.jsonl
export QUALITY_AIME=/path/to/aime-candidates.jsonl
export QUALITY_MATH=/path/to/math500-candidates.jsonl
```

Example:

```bash
experiments/open_loop_strict/run_pair.sh \
  0 12 poisson 0.0262 1.0x 1 /path/to/results
```

The positional arguments are GPU index, parents per dataset, arrival mode,
arrival rate, label, seed, and result directory. The reported nominal
concurrency is `REFERENCE_CONCURRENCY=36`, which derives `L0=floor(256/36)=7`.
Set `REFERENCE_CONCURRENCY` or `N_MAX` to override these reported defaults.
