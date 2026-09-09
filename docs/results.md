# Results

## Evaluation contract

- Each serving arm uses one GPU.
- The baseline and method use the same model-matched candidate trace.
- BCT is the time until every parent request completes.
- Parent E2E is arrival-to-final-answer latency.
- p95 parent E2E uses linear interpolation at position `0.95 * (P - 1)` in
  the sorted parent latencies.
- Deadline-qualified goodput is the offered parent rate multiplied by the
  fraction completing within an arrival-relative latency deadline.
- Answer accuracy is reported separately from deadline qualification.
- `adaptive`, `naive_refill`, and `tail_refill` share one serving code path.

## Essential eligibility ablation

The following Qwen2.5-7B-Instruct, `N=128` results average three
frozen-candidate serving runs. Percentages are relative to
Adaptive-Consistency stopping without refill.

| Dataset | Policy | Mean BCT | BCT change | Mean parent E2E | E2E change |
|---|---|---:|---:|---:|---:|
| AMC (40 parents) | Naive refill | 769.1 s | -46.5% | 281.4 s | -20.9% |
| AMC (40 parents) | **RePLAN** | **688.7 s** | **-52.1%** | **219.9 s** | **-38.2%** |
| AIME (30 parents) | Naive refill | 2137.9 s | -7.2% | 1353.8 s | **+15.8%** |
| AIME (30 parents) | **RePLAN** | **1595.1 s** | **-30.7%** | **874.9 s** | **-25.2%** |

Accuracy is unchanged within every three-arm dataset comparison: `25/40` for
AMC and `4/30` for AIME. Naive refill can improve BCT by creating more
parallel work, but on AIME it makes the average request slower. Selecting only
persistent tails is therefore materially different from filling every spare
slot.

## Cross-model validation

The same `ceil(N/2)` Equal Tail Refill policy was run without model-specific
tuning. Every row contains three Adaptive and three RePLAN repetitions.

| Model / dataset | N | BCT change | Mean E2E change | Repetitions |
|---|---:|---:|---:|---|
| Ministral-3-8B-Reasoning-2512 / AIME | 64 | -14.9% | -18.1% | 3 + 3 |
| Ministral-3-8B-Reasoning-2512 / AIME | 128 | -26.1% | -24.3% | 3 + 3 |
| Qwen3.5-9B / AIME | 64 | -7.3% | -9.1% | 3 + 3 |
| Qwen3.5-9B / AIME | 128 | -18.2% | -15.7% | 3 + 3 |

## Open-loop serving

The mixed `N=128` workload was released under repeated Poisson and bursty
arrivals. Values are relative to Adaptive-only and average three paired arrival
seeds.

| Arrival process | BCT change | Mean E2E change | p95 E2E change |
|---|---:|---:|---:|
| Poisson 0.8x | -14.6% | -48.0% | -51.0% |
| Poisson 1.0x | -19.1% | -48.0% | -51.8% |
| Poisson 1.2x | -22.5% | -48.5% | -51.9% |
| Bursty 1.0x | -19.9% | -41.9% | -48.2% |

Within 600 seconds of each parent's arrival, RePLAN completes all 36 parents
on average in every arrival condition; Adaptive-only completes 29.0--31.0.
The corresponding deadline-qualified goodput improvement is 16.1--24.1%.

## Midpoint sensitivity

The analytical midpoint is the fixed, model-free method. The implementation
also exposes the activation boundary only for sensitivity analysis:

```text
--tail-activation-evidence 32   # N/4 when N=128
--tail-activation-evidence 64   # N/2
--tail-activation-evidence 96   # 3N/4
```

A coarse mixed-workload `N=128` study produced:

| Boundary | Mean BCT | Mean parent E2E |
|---|---:|---:|
| N/4 | 829.9 s | 333.2 s |
| **N/2** | **778.7 s** | **295.4 s** |
| 3N/4 | 990.2 s | 353.7 s |

This validates the midpoint as a robust default in the evaluated mixed
workload; it is not presented as a universal per-workload wall-clock optimum.

## Allocation checks

The posterior-weighted variant retains the midpoint gate but prioritises each
tail's next branches using their posterior probability of being needed before
stopping. The table reports its BCT relative to equal RePLAN in the original
frozen-replay matrix; positive values mean posterior weighting is slower.

| Dataset | N=32 | N=64 | N=128 |
|---|---:|---:|---:|
| AMC | +0.10% | +0.01% | -0.13% |
| AIME | +1.13% | +0.77% | -2.53% |
| MATH500 | +0.29% | -0.05% | +0.13% |

The sign changes across datasets and budgets, so equal allocation remains the
main policy. A separate mixed `N=128` check assigned the final integer
remainder using cumulative allocation deficits. Relative to floor division,
exact rounding changed BCT, mean E2E, and p95 E2E by +0.12%, +0.11%, and
+0.12%; only 16 remainder slots were assigned across changed allocation states.

## Runtime KV diagnostics

The three-run strong-baseline logs include vLLM runtime reports at roughly
ten-second intervals. Means are first computed per run and then across runs.

| Policy | Mean KV use | Intervals at >=99% KV | Waiting requests | Generation tok./s |
|---|---:|---:|---:|---:|
| Adaptive-only | 46.3% | 5.6% | 2.9 | 1,112 |
| Remaining-budget refill | 94.5% | 76.4% | 54.3 | 1,501 |
| Largest-evidence refill | 84.2% | 45.7% | 35.4 | 1,864 |
| **RePLAN** | **73.4%** | **21.6%** | **14.8** | **1,974** |

These counters are diagnostic only: RePLAN does not observe KV usage, and
the periodic logger does not expose exact preemption or recomputed-token
counts. The aggregate can be regenerated from the serving logs with
`experiments/analysis/summarize_vllm_runtime.py`.
