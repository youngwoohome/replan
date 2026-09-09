# Method

## Overview

For one Best-of-N request, RePLAN treats the user request as a **parent** and
each candidate generation as a **branch**. It changes only parent-level branch
admission:

```text
candidate completions
        |
        v
causal confidence stopping
        |
        v
persistent-tail detection at ceil(N/2)
        |
        v
equal refill of returned branch slots
        |
        v
stock vLLM token scheduler
```

## 1. Causal confidence stopping

Each parent has a fixed candidate order. Candidate `j` becomes stopping
evidence only after candidates `0..j-1` have completed. This avoids making
short candidates statistically more important merely because they finish
first.

Let `v1` and `v2` be the two largest answer counts in the causal prefix.
RePLAN preserves the Adaptive-Consistency rule:

```text
P(Beta(v1 + 1, v2 + 1) > 0.5) > 0.95.
```

The rule is first evaluated after four sampling-ordered candidates. The parent
stops at the first confident prefix thereafter or at `N`.

## 2. Persistent-tail detection

For `P` simultaneously submitted parents and configured outstanding-branch
capacity `B`, every parent starts with the same base speculative lead:

```text
l0 = min(N, floor(B / P)).
```

For a fixed batch, `P` is the submitted parent count. For open-loop serving it
is the nominal reference concurrency fixed before arrivals begin and supplied
through `--reference-concurrency`. This is the largest equal initial allocation
that fits the admission bound. A
lead bounds `launched - evidence`: candidates already submitted but not yet
incorporated into the sampling-ordered stopping state. A completed higher-index
candidate waiting behind a missing lower index still occupies lead allowance.
A parent becomes eligible for extra capacity only if it remains unresolved after
the causal midpoint:

```text
tail(i) = [evidence(i) >= ceil(N / 2)].
```

This uses no future answer, future output length, dataset label, or offline
training.

### Why the boundary is N/2

Suppose refill eligibility begins after `k` candidates. There are two opposing
risks:

- **Too early:** up to `N-k` remaining candidates may receive unnecessary
  additional parallelism.
- **Too late:** a genuinely persistent parent waits through the first `k`
  candidates before receiving additional parallelism.

Sibling candidates use the same prompt, model, and sampling configuration, so
candidate index does not systematically change the expected candidate token
cost. Let that expected cost be `mu`. Without a calibrated prior that tells us
which error is more likely, the robust single-boundary rule minimizes the
larger candidate-token exposure:

```text
k* = argmin_k max(k * mu, (N - k) * mu).
```

The maximum is minimized when the two terms are balanced:

```text
k * mu = (N - k) * mu
k* = ceil(N / 2).
```

This is a parameter-free minimax default, not a claim that `N/2` is the
wall-clock optimum for every workload. If early- and late-refill penalties are
assigned different known coefficients, the more general balance is:

```text
k* = N * c_early / (c_early + c_late).
```

Those coefficients depend on an unobserved counterfactual—what would have
happened under the action not taken—and varied online estimators did not
consistently beat the midpoint. The artifact therefore keeps `N/2` as the
model-free method and exposes other boundaries only as sensitivity ablations.

## 3. Equal Tail Refill

Let:

- `A` be the unresolved parents,
- `T` be the persistent tails,
- `B` be the configured outstanding-branch capacity.

RePLAN first reserves the base allocation:

```text
spare = B - l0 * |A|.
```

Every parent in `T` receives the same additional allowance:

```text
extra_per_tail = floor(spare / |T|).
```

The indivisible remainder stays unused, so an arbitrary ordering cannot give
one tail more speculative work than another. Every non-tail retains `l0`, so
it can continue collecting evidence and cannot starve.

The allowance is a maximum evidence-relative speculative lead, not an
instruction to launch all remaining candidates immediately. Branches are
admitted progressively as vLLM capacity becomes available.

For efficiency, all adaptive policies batch admission callbacks until either
the engine becomes idle or at least `ceil(sqrt(B))` configured slots are free.
This batching rule is identical for Adaptive-only, Naive Refill, and RePLAN;
the ablation changes only which parents receive the spare allowance.

## 4. Why tail selection matters

The `naive_refill` ablation gives spare capacity to every unresolved parent.
It shares the exact same stopping rule and execution code as RePLAN. The only
difference is eligibility:

```text
naive_refill: all unresolved parents
RePLAN:     only unresolved parents with evidence >= ceil(N/2)
```

This isolates whether the gain comes from merely filling free slots or from
directing them to persistent tails.

## 5. System-algorithm co-design

The algorithm layer supplies causal answer confidence and decides when a
parent no longer needs candidates. The system layer reallocates the resulting
branch capacity among parents that remain sampling-critical. Adaptive sampling
alone has no returned-capacity policy; a conventional serving scheduler lacks
the answer evidence required to make this parent-level decision.

## 6. Quality contract

With frozen candidate realizations, scheduling changes only launch timing and
concurrency. It preserves candidate order, confidence rule, stopping prefix,
selected answer, and maximum `N`. Live generation is evaluated with repeated
accuracy and quality-latency measurements rather than assumed to be bitwise
identical across schedules.
