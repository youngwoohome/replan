"""Pure policy for progressive, quality-aware Best-of-N allocation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import cache

from replan.answers import normalize_numeric_answer


@cache
def _beta_probability_from_counts(leader: int, runner_up: int) -> float:
    alpha = leader + 1
    beta = runner_up + 1
    trials = alpha + beta - 1
    numerator = sum(math.comb(trials, value) for value in range(alpha))
    return numerator / (2**trials)


@cache
def _marginal_need_curve(
    leader: int,
    runner_up: int,
    evidence: int,
    min_evidence: int,
    n_max: int,
    confidence_threshold: float,
) -> tuple[float, ...]:
    """Probability that each later candidate is needed before stopping."""

    states: dict[tuple[int, int], float] = {(leader, runner_up): 1.0}
    probabilities: list[float] = []
    remaining = max(0, n_max - evidence)
    for step in range(1, remaining + 1):
        probabilities.append(min(1.0, max(0.0, sum(states.values()))))
        next_states: dict[tuple[int, int], float] = {}
        for (left_count, right_count), mass in states.items():
            denominator = left_count + right_count + 2
            transitions = (
                (left_count + 1, right_count, (left_count + 1) / denominator),
                (left_count, right_count + 1, (right_count + 1) / denominator),
            )
            for left, right, probability in transitions:
                ordered = tuple(sorted((left, right), reverse=True))
                observed = evidence + step
                if observed >= min_evidence:
                    confidence = _beta_probability_from_counts(*ordered)
                    if confidence > confidence_threshold:
                        continue
                next_states[ordered] = (
                    next_states.get(ordered, 0.0) + mass * probability
                )
        states = next_states
        if not states:
            probabilities.extend([0.0] * (remaining - step))
            break
    return tuple(probabilities)


def modal_answer(answers: list[str | None]) -> tuple[str | None, int]:
    """Return a deterministic modal answer and its count."""

    normalized = [normalize_numeric_answer(answer) for answer in answers]
    counts = Counter(answer for answer in normalized if answer is not None)
    if not counts:
        return None, 0
    leader = min(counts, key=lambda answer: (-counts[answer], answer))
    return leader, counts[leader]


def beta_leader_probability(answers: list[str | None]) -> float:
    """Adaptive-Consistency probability that the leader beats the runner-up.

    The paper uses a Beta(v1 + 1, v2 + 1) posterior with a uniform prior,
    where v1 and v2 are the two largest answer counts. For integer parameters,
    the probability above 0.5 is an exact binomial tail.
    """

    normalized = [normalize_numeric_answer(answer) for answer in answers]
    counts = sorted(
        Counter(answer for answer in normalized if answer is not None).values(),
        reverse=True,
    )
    if not counts:
        return 0.0
    leader_count = counts[0]
    runner_up_count = counts[1] if len(counts) > 1 else 0
    return _beta_probability_from_counts(leader_count, runner_up_count)


@dataclass(frozen=True)
class AllocationDecision:
    stop: bool
    next_branches: int
    selected_answer: str | None
    consensus_lower_bound: float


class StreamingParent:
    """Online evidence tracker that permits asynchronous sibling launches."""

    def __init__(
        self,
        n_max: int,
        confidence_rule: str = "beta",
        max_speculative_lead: int | None = None,
        evidence_order: str = "sampling",
    ):
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        if confidence_rule != "beta":
            raise ValueError("only the paper's beta confidence rule is supported")
        if max_speculative_lead is not None and max_speculative_lead <= 0:
            raise ValueError("speculative lead must be positive")
        if evidence_order not in {"sampling", "completion"}:
            raise ValueError("evidence order must be sampling or completion")
        self.n_max = n_max
        self.min_evidence = min(4, n_max)
        self.confidence_rule = confidence_rule
        self.confidence_threshold = 0.95
        self.max_speculative_lead = max_speculative_lead
        self.evidence_order = evidence_order
        self.launched = 0
        self.completed_candidates: dict[int, str | None] = {}
        self.completion_sequence: list[int] = []

    @property
    def completed(self) -> int:
        return len(self.completed_candidates)

    @property
    def evidence(self) -> int:
        """Number of candidates usable under the configured evidence order."""

        return len(self._evidence_answers())

    def marginal_need_curve(self) -> tuple[float, ...]:
        """Return posterior need probabilities after the causal prefix."""

        answers = self._evidence_answers()
        counts = sorted(
            Counter(answer for answer in answers if answer is not None).values(),
            reverse=True,
        )
        leader = counts[0] if counts else 0
        runner_up = counts[1] if len(counts) > 1 else 0
        return _marginal_need_curve(
            leader,
            runner_up,
            len(answers),
            self.min_evidence,
            self.n_max,
            self.confidence_threshold,
        )

    def record(self, candidate_index: int, answer: str | None) -> None:
        if self.completed >= self.launched:
            raise RuntimeError("cannot record an unlaunched branch")
        if candidate_index < 0 or candidate_index >= self.launched:
            raise ValueError("candidate index was not launched")
        if candidate_index in self.completed_candidates:
            raise ValueError("candidate was already recorded")
        self.completed_candidates[candidate_index] = normalize_numeric_answer(answer)
        self.completion_sequence.append(candidate_index)

    def _unbiased_prefix_answers(self) -> list[str | None]:
        answers: list[str | None] = []
        for candidate_index in range(self.launched):
            if candidate_index not in self.completed_candidates:
                break
            answers.append(self.completed_candidates[candidate_index])
        return answers

    def _evidence_answers(self) -> list[str | None]:
        if self.evidence_order == "sampling":
            return self._unbiased_prefix_answers()
        return [
            self.completed_candidates[candidate_index]
            for candidate_index in self.completion_sequence
        ]

    def _confidence(self, answers: list[str | None]) -> tuple[str | None, float]:
        leader, _ = modal_answer(answers)
        if leader is None:
            return None, 0.0
        return leader, beta_leader_probability(answers)

    def decide(self) -> AllocationDecision:
        prefix_answers = self._evidence_answers()
        evidence = len(prefix_answers)
        leader: str | None = None
        confidence = 0.0
        for prefix_size in range(self.min_evidence, evidence + 1):
            prefix_leader, prefix_confidence = self._confidence(
                prefix_answers[:prefix_size]
            )
            if prefix_confidence > self.confidence_threshold:
                return AllocationDecision(
                    True,
                    0,
                    prefix_leader,
                    prefix_confidence,
                )
        if prefix_answers:
            leader, confidence = self._confidence(prefix_answers)
        exhausted = self.completed >= self.n_max
        if exhausted:
            return AllocationDecision(True, 0, leader, confidence)

        # The initial fair share is seeded by the serving layer. Do not give a
        # parent additional siblings before it has enough completed evidence
        # to establish whether more sampling is useful.
        within_speculation_bound = (
            self.max_speculative_lead is None
            or self.launched - evidence < self.max_speculative_lead
        )
        can_launch = (
            evidence >= self.min_evidence
            and self.launched < self.n_max
            and within_speculation_bound
        )
        return AllocationDecision(
            False,
            int(can_launch),
            leader,
            confidence,
        )

    def commit_launch(self, branches: int = 1) -> None:
        if branches <= 0:
            raise ValueError("branches must be positive")
        if self.launched + branches > self.n_max:
            raise ValueError("launch exceeds n_max")
        self.launched += branches

    def set_max_speculative_lead(self, branches: int) -> None:
        if branches <= 0:
            raise ValueError("speculative lead must be positive")
        self.max_speculative_lead = min(branches, self.n_max)


__all__ = [
    "AllocationDecision",
    "StreamingParent",
    "beta_leader_probability",
    "modal_answer",
]
