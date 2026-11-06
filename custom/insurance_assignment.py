"""Validation for explicitly configured computer insurance assignments.

The frozen paper protocol leaves the 25F allocation probabilities unresolved.
There is deliberately no default distribution in this implementation.
"""

import math
from collections.abc import Sequence

from protocol.paper_spec import Treatment


ASSIGNMENT_OUTCOMES = ("both_insured", "first_only", "second_only")


def validate_assignment_configuration(
    treatment: Treatment,
    probabilities: Sequence[float] | None,
    assignment_seed: int | None = None,
) -> tuple[float, float, float] | None:
    """Validate a joint distribution without guessing or normalizing weights."""

    if treatment is not Treatment.FORCED_INSURANCE_25:
        if probabilities is not None or assignment_seed is not None:
            raise ValueError("Forced assignment configuration is only allowed for 25F.")
        return None
    if probabilities is None:
        raise ValueError(
            "25F assignment probabilities remain unresolved in paper_spec. "
            "Explicitly supply forced_assignment_probabilities: "
            "P(both insured), P(first only), P(second only); no default is assumed."
        )
    if (
        not isinstance(probabilities, Sequence)
        or isinstance(probabilities, (str, bytes))
        or len(probabilities) != 3
    ):
        raise ValueError("25F assignment probabilities must contain exactly three numbers.")
    for probability in probabilities:
        if (
            type(probability) not in (int, float)
            or not 0 <= probability <= 1
            or not math.isfinite(probability)
        ):
            raise ValueError("25F assignment probabilities must be finite numbers in [0, 1].")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("25F assignment probabilities must sum to 1; weights are not normalized.")
    if assignment_seed is not None and type(assignment_seed) is not int:
        raise ValueError("assignment_seed must be an integer.")
    return tuple(float(value) for value in probabilities)
