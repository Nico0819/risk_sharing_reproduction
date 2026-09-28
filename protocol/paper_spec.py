"""Declarative experimental protocol from Section 2 and Appendix C.

This module defines paper rules only. It contains no environment, agent,
random-number, or LLM implementation.
"""

from enum import Enum


class Treatment(str, Enum):
    """The three treatments in the paper."""

    BASELINE = "baseline"
    VOLUNTARY_INSURANCE_25 = "25V"
    FORCED_INSURANCE_25 = "25F"


# Experimental sample sizes.
TREATMENT_SUBJECT_COUNTS = {
    Treatment.BASELINE: 32,
    Treatment.VOLUNTARY_INSURANCE_25: 36,
    Treatment.FORCED_INSURANCE_25: 40,
}
TREATMENT_PAIR_COUNTS = {
    Treatment.BASELINE: 16,
    Treatment.VOLUNTARY_INSURANCE_25: 18,
    Treatment.FORCED_INSURANCE_25: 20,
}
TOTAL_SUBJECT_COUNT = 108
TOTAL_PAIR_COUNT = 54


# Matching and information structure.
PAIR_SIZE = 2
PAIRING_IS_RANDOM = True
PAIRING_REMAINS_FIXED = True
DIRECT_COMMUNICATION_ALLOWED = False

# Each visibility constant describes what can be seen within the fixed pair.
# No information from another pair is available.
CURRENT_RANDOM_INCOME_VISIBILITY = "fixed_pair_members"
RANDOM_INCOME_HISTORY_VISIBILITY = "fixed_pair_members"
TRANSFER_HISTORY_VISIBILITY = "fixed_pair_members"
INSURANCE_STATUS_VISIBILITY = "fixed_pair_members"
INSURANCE_HISTORY_VISIBILITY = "fixed_pair_members"
OTHER_PAIRS_INFORMATION_IS_VISIBLE = False


# Analysis phases.
PHASE_1_START = 1
PHASE_1_END = 15
PHASE_2_START = 16
PHASE_2_END = 30
PHASE_3_START = 31
PHASE_3_END = 45
ANALYSIS_LAST_PERIOD = PHASE_3_END
PHASES = {
    1: (PHASE_1_START, PHASE_1_END),
    2: (PHASE_2_START, PHASE_2_END),
    3: (PHASE_3_START, PHASE_3_END),
}


# Per-period income design.
FIXED_INCOME = 125
HIGH_RANDOM_INCOME = 200
LOW_RANDOM_INCOME = 0
HIGH_INCOME_MEMBER_PROBABILITY = 0.5
EXACTLY_ONE_MEMBER_RECEIVES_HIGH_RANDOM_INCOME_PER_PERIOD = True
OTHER_MEMBER_RECEIVES_LOW_RANDOM_INCOME_PER_PERIOD = True
PAIR_RANDOM_INCOME_SAMPLING_IS_INDEPENDENT = False


# Insurance contract and timing.
INSURANCE_COST = 25
INSURANCE_PAYOUT_LOW_RANDOM_INCOME = 50
INSURANCE_PAYOUT_HIGH_RANDOM_INCOME = 0
INSURANCE_COVERAGE_FRACTION = 0.25
INSURANCE_OPPORTUNITY_REVEALED_PERIOD = 16
INSURANCE_DECISION_PERIOD = 16
INSURANCE_DECISION_COUNT = 1
INSURANCE_DECISION_IS_ONCE_AND_FOR_ALL = True
INSURANCE_ACTIVE_START_PERIOD = 16
INSURANCE_ACTIVE_END_PERIOD = 30
INSURANCE_REMOVAL_REVEALED_PERIOD = 31

# In 25V, both members submit their period-16 insurance decisions without
# seeing the partner's current decision. The pair's insurance statuses become
# visible only after both submissions are complete.
TREATMENT_25V_INSURANCE_DECISIONS_ARE_SIMULTANEOUS = True
TREATMENT_25V_INSURANCE_SUBMISSIONS_ARE_INDEPENDENT = True
TREATMENT_25V_PARTNER_DECISION_VISIBLE_BEFORE_OWN_SUBMISSION = False
TREATMENT_25V_INSURANCE_STATUS_VISIBLE_AFTER_BOTH_SUBMISSIONS = True


# Transfer institution.
MIN_TRANSFER = 0
HIGH_INCOME_MEMBER_CAN_TRANSFER = True
LOW_INCOME_MEMBER_CAN_TRANSFER = False
TRANSFER_MAXIMUM_IS_SENDER_CURRENT_INCOME = True


# The participant-facing description and the backend realization are kept
# separate. Participants are told only the qualitative horizon information.
PARTICIPANT_MINIMUM_PERIOD_COUNT = 45
PARTICIPANT_IS_TOLD_ENDING_IS_STOCHASTIC = True
PARTICIPANT_IS_TOLD_EXACT_ENDING_PROBABILITIES = False
PARTICIPANT_IS_TOLD_MAXIMUM_PERIOD_COUNT = False

# Actual experimental termination mechanism. The first 47 rounds cannot end;
# round 48 is the first possible ending round; each of rounds 48 and 49 has a
# 78% ending probability; round 50 is forced.
REAL_TERMINATION_PERIOD_1_47_ALLOWED = False
REAL_TERMINATION_FIRST_POSSIBLE_PERIOD = 48
REAL_ENDING_PROBABILITY_PERIOD_1_47 = 0.0
REAL_ENDING_PROBABILITY_PERIOD_48_49 = 0.78
REAL_TERMINATION_FORCED_PERIOD = 50
REAL_ENDING_PROBABILITY_PERIOD_50 = 1.0
REAL_MAXIMUM_PERIOD_COUNT = 50


# Final-payment mechanism: one realized period is selected at random.
PAYMENT_SELECTED_PERIOD_COUNT = 1
PAYMENT_PERIOD_SELECTION_IS_RANDOM = True


# 25F assignment restrictions and unresolved probability.
# The paper states that assignments are computer-generated/random, but this
# specification does not silently choose an assignment probability. The
# paper's design excludes the outcome in which both members remain uninsured.
F25_INSURANCE_ASSIGNMENT_PROBABILITY = None
F25_BOTH_UNINSURED_ALLOWED = False
# TODO: resolve the 25F assignment probability from the original instructions
# or author data before implementing assignment draws.
