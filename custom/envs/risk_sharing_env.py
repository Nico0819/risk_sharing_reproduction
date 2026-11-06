"""Two-person income-sharing environment for a controlled simulation.

The experimental constants live in :mod:`protocol.paper_spec`.  This module
implements the mechanical transfer and insurance rules. Economic
choices belong to participants, not to the environment or its Router adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Hashable, Iterator, Mapping, Sequence

from agentsociety2 import EnvBase
from agentsociety2.env import ReActRouter, tool

from custom.insurance_assignment import ASSIGNMENT_OUTCOMES, validate_assignment_configuration
from protocol import paper_spec as spec


AgentId = Hashable


PARTICIPANT_DECISION_PREFIX = "RISK_SHARING_PARTICIPANT_DECISION:"


def parse_participant_decision(
    instruction: str, variables: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Read an explicit participant choice, without inferring missing values.

    Accept labelled prose, the previous JSON envelope, and explicit values in
    the framework's legacy ``variables`` argument. All values must come from
    the participant request; neither observations nor Router output are inputs.
    """

    if not isinstance(instruction, str):
        return None
    fields = {"decision_type", "decision_id", "amount", "purchase"}
    if variables is not None and not isinstance(variables, Mapping):
        raise ValueError("participant variables must be an object")
    supplied = {key: value for key, value in (variables or {}).items() if key in fields}
    instruction = instruction.strip()
    if instruction.startswith(PARTICIPANT_DECISION_PREFIX):
        raw = instruction[len(PARTICIPANT_DECISION_PREFIX) :].strip()

        def unique_fields(pairs):
            payload = {}
            for name, value in pairs:
                if name in payload:
                    raise ValueError("participant decision contains duplicate or ambiguous fields")
                payload[name] = value
            return payload

        try:
            payload = json.loads(raw, object_pairs_hook=unique_fields)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("participant decision envelope is not valid JSON") from exc
    else:
        labels = list(re.finditer(
            r"(?<!\w)(decision_type|decision_id|amount|purchase)[ \t]*[:=][ \t]*",
            instruction,
        ))
        if not labels and not supplied:
            return None
        if len({match[1] for match in labels}) != len(labels):
            raise ValueError("participant decision contains duplicate or ambiguous fields")
        payload = {}
        for index, match in enumerate(labels):
            end = labels[index + 1].start() if index + 1 < len(labels) else len(instruction)
            value = re.split(r"[\r\n]", instruction[match.end():end], maxsplit=1)[0]
            payload[match[1]] = value.strip(" \t;")
        for name, value in payload.items():
            if value == "{" + name + "}":
                if name not in supplied:
                    raise ValueError(f"participant {name} has an unresolved placeholder")
                payload[name] = supplied[name]
            elif name == "amount":
                # Only literal prose digits are parsed. Typed variables must
                # already be integers; never coerce floats, strings or bools.
                if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
                    raise ValueError("participant amount must be a strict integer")
                payload[name] = int(value)
            elif name == "purchase":
                if value not in {"true", "false"}:
                    raise ValueError("participant purchase must be a strict boolean (true or false)")
                payload[name] = value == "true"
    if not isinstance(payload, dict):
        raise ValueError("participant decision envelope must be a JSON object")
    for name, value in supplied.items():
        if name in payload and (type(payload[name]) is not type(value) or payload[name] != value):
            raise ValueError(f"participant {name} conflicts with variables")
        payload[name] = value
    value_field = "purchase" if payload.get("decision_type") == "insurance" else "amount"
    required = {"decision_type", "decision_id", value_field}
    if required.difference(payload):
        raise ValueError(
            f"participant decision must contain decision_type, decision_id, and {value_field}"
        )
    if set(payload).difference(required):
        raise ValueError("participant decision contains unexpected fields")
    if not isinstance(payload["decision_type"], str):
        raise ValueError("participant decision_type must be a string")
    if not isinstance(payload["decision_id"], str):
        raise ValueError("participant decision_id must be a string")
    if value_field == "amount" and type(payload["amount"]) is not int:
        raise ValueError("participant amount must be a strict integer")
    if value_field == "purchase" and type(payload["purchase"]) is not bool:
        raise ValueError("participant purchase must be a strict boolean")
    return payload


_PARTICIPANT_HISTORY_FIELDS = (
    "period",
    "agent_id",
    "partner_id",
    "fixed_income",
    "random_income",
    "had_transfer_opportunity",
    "transfer_sent",
    "transfer_received",
    "income_before_transfer",
    "final_income",
)

_INSURANCE_HISTORY_FIELDS = (
    "insurance_purchased", "insured", "insurance_cost", "insurance_payout",
)


class Stage(str, Enum):
    """Participant action stages; these are not researcher analysis phases."""

    INSURANCE_DECISION = "insurance_decision"
    TRANSFER_DECISION = "transfer_decision"
    FINISHED = "finished"


@dataclass
class _PairRound:
    """Private state for one pair in the current simulation period."""

    pair_id: int
    members: tuple[AgentId, AgentId]
    random_income: dict[AgentId, int]
    income_before_transfer: dict[AgentId, int]
    transfer_amount: dict[AgentId, int] = field(default_factory=dict)
    decision_ids: dict[AgentId, str] = field(default_factory=dict)
    settled: bool = False


class RiskSharingEnv(EnvBase):
    """One session containing fixed pairs and the simulation transfer rules.

    ``agent_id`` remains in the tool signature for compatibility with the
    AgentSociety2 tool schema. The project router adapter binds the trusted
    caller before dispatch, and every method checks the argument against it.
    """

    def __init__(
        self,
        participants: Sequence[AgentId],
        treatment: spec.Treatment = spec.Treatment.BASELINE,
        *,
        seed: int | None = None,
        rng: Any | None = None,
        session_id: str = "risk-sharing-session",
        forced_assignment_probabilities: Sequence[float] | None = None,
        assignment_seed: int | None = None,
        assignment_rng: Any | None = None,
    ) -> None:
        super().__init__()

        try:
            normalized_treatment = spec.Treatment(treatment)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown treatment: {treatment!r}") from exc
        assignment_probabilities = validate_assignment_configuration(
            normalized_treatment, forced_assignment_probabilities, assignment_seed,
        )
        if assignment_rng is not None:
            if normalized_treatment is not spec.Treatment.FORCED_INSURANCE_25:
                raise ValueError("assignment_rng is only allowed for 25F.")
            if assignment_seed is not None:
                raise ValueError("Pass either assignment_seed or assignment_rng, not both.")
            if assignment_rng is rng:
                raise ValueError("Income and insurance assignment must use independent RNG objects.")
            if not callable(getattr(assignment_rng, "random", None)):
                raise TypeError("assignment_rng must provide random().")
        if seed is not None and rng is not None:
            raise ValueError("Pass either seed or rng, not both.")
        if rng is not None and not all(
            callable(getattr(rng, method, None)) for method in ("shuffle", "random")
        ):
            raise TypeError("rng must provide shuffle(sequence) and random().")

        participant_list = list(participants)
        if len(participant_list) == 0:
            raise ValueError("At least one pair is required.")
        if len(participant_list) % spec.PAIR_SIZE != 0:
            raise ValueError(
                f"The number of participants must be divisible by {spec.PAIR_SIZE}."
            )
        try:
            if len(set(participant_list)) != len(participant_list):
                raise ValueError("Participant ids must be unique.")
        except TypeError as exc:
            raise TypeError("Participant ids must be hashable.") from exc

        self.session_id = session_id
        self.treatment = normalized_treatment
        self.participants: tuple[AgentId, ...] = tuple(participant_list)
        self._rng = rng if rng is not None else random.Random(seed)

        shuffled = list(self.participants)
        self._rng.shuffle(shuffled)
        self.pair_members: dict[int, tuple[AgentId, AgentId]] = {}
        self.pair_by_agent: dict[AgentId, int] = {}
        self.partner_by_agent: dict[AgentId, AgentId] = {}
        for offset in range(0, len(shuffled), spec.PAIR_SIZE):
            pair_id = offset // spec.PAIR_SIZE + 1
            members = (shuffled[offset], shuffled[offset + 1])
            self.pair_members[pair_id] = members
            for agent_id, partner_id in ((members[0], members[1]), (members[1], members[0])):
                self.pair_by_agent[agent_id] = pair_id
                self.partner_by_agent[agent_id] = partner_id

        self.current_period = 1
        self.stage = Stage.TRANSFER_DECISION
        self.finished = False
        self.history: list[dict[str, Any]] = []
        self.pair_history: dict[int, list[dict[str, Any]]] = {
            pair_id: [] for pair_id in self.pair_members
        }
        self.completed_periods: list[int] = []
        self.ending_outcomes: dict[int, bool] = {}
        self._ending_drawn_periods: set[int] = set()
        self._decision_receipts: dict[str, tuple[AgentId, int, str, int | bool]] = {}
        self._insurance_choices: dict[AgentId, bool] = {}
        self._forced_insurance: dict[AgentId, bool] = {}
        self._insurance_assignment_records: list[dict[str, Any]] = []
        self._assignment_configuration: dict[str, Any] | None = None
        self._assignment_rng = None
        if assignment_probabilities is not None:
            # A separate stream preserves matching, income shocks and ending
            # draws under the original experiment seed across treatments.
            if assignment_rng is None:
                if assignment_seed is None:
                    assignment_seed = (
                        int.from_bytes(hashlib.sha256(
                            f"risk-sharing-25f-assignment-v1:{seed}".encode()
                        ).digest(), "big")
                        if seed is not None else random.SystemRandom().getrandbits(128)
                    )
                self._assignment_rng = random.Random(assignment_seed)
            else:
                self._assignment_rng = assignment_rng
            self._assignment_configuration = {
                "probabilities": dict(zip(ASSIGNMENT_OUTCOMES, assignment_probabilities)),
                "seed": assignment_seed,
                "rng_source": "injected" if assignment_rng is not None else "seeded",
                "probability_source": "explicit_run_configuration",
                "paper_probability_resolved": False,
            }
        # EnvBase registers all decorated tools. Publish only the tools of the
        # current stage; otherwise tool schemas would reveal the future option.
        self._stage_tool_catalog = copy.deepcopy(self._llm_tools)
        self._rounds: dict[int, _PairRound] = {}
        self._trusted_caller: ContextVar[AgentId | None] = ContextVar(
            f"risk_sharing_caller_{id(self)}", default=None
        )
        self._expected_participant_decision: ContextVar[
            tuple[str, str, int | bool] | None
        ] = ContextVar(
            f"risk_sharing_expected_decision_{id(self)}", default=None
        )
        self._participant_request_requires_decision: ContextVar[bool] = ContextVar(
            f"risk_sharing_participant_request_{id(self)}", default=False
        )
        self.participant_decisions: list[dict[str, Any]] = []
        self.participant_rejections: list[dict[str, Any]] = []
        self._prepare_period()

    @staticmethod
    def _analysis_phase_for_period(period: int) -> int | None:
        for phase, (start, end) in spec.PHASES.items():
            if start <= period <= end:
                return phase
        return None

    @property
    def analysis_phase(self) -> int | None:
        """Return the researcher analysis phase for the current period."""

        return self._analysis_phase_for_period(self.current_period)

    @property
    def round_states(self) -> Mapping[int, _PairRound]:
        """Return a detached snapshot of current private round state."""

        return copy.deepcopy(self._rounds)

    def _validate_registered_agent(self, agent_id: AgentId) -> AgentId:
        try:
            known = agent_id in self.pair_by_agent
        except TypeError as exc:
            raise PermissionError("Unknown participant identity.") from exc
        if not known:
            raise PermissionError("The caller is not a participant in this session.")
        return agent_id

    @contextmanager
    def caller_context(self, caller_id: AgentId) -> Iterator[None]:
        """Bind a framework-verified caller for one environment request.

        This is an integration boundary, not a participant-facing tool.
        ``RiskSharingRouterAdapter`` calls it using the identity overlay
        supplied by ``PersonAgent``. Direct tool calls without this binding
        are rejected.
        """

        caller_id = self._validate_registered_agent(caller_id)
        token = self._trusted_caller.set(caller_id)
        try:
            yield
        finally:
            self._trusted_caller.reset(token)

    @contextmanager
    def expected_participant_decision(
        self, decision_type: str, decision_id: str, amount: int | bool
    ) -> Iterator[None]:
        """Lock the exact participant payload accepted for this request."""

        token = self._expected_participant_decision.set(
            (decision_type, decision_id, amount)
        )
        try:
            yield
        finally:
            self._expected_participant_decision.reset(token)

    @contextmanager
    def participant_request_requires_decision(self) -> Iterator[None]:
        """Require this mutation request to carry a participant envelope."""

        token = self._participant_request_requires_decision.set(True)
        try:
            yield
        finally:
            self._participant_request_requires_decision.reset(token)

    def record_participant_decision(
        self,
        agent_id: AgentId,
        decision: Mapping[str, Any],
        *,
        raw_request: str | None = None,
        observation: Mapping[str, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the structured decision emitted by a participant model."""

        self.participant_decisions.append(
            {
                "agent_id": agent_id,
                "decision_source": "participant_model",
                "decision_type": decision["decision_type"],
                "decision_id": decision["decision_id"],
                **{name: decision[name] for name in ("amount", "purchase") if name in decision},
                **({"raw_request": raw_request} if raw_request else {}),
                "participant_variables": {
                    name: copy.deepcopy(value) for name, value in (variables or {}).items()
                    if name in {"decision_type", "decision_id", "amount", "purchase"}
                },
                **(
                    {"participant_observation": copy.deepcopy(dict(observation))}
                    if observation is not None
                    else {}
                ),
            }
        )

    def _validate_agent(self, agent_id: AgentId) -> AgentId:
        bound_agent_id = self._trusted_caller.get()
        if bound_agent_id is None:
            raise PermissionError(
                "No trusted caller is bound; use the project Router adapter "
                "or an explicit caller_context."
            )
        if agent_id != bound_agent_id:
            raise PermissionError(
                "The supplied participant identity does not match the trusted caller."
            )
        return self._validate_registered_agent(bound_agent_id)

    def _prepare_period(self) -> None:
        if self.finished:
            return
        if (
            self.treatment is spec.Treatment.FORCED_INSURANCE_25
            and self.current_period == spec.INSURANCE_DECISION_PERIOD
        ):
            self._assign_forced_insurance()
        if (
            self.treatment is spec.Treatment.VOLUNTARY_INSURANCE_25
            and self.current_period == spec.INSURANCE_DECISION_PERIOD
            and len(self._insurance_choices) != len(self.participants)
        ):
            # Appendix C, Fig. C.4: choose before the income shock. Neither
            # incomes nor ending draws are consumed during this substage.
            self.stage = Stage.INSURANCE_DECISION
            self._rounds = {}
            self._publish_stage_tools()
            return
        self.stage = Stage.TRANSFER_DECISION
        self._publish_stage_tools()
        self._rounds = {}
        for pair_id, members in self.pair_members.items():
            first_is_high = self._rng.random() < spec.HIGH_INCOME_MEMBER_PROBABILITY
            high_member = members[0] if first_is_high else members[1]
            low_member = members[1] if first_is_high else members[0]
            random_income = {
                high_member: spec.HIGH_RANDOM_INCOME,
                low_member: spec.LOW_RANDOM_INCOME,
            }
            income_before_transfer = {
                agent_id: (
                    spec.FIXED_INCOME + amount
                    - (spec.INSURANCE_COST if self._is_insured(agent_id) else 0)
                    + self._insurance_payout(agent_id, amount)
                )
                for agent_id, amount in random_income.items()
            }
            self._rounds[pair_id] = _PairRound(
                pair_id=pair_id,
                members=members,
                random_income=random_income,
                income_before_transfer=income_before_transfer,
            )

    def _publish_stage_tools(self) -> None:
        allowed = {"get_observation"}
        if self.stage is Stage.INSURANCE_DECISION:
            allowed.add("submit_insurance_decision")
        elif self.stage is Stage.TRANSFER_DECISION:
            allowed.add("submit_decision")
        self._set_llm_tools([
            item for item in self._stage_tool_catalog
            if item["function"]["name"] in allowed
        ])

    def _is_insured(self, agent_id: AgentId) -> bool:
        coverage = (
            self._forced_insurance
            if self.treatment is spec.Treatment.FORCED_INSURANCE_25
            else self._insurance_choices
        )
        return (
            spec.INSURANCE_ACTIVE_START_PERIOD <= self.current_period <= spec.INSURANCE_ACTIVE_END_PERIOD
            and coverage.get(agent_id, False)
        )

    @property
    def assignment_configuration(self) -> dict[str, Any] | None:
        """Researcher-only distribution and seed; never part of observations."""

        return copy.deepcopy(self._assignment_configuration)

    @property
    def insurance_assignments(self) -> list[dict[str, Any]]:
        """Detached computer assignment records, separate from model decisions."""

        return copy.deepcopy(self._insurance_assignment_records)

    def _assign_forced_insurance(self) -> None:
        """Assign once before period-16 income, only at the scheduler boundary."""

        if self._forced_insurance:
            return
        assert self._assignment_configuration is not None
        assert self._assignment_rng is not None
        probabilities = self._assignment_configuration["probabilities"]
        possible = [(name, probability) for name, probability in probabilities.items() if probability > 0]
        coverage_by_outcome = {
            "both_insured": (True, True),
            "first_only": (True, False),
            "second_only": (False, True),
        }
        coverage = {}
        records = []
        for pair_id, members in self.pair_members.items():
            draw = self._assignment_rng.random()
            if type(draw) not in (int, float) or not 0 <= draw < 1 or not math.isfinite(draw):
                raise ValueError("Assignment RNG must return a finite number in [0, 1).")
            # The validator tolerates floating-point sums close to one. Any
            # trailing rounding residue belongs to the last positive outcome,
            # never to a combination explicitly assigned probability zero.
            outcome = possible[-1][0]
            cumulative = 0.0
            for candidate, probability in possible:
                cumulative += probability
                if draw < cumulative:
                    outcome = candidate
                    break
            statuses = coverage_by_outcome[outcome]
            for agent_id, insured in zip(members, statuses):
                coverage[agent_id] = insured
                records.append({
                    "agent_id": agent_id, "pair_id": pair_id,
                    "partner_id": self.partner_by_agent[agent_id],
                    "period": self.current_period, "insured": insured,
                    "assignment_source": "environment_random",
                    "assignment_outcome": outcome, "assignment_draw": draw,
                    "pair_members": list(members), "treatment": self.treatment.value,
                })
        self._forced_insurance = coverage
        self._insurance_assignment_records = records

    def _insurance_payout(self, agent_id: AgentId, random_income: int) -> int:
        if not self._is_insured(agent_id):
            return 0
        return (
            spec.INSURANCE_PAYOUT_LOW_RANDOM_INCOME
            if random_income == spec.LOW_RANDOM_INCOME
            else spec.INSURANCE_PAYOUT_HIGH_RANDOM_INCOME
        )

    def _insurance_account(self, agent_id: AgentId, random_income: int) -> dict[str, Any]:
        original_status = (
            {"insurance_assigned": self._forced_insurance.get(agent_id)}
            if self.treatment is spec.Treatment.FORCED_INSURANCE_25
            else {"insurance_purchased": self._insurance_choices.get(agent_id)}
        )
        return {
            **original_status,
            "insured": self._is_insured(agent_id),
            "insurance_cost": spec.INSURANCE_COST if self._is_insured(agent_id) else 0,
            "insurance_payout": self._insurance_payout(agent_id, random_income),
        }

    def _decision_id(
        self, pair_id: int, agent_id: AgentId, decision_type: str = "transfer",
    ) -> str:
        """Return a stable opaque token for this participant's current turn.

        The digest is deterministic and does not consume the experiment RNG.
        It binds the token to the session, period, pair and participant without
        exposing the researcher's session label (which may contain treatment,
        seed, or horizon metadata) to a participant.
        """

        identity = (
            "risk-sharing-decision-v1",
            self.session_id,
            self.current_period,
            pair_id,
            agent_id,
        )
        # Preserve existing transfer tokens; distinguish the two decisions in
        # period 16 without consuming income/termination RNG state.
        if decision_type != "transfer":
            identity += (decision_type,)
        material = repr(identity).encode("utf-8")
        return f"decision-{hashlib.sha256(material).hexdigest()[:32]}"

    def _sample_ending_after_settlement(self) -> bool:
        """Apply the backend ending rule exactly once for this period."""

        period = self.current_period
        if period in self._ending_drawn_periods:
            return self.ending_outcomes[period]

        self._ending_drawn_periods.add(period)
        first_possible = spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD
        forced_period = spec.REAL_TERMINATION_FORCED_PERIOD
        if period < first_possible:
            ended = (
                spec.REAL_TERMINATION_PERIOD_1_47_ALLOWED
                and spec.REAL_ENDING_PROBABILITY_PERIOD_1_47 > 0.0
            )
        elif period < forced_period:
            ended = self._rng.random() < spec.REAL_ENDING_PROBABILITY_PERIOD_48_49
        elif period == forced_period:
            ended = spec.REAL_ENDING_PROBABILITY_PERIOD_50 >= 1.0
        else:
            ended = spec.REAL_ENDING_PROBABILITY_PERIOD_50 >= 1.0
        self.ending_outcomes[period] = ended
        return ended

    def _settle_current_period(self) -> list[dict[str, Any]]:
        if any(not state.transfer_amount for state in self._rounds.values()):
            raise RuntimeError("Cannot settle before every high-income decision is submitted.")

        new_records: list[dict[str, Any]] = []
        for pair_id, state in self._rounds.items():
            if state.settled:
                continue
            high_member = next(
                agent_id
                for agent_id in state.members
                if state.random_income[agent_id] == spec.HIGH_RANDOM_INCOME
            )
            low_member = next(agent_id for agent_id in state.members if agent_id != high_member)
            transfer = state.transfer_amount[high_member]
            final_income = {
                high_member: state.income_before_transfer[high_member] - transfer,
                low_member: state.income_before_transfer[low_member] + transfer,
            }
            transfer_received = {high_member: 0, low_member: transfer}
            for agent_id in state.members:
                partner_id = self.partner_by_agent[agent_id]
                record = {
                    "session_id": self.session_id,
                    "treatment": self.treatment.value,
                    "pair_id": pair_id,
                    "agent_id": agent_id,
                    "partner_id": partner_id,
                    "period": self.current_period,
                    "analysis_phase": self._analysis_phase_for_period(self.current_period),
                    "fixed_income": spec.FIXED_INCOME,
                    "random_income": state.random_income[agent_id],
                    "had_transfer_opportunity": agent_id == high_member,
                    "transfer_sent": transfer if agent_id == high_member else 0,
                    "transfer_received": transfer_received[agent_id],
                    "income_before_transfer": state.income_before_transfer[agent_id],
                    "final_income": final_income[agent_id],
                }
                if self.treatment is not spec.Treatment.BASELINE:
                    record.update(self._insurance_account(agent_id, state.random_income[agent_id]))
                new_records.append(record)
                self.history.append(record)
                self.pair_history[pair_id].append(record)
            state.settled = True

        self.completed_periods.append(self.current_period)
        return new_records

    def _participant_history(self, pair_id: int) -> list[dict[str, Any]]:
        """Return only fields that belong in a participant's pair view."""

        return [
            {
                field: copy.deepcopy(record[field])
                for field in _PARTICIPANT_HISTORY_FIELDS + (
                    (
                        ("insurance_assigned", "insured", "insurance_cost", "insurance_payout")
                        if self.treatment is spec.Treatment.FORCED_INSURANCE_25
                        else _INSURANCE_HISTORY_FIELDS
                    )
                    if self.treatment is not spec.Treatment.BASELINE
                    and record["period"] >= spec.INSURANCE_OPPORTUNITY_REVEALED_PERIOD
                    else ()
                )
            }
            for record in self.pair_history[pair_id]
        ]

    @tool(
        readonly=True,
        name="get_observation",
        description=(
            "Read the caller's current pair observation once. Follow its current task "
            "using the returned decision_id and legal choices; "
            "do not repeat this observation in the same request."
        ),
        kind="observe",
    )
    async def get_observation(self, agent_id: int) -> dict[str, Any]:
        """Return only the current caller's legally visible pair information."""

        agent_id = self._validate_agent(agent_id)
        pair_id = self.pair_by_agent[agent_id]
        partner_id = self.partner_by_agent[agent_id]
        if self.finished:
            return {
                "current_period": self.current_period,
                "stage": self.stage.value,
                "task": "finished",
                "agent_id": agent_id,
                "partner_id": partner_id,
                "history": self._participant_history(pair_id),
            }

        if self.stage is Stage.INSURANCE_DECISION:
            submitted = agent_id in self._insurance_choices
            pair_ready = all(member in self._insurance_choices for member in self.pair_members[pair_id])
            return {
                "current_period": self.current_period,
                "stage": self.stage.value,
                "task": "waiting" if submitted else "insurance",
                "decision_id": self._decision_id(pair_id, agent_id, "insurance"),
                "agent_id": agent_id,
                "partner_id": partner_id,
                "insurance_submitted": submitted,
                "own_purchase": self._insurance_choices.get(agent_id),
                **({"partner_purchase": self._insurance_choices[partner_id]} if pair_ready else {}),
                "legal_insurance_choices": [True, False],
                "insurance_offer": {
                    "cost_per_period": spec.INSURANCE_COST,
                    "payout_if_low_random_income": spec.INSURANCE_PAYOUT_LOW_RANDOM_INCOME,
                    "payout_if_high_random_income": spec.INSURANCE_PAYOUT_HIGH_RANDOM_INCOME,
                    "announcement": (
                        "From this period you may purchase insurance before the income shock. "
                        "You can decide only in this period; your choice will be replicated "
                        "automatically in later periods. Income allocation is independent "
                        "of this choice. Both partners choose separately without seeing "
                        "the other's choice; choices are shown after both are submitted."
                    ),
                },
                "history": self._participant_history(pair_id),
            }

        state = self._rounds[pair_id]
        has_opportunity = state.random_income[agent_id] == spec.HIGH_RANDOM_INCOME
        submitted = agent_id in state.transfer_amount
        max_transfer = state.income_before_transfer[agent_id] if has_opportunity else None
        observation = {
            "current_period": self.current_period,
            "stage": self.stage.value,
            "task": "transfer" if has_opportunity and not submitted else "waiting",
            "decision_id": self._decision_id(pair_id, agent_id) if has_opportunity else None,
            "agent_id": agent_id,
            "partner_id": partner_id,
            "fixed_income": spec.FIXED_INCOME,
            "random_income": state.random_income[agent_id],
            "income_before_transfer": state.income_before_transfer[agent_id],
            "had_transfer_opportunity": has_opportunity,
            "transfer_submitted": submitted,
            "legal_transfer_range": (
                {"min": spec.MIN_TRANSFER, "max": max_transfer}
                if has_opportunity
                else None
            ),
            "partner": {
                "agent_id": partner_id,
                "fixed_income": spec.FIXED_INCOME,
                "random_income": state.random_income[partner_id],
                "income_before_transfer": state.income_before_transfer[partner_id],
            },
            "history": self._participant_history(pair_id),
        }
        if (
            self.treatment is not spec.Treatment.BASELINE
            and self.current_period >= spec.INSURANCE_OPPORTUNITY_REVEALED_PERIOD
        ):
            observation.update(self._insurance_account(agent_id, state.random_income[agent_id]))
            observation["partner"].update(
                self._insurance_account(partner_id, state.random_income[partner_id])
            )
            if self.treatment is spec.Treatment.FORCED_INSURANCE_25:
                observation["insurance_assignment_source"] = "computer_random"
                if self.current_period <= spec.INSURANCE_ACTIVE_END_PERIOD:
                    observation["insurance_terms"] = {
                        "cost_per_period": spec.INSURANCE_COST,
                        "payout_if_low_random_income": spec.INSURANCE_PAYOUT_LOW_RANDOM_INCOME,
                        "payout_if_high_random_income": spec.INSURANCE_PAYOUT_HIGH_RANDOM_INCOME,
                    }
                    observation["announcement"] = (
                        "Insurance status for you and your partner was assigned randomly "
                        "by the computer before the income draw in period 16. Neither "
                        "participant chose this status or can change it. The assignment "
                        "continues automatically in subsequent periods. Each insured "
                        "participant pays 25 per period and receives 50 if their random "
                        "income is 0, and no payout if it is 200. The computer assignment "
                        "does not change income allocation or who can make a transfer."
                    )
            if self.current_period == spec.INSURANCE_REMOVAL_REVEALED_PERIOD:
                observation["announcement"] = (
                    "Insurance is no longer available from this period. No premium is "
                    "charged and no insurance payment is made. Income allocation and "
                    "transfers continue as before the option was introduced."
                )
        return observation

    def _validate_expected_decision(self, decision_type: str, decision_id: str, value: int | bool) -> None:
        expected = self._expected_participant_decision.get()
        if self._participant_request_requires_decision.get() and expected is None:
            raise PermissionError(
                "An environment request cannot create an economic decision without a participant model envelope."
            )
        if expected is not None and (
            (decision_type, decision_id, value) != expected or type(value) is not type(expected[2])
        ):
            raise PermissionError(
                "The environment adapter may only forward the exact decision emitted by the participant model."
            )

    @tool(
        readonly=False,
        name="submit_insurance_decision",
        description="Submit the caller's current one-time choice with purchase=true or purchase=false.",
    )
    async def submit_insurance_decision(
        self, agent_id: int, decision_id: str, decision_type: str, purchase: bool,
    ) -> dict[str, Any]:
        """Save a sealed choice; never allocate income, settle, or advance."""

        agent_id = self._validate_agent(agent_id)
        if not isinstance(decision_id, str) or not isinstance(decision_type, str):
            raise ValueError("decision_id and decision_type must be strings.")
        if type(purchase) is not bool:
            raise ValueError("purchase must be a strict boolean.")
        self._validate_expected_decision(decision_type, decision_id, purchase)
        receipt = self._decision_receipts.get(decision_id)
        expected = (agent_id, self.current_period, decision_type, purchase)
        if receipt is not None:
            if receipt[1] != self.current_period:
                raise ValueError("This decision_id belongs to a previous period and is stale.")
            if receipt != expected or type(receipt[3]) is not bool:
                raise ValueError("This decision_id is already locked to a different decision.")
            return {"status": "already_recorded", "decision_id": decision_id,
                    "period": self.current_period, "purchase": purchase}
        if self.stage is not Stage.INSURANCE_DECISION or self.finished:
            raise ValueError("This decision is not available in the current task.")
        if decision_type != "insurance":
            raise ValueError("decision_type does not match the current task.")
        if decision_id != self._decision_id(self.pair_by_agent[agent_id], agent_id, "insurance"):
            raise ValueError("decision_id is not the current valid decision for this participant.")
        if agent_id in self._insurance_choices:
            raise ValueError("The submitted choice is already locked.")
        self._insurance_choices[agent_id] = purchase
        self._decision_receipts[decision_id] = expected
        return {"status": "submitted", "decision_id": decision_id,
                "period": self.current_period, "purchase": purchase}

    @tool(
        readonly=False,
        name="submit_decision",
        description=(
            "Submit the caller's current transfer decision using the numeric agent_id, "
            "current decision_id, and a legal integer amount from the observation."
        ),
    )
    async def submit_decision(
        self,
        agent_id: int,
        decision_id: str,
        decision_type: str,
        amount: int,
    ) -> dict[str, Any]:
        """Validate and record one current transfer decision."""

        agent_id = self._validate_agent(agent_id)
        if not isinstance(decision_id, str):
            raise ValueError("decision_id must be a string.")
        if not isinstance(decision_type, str):
            raise ValueError("decision_type must be a string.")
        if type(amount) is not int:
            raise ValueError("Transfer amount must be a non-negative integer.")
        self._validate_expected_decision(decision_type, decision_id, amount)
        receipt = self._decision_receipts.get(decision_id)
        if receipt is not None:
            recorded_agent, period, recorded_type, recorded_amount = receipt
            if period != self.current_period:
                raise ValueError("This decision_id belongs to a previous period and is stale.")
            if (
                recorded_agent == agent_id
                and recorded_type == decision_type == "transfer"
                and type(recorded_amount) is int
                and recorded_amount == amount
            ):
                return {
                    "status": "already_recorded",
                    "decision_id": decision_id,
                    "period": period,
                    "amount": recorded_amount,
                }
            raise ValueError("This decision_id is already locked to a different decision.")

        if self.finished:
            raise ValueError("The session has already finished.")
        if self.stage is not Stage.TRANSFER_DECISION:
            raise ValueError(f"Decisions are not accepted in stage {self.stage.value}.")
        pair_id = self.pair_by_agent[agent_id]
        state = self._rounds[pair_id]
        if state.random_income[agent_id] != spec.HIGH_RANDOM_INCOME:
            raise ValueError("Only the high-random-income participant has a transfer opportunity.")
        expected_id = self._decision_id(pair_id, agent_id)
        if decision_id != expected_id:
            raise ValueError("decision_id is not the current valid decision for this participant.")
        if decision_type != "transfer":
            raise ValueError("Only the current transfer decision is accepted.")
        sender_current_income = state.income_before_transfer[agent_id]
        if not spec.MIN_TRANSFER <= amount <= sender_current_income:
            raise ValueError(
                "Transfer amount must be between MIN_TRANSFER and the sender's current income."
            )
        if agent_id in state.transfer_amount:
            previous_amount = state.transfer_amount[agent_id]
            if previous_amount == amount and state.decision_ids[agent_id] == decision_id:
                return {
                    "status": "already_recorded",
                    "decision_id": decision_id,
                    "period": self.current_period,
                    "amount": amount,
                }
            raise ValueError("A participant cannot submit a different second decision this period.")

        state.transfer_amount[agent_id] = amount
        state.decision_ids[agent_id] = decision_id
        self._decision_receipts[decision_id] = (
            agent_id,
            self.current_period,
            decision_type,
            amount,
        )

        return {
            "status": "submitted",
            "decision_id": decision_id,
            "period": self.current_period,
            "amount": amount,
        }

    def _all_required_decisions_submitted(self) -> bool:
        if self.stage is Stage.INSURANCE_DECISION:
            return len(self._insurance_choices) == len(self.participants)
        return all(bool(state.transfer_amount) for state in self._rounds.values())

    async def init(self, start_datetime: datetime) -> None:
        """Set framework time; paper-period state is initialized in the constructor."""

        self.t = start_datetime

    async def step(self, tick: int, t: datetime) -> None:
        """Advance framework time and settle a fully submitted paper period."""

        self.t = t
        if self.finished or not self._all_required_decisions_submitted():
            return

        if self.stage is Stage.INSURANCE_DECISION:
            # A scheduler boundary within period 16, not a settled paper round.
            self._prepare_period()
            return

        self._settle_current_period()
        ended = self._sample_ending_after_settlement()
        if ended:
            self.finished = True
            self.stage = Stage.FINISHED
            self._publish_stage_tools()
            return

        self.current_period += 1
        self._prepare_period()



class RiskSharingReActRouter(ReActRouter):
    """Use the real Router tool loop, returning tool data without a summary LLM."""

    async def ask(self, *args, **kwargs):
        # 2.9.0 caches schemas at construction. Refresh this project Router's
        # cache at the trusted call boundary, without modifying the platform.
        self._all_tools = []
        self._all_readonly_tools = []
        self._tool_name_to_module = {}
        self._tool_name_to_tool_obj = {}
        self._collect_all_tools()
        return await super().ask(*args, **kwargs)

    async def generate_final_answer(
        self, ctx, instruction, results, process_text="", status="success", error=None
    ):
        return json.dumps(results, ensure_ascii=False, default=str), status


class RiskSharingRouterAdapter:
    """Bind trusted identity and an immutable choice around a real Router call.

    AgentSociety2 2.9.0 supplies ``id``, ``agent_id`` and ``person_id`` in the
    Agent-to-Router context. The Router may format and route the request, but
    the environment rejects economic fields differing from the participant.
    """

    def __init__(self, router: Any, environment: RiskSharingEnv) -> None:
        self._router = router
        self._environment = environment

    @property
    def participant_decisions(self) -> list[dict[str, Any]]:
        return self._environment.participant_decisions

    @property
    def participant_rejections(self) -> list[dict[str, Any]]:
        return self._environment.participant_rejections

    def record_participant_rejection(
        self, agent_id: AgentId, instruction: str, error: str,
        *, variables: Mapping[str, Any] | None = None,
    ) -> None:
        self._environment.participant_rejections.append(
            {
                "agent_id": agent_id,
                "pair_id": self._environment.pair_by_agent[agent_id],
                "action": "participant_decision_rejected",
                "decision_source": "participant_model",
                "ok": False,
                "raw_request": instruction,
                "participant_variables": {
                    name: copy.deepcopy(value) for name, value in (variables or {}).items()
                    if name in {"decision_type", "decision_id", "amount", "purchase"}
                },
                "error": error,
                "router_action": "not_forwarded",
                "router_model_called": False,
            }
        )

    def record_participant_decision(
        self,
        agent_id: AgentId,
        decision: Mapping[str, Any],
        *,
        raw_request: str | None = None,
        observation: Mapping[str, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
    ) -> None:
        self._environment.record_participant_decision(
            agent_id,
            decision,
            raw_request=raw_request,
            observation=observation,
            variables=variables,
        )

    @staticmethod
    def _caller_from_context(ctx: Mapping[str, Any]) -> AgentId:
        if not isinstance(ctx, Mapping):
            raise PermissionError("Agent context is missing a trusted caller identity.")
        identities = [ctx.get(key) for key in ("id", "agent_id", "person_id")]
        if any(identity is None for identity in identities):
            raise PermissionError(
                "Agent context must contain id, agent_id, and person_id before an environment call."
            )
        if not (identities[0] == identities[1] == identities[2]):
            raise PermissionError("Agent caller identity fields do not agree.")
        return identities[0]

    @staticmethod
    def _is_observation_request(instruction: str) -> bool:
        return instruction.strip() in {"<observe>", "observe"}

    @staticmethod
    def _result_pair(tool_name: str, result: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        """Return the same result shape that ``PersonAgent.ask_env`` expects."""

        return {tool_name: dict(result)}, json.dumps(
            result, ensure_ascii=False, sort_keys=True
        )

    async def _route(
        self, ctx, tool_name, arguments, *, readonly, instruction,
        template_mode, trace_id, parent_span_id,
    ):
        request = {"tool": tool_name, "arguments": arguments}
        routed_instruction = (
            "Route this request using exactly the supplied arguments. Economic "
            "values were chosen by the participant; never create or change them. "
            "After the tool succeeds, end this request.\n<risk_sharing_rpc>\n"
            + json.dumps(request, ensure_ascii=False)
            + "\n</risk_sharing_rpc>\nParticipant request:\n"
            + instruction
        )
        result, _ = await self._router.ask(
            dict(ctx), routed_instruction, readonly=readonly,
            template_mode=template_mode, trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        value = result.get(tool_name) if isinstance(result, Mapping) else None
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Router did not successfully execute {tool_name}.")
        return value

    async def ask(
        self,
        ctx: dict,
        instruction: str,
        readonly: bool = False,
        template_mode: bool = False,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Any:
        caller_id = self._caller_from_context(ctx)

        participant_decision = parse_participant_decision(instruction, ctx.get("variables"))
        with self._environment.caller_context(caller_id):
            if participant_decision is not None:
                if readonly:
                    raise PermissionError(
                        "A participant decision cannot be submitted in readonly mode."
                    )
                # Bind all participant choices before the Router model runs.
                # Tool execution rejects altered, omitted or fabricated values.
                decision_type = participant_decision["decision_type"]
                decision_id = participant_decision["decision_id"]
                value_field = "purchase" if decision_type == "insurance" else "amount"
                value = participant_decision[value_field]
                tool_name = "submit_insurance_decision" if decision_type == "insurance" else "submit_decision"
                available = {tool["function"]["name"] for tool in self._environment._llm_tools}
                if tool_name not in available:
                    raise ValueError("This decision is not available in the current task.")
                with self._environment.participant_request_requires_decision():
                    with self._environment.expected_participant_decision(
                        decision_type, decision_id, value
                    ):
                        receipt = await self._route(
                            ctx, tool_name,
                            {"agent_id": caller_id, "decision_type": decision_type,
                             "decision_id": decision_id, value_field: value},
                            readonly=False, instruction=instruction,
                            template_mode=template_mode, trace_id=trace_id,
                            parent_span_id=parent_span_id,
                        )
                expected = (caller_id, self._environment.current_period, decision_type, value)
                if self._environment._decision_receipts.get(decision_id) != expected:
                    raise RuntimeError("Router did not save the participant's exact decision.")
                if (
                    receipt.get("decision_id") != decision_id
                    or receipt.get(value_field) != value
                    or type(receipt.get(value_field)) is not type(value)
                    or receipt.get("period") != self._environment.current_period
                ):
                    raise RuntimeError("Router returned a receipt inconsistent with the environment.")
                return self._result_pair(tool_name, receipt)

            if self._is_observation_request(instruction):
                # A fixed readonly request needs identity binding, not an LLM.
                # Keep the same decorated environment tool and caller checks.
                observation = await self._environment.get_observation(agent_id=caller_id)
                if observation.get("agent_id") != caller_id:
                    raise PermissionError("Router returned another participant's observation.")
                return self._result_pair("get_observation", observation)

        raise ValueError(
            "Unsupported risk-sharing environment request. "
            "Only <observe> and explicit participant decisions are allowed."
        )

    def set_current_time(self, t: datetime) -> None:
        self._router.t = t
        self._environment.t = t

    async def init(self, start_datetime: datetime) -> None:
        await self._environment.init(start_datetime)

    async def step(self, tick: int, t: datetime) -> None:
        await self._environment.step(tick, t)

    async def get_world_description(self) -> str:
        return "Risk-sharing environment with private pair observations and transfer decisions."
