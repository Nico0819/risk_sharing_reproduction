"""Two-person income-sharing environment for a controlled simulation.

The experimental constants live in :mod:`protocol.paper_spec`.  This module
implements only the mechanical transfer rules; it deliberately contains no
insurance, agent, LLM, or analysis logic.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
    required = {"decision_type", "decision_id", "amount"}
    if variables is not None and not isinstance(variables, Mapping):
        raise ValueError("participant variables must be an object")
    supplied = {key: value for key, value in (variables or {}).items() if key in required}
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
            r"(?<!\w)(decision_type|decision_id|amount)[ \t]*[:=][ \t]*",
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
    if not isinstance(payload, dict):
        raise ValueError("participant decision envelope must be a JSON object")
    for name, value in supplied.items():
        if name in payload and (type(payload[name]) is not type(value) or payload[name] != value):
            raise ValueError(f"participant {name} conflicts with variables")
        payload[name] = value
    if required.difference(payload):
        raise ValueError(
            "participant decision must contain decision_type, decision_id, and amount"
        )
    if set(payload).difference(required):
        raise ValueError("participant decision contains unexpected fields")
    if not isinstance(payload["decision_type"], str):
        raise ValueError("participant decision_type must be a string")
    if not isinstance(payload["decision_id"], str):
        raise ValueError("participant decision_id must be a string")
    if type(payload["amount"]) is not int:
        raise ValueError("participant amount must be a strict integer")
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


class Stage(str, Enum):
    """The two externally meaningful states in the environment."""

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
    ) -> None:
        super().__init__()

        try:
            normalized_treatment = spec.Treatment(treatment)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown treatment: {treatment!r}") from exc
        if normalized_treatment is not spec.Treatment.BASELINE:
            raise NotImplementedError(
                "RiskSharingEnv currently implements Baseline only; "
                f"{normalized_treatment.value} is not implemented."
            )
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
        self._decision_receipts: dict[str, tuple[AgentId, int, str, int]] = {}
        self._rounds: dict[int, _PairRound] = {}
        self._trusted_caller: ContextVar[AgentId | None] = ContextVar(
            f"risk_sharing_caller_{id(self)}", default=None
        )
        self._expected_participant_decision: ContextVar[
            tuple[str, str, int] | None
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
        self, decision_type: str, decision_id: str, amount: int
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
                "amount": decision["amount"],
                **({"raw_request": raw_request} if raw_request else {}),
                "participant_variables": {
                    name: copy.deepcopy(value) for name, value in (variables or {}).items()
                    if name in {"decision_type", "decision_id", "amount"}
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
        self.stage = Stage.TRANSFER_DECISION
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
                agent_id: spec.FIXED_INCOME + amount
                for agent_id, amount in random_income.items()
            }
            self._rounds[pair_id] = _PairRound(
                pair_id=pair_id,
                members=members,
                random_income=random_income,
                income_before_transfer=income_before_transfer,
            )

    def _decision_id(self, pair_id: int, agent_id: AgentId) -> str:
        """Return a stable opaque token for this participant's current turn.

        The digest is deterministic and does not consume the experiment RNG.
        It binds the token to the session, period, pair and participant without
        exposing the researcher's session label (which may contain treatment,
        seed, or horizon metadata) to a participant.
        """

        material = repr(
            (
                "risk-sharing-decision-v1",
                self.session_id,
                self.current_period,
                pair_id,
                agent_id,
            )
        ).encode("utf-8")
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
                new_records.append(record)
                self.history.append(record)
                self.pair_history[pair_id].append(record)
            state.settled = True

        self.completed_periods.append(self.current_period)
        return new_records

    def _participant_history(self, pair_id: int) -> list[dict[str, Any]]:
        """Return only fields that belong in a participant's pair view."""

        return [
            {field: copy.deepcopy(record[field]) for field in _PARTICIPANT_HISTORY_FIELDS}
            for record in self.pair_history[pair_id]
        ]

    @tool(
        readonly=True,
        name="get_observation",
        description=(
            "Read the caller's current pair observation once. If task is transfer, "
            "use the returned decision_id and immediately call submit_decision; "
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

        state = self._rounds[pair_id]
        has_opportunity = state.random_income[agent_id] == spec.HIGH_RANDOM_INCOME
        submitted = agent_id in state.transfer_amount
        max_transfer = state.income_before_transfer[agent_id] if has_opportunity else None
        return {
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
        expected = self._expected_participant_decision.get()
        if self._participant_request_requires_decision.get() and expected is None:
            raise PermissionError(
                "An environment request cannot create an economic decision without a participant model envelope."
            )
        if expected is not None and (decision_type, decision_id, amount) != expected:
            raise PermissionError(
                "The environment adapter may only forward the exact decision emitted by the participant model."
            )
        receipt = self._decision_receipts.get(decision_id)
        if receipt is not None:
            recorded_agent, period, recorded_type, recorded_amount = receipt
            if period != self.current_period:
                raise ValueError("This decision_id belongs to a previous period and is stale.")
            if (
                recorded_agent == agent_id
                and recorded_type == decision_type
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
        return all(bool(state.transfer_amount) for state in self._rounds.values())

    async def init(self, start_datetime: datetime) -> None:
        """Set framework time; paper-period state is initialized in the constructor."""

        self.t = start_datetime

    async def step(self, tick: int, t: datetime) -> None:
        """Advance framework time and settle a fully submitted paper period."""

        self.t = t
        if self.finished or not self._all_required_decisions_submitted():
            return

        self._settle_current_period()
        ended = self._sample_ending_after_settlement()
        if ended:
            self.finished = True
            self.stage = Stage.FINISHED
            return

        self.current_period += 1
        self._prepare_period()



class RiskSharingReActRouter(ReActRouter):
    """Use the real Router tool loop, returning tool data without a summary LLM."""

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
                    if name in {"decision_type", "decision_id", "amount"}
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
                amount = participant_decision["amount"]
                with self._environment.participant_request_requires_decision():
                    with self._environment.expected_participant_decision(
                        decision_type, decision_id, amount
                    ):
                        receipt = await self._route(
                            ctx, "submit_decision",
                            {"agent_id": caller_id, "decision_type": decision_type,
                             "decision_id": decision_id, "amount": amount},
                            readonly=False, instruction=instruction,
                            template_mode=template_mode, trace_id=trace_id,
                            parent_span_id=parent_span_id,
                        )
                expected = (caller_id, self._environment.current_period, decision_type, amount)
                if self._environment._decision_receipts.get(decision_id) != expected:
                    raise RuntimeError("Router did not save the participant's exact decision.")
                if (
                    receipt.get("decision_id") != decision_id
                    or receipt.get("amount") != amount
                    or receipt.get("period") != self._environment.current_period
                ):
                    raise RuntimeError("Router returned a receipt inconsistent with the environment.")
                return self._result_pair("submit_decision", receipt)

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
