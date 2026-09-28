"""检查 paper_spec.py 中已经声明的协议值及其内部一致性。

在项目根目录运行：
    python -m pytest tests/test_paper_spec.py -v

边界：不运行环境、不生成随机数、不调用 AgentSociety2 或 LLM。
预期值独立写在测试中，不能从被测常量复制，否则无法发现常量被误改。
25V 的提交约束按当前协议检查；25F 的 None 是暂时的研究待定状态。
"""

import pytest

from protocol import paper_spec as spec


# 1. Treatment 和原论文样本信息，不限制调试仿真的人数。
def test_treatment_definitions() -> None:
    actual = {
        name: member.value
        for name, member in spec.Treatment.__members__.items()
    }
    assert actual == {
        "BASELINE": "baseline",
        "VOLUNTARY_INSURANCE_25": "25V",
        "FORCED_INSURANCE_25": "25F",
    }


@pytest.mark.parametrize(
    "treatment_name, expected_subjects, expected_pairs",
    [
        ("BASELINE", 32, 16),
        ("VOLUNTARY_INSURANCE_25", 36, 18),
        ("FORCED_INSURANCE_25", 40, 20),
    ],
)
def test_original_sample_sizes(
    treatment_name: str, expected_subjects: int, expected_pairs: int
) -> None:
    treatment = spec.Treatment[treatment_name]
    subjects = spec.TREATMENT_SUBJECT_COUNTS[treatment]
    pairs = spec.TREATMENT_PAIR_COUNTS[treatment]

    assert type(subjects) is int
    assert type(pairs) is int
    assert subjects == expected_subjects
    assert pairs == expected_pairs
    assert subjects == pairs * spec.PAIR_SIZE


def test_original_sample_totals_are_consistent() -> None:
    treatments = set(spec.Treatment)
    assert set(spec.TREATMENT_SUBJECT_COUNTS) == treatments
    assert set(spec.TREATMENT_PAIR_COUNTS) == treatments
    assert spec.TOTAL_SUBJECT_COUNT == 108
    assert spec.TOTAL_PAIR_COUNT == 54
    assert sum(spec.TREATMENT_SUBJECT_COUNTS.values()) == spec.TOTAL_SUBJECT_COUNT
    assert sum(spec.TREATMENT_PAIR_COUNTS.values()) == spec.TOTAL_PAIR_COUNT
    assert spec.TOTAL_SUBJECT_COUNT == spec.TOTAL_PAIR_COUNT * spec.PAIR_SIZE


# 2. 配对和信息可见性。此处只检查声明，不证明接口已实现隔离。
def test_pairing_and_communication_rules() -> None:
    assert type(spec.PAIR_SIZE) is int
    assert spec.PAIR_SIZE == 2
    assert spec.PAIRING_IS_RANDOM is True
    assert spec.PAIRING_REMAINS_FIXED is True
    assert spec.DIRECT_COMMUNICATION_ALLOWED is False
    assert spec.OTHER_PAIRS_INFORMATION_IS_VISIBLE is False


@pytest.mark.parametrize(
    "constant_name",
    [
        "CURRENT_RANDOM_INCOME_VISIBILITY",
        "RANDOM_INCOME_HISTORY_VISIBILITY",
        "TRANSFER_HISTORY_VISIBILITY",
        "INSURANCE_STATUS_VISIBILITY",
        "INSURANCE_HISTORY_VISIBILITY",
    ],
)
def test_visibility_is_limited_to_fixed_pair(constant_name: str) -> None:
    # 不设置默认值：字段缺失应直接报错，不能自动跳过。
    actual = getattr(spec, constant_name)
    assert actual == "fixed_pair_members", f"{constant_name} 的可见范围错误"


# 3. 分析阶段：检查准确边界、字典一致性、无重叠且无空缺。
@pytest.mark.parametrize(
    "phase, expected_start, expected_end",
    [(1, 1, 15), (2, 16, 30), (3, 31, 45)],
)
def test_phase_boundaries(
    phase: int, expected_start: int, expected_end: int
) -> None:
    start = getattr(spec, f"PHASE_{phase}_START")
    end = getattr(spec, f"PHASE_{phase}_END")

    assert type(start) is int
    assert type(end) is int
    assert (start, end) == (expected_start, expected_end)
    assert spec.PHASES[phase] == (expected_start, expected_end)
    assert end - start + 1 == 15


def test_phases_cover_analysis_window_without_gaps_or_overlap() -> None:
    assert set(spec.PHASES) == {1, 2, 3}
    assert spec.ANALYSIS_LAST_PERIOD == 45
    assert spec.ANALYSIS_LAST_PERIOD == spec.PHASE_3_END

    covered_periods = []
    for phase in (1, 2, 3):
        start, end = spec.PHASES[phase]
        covered_periods.extend(range(start, end + 1))

    # 重复轮次、漏轮次和错误顺序都会导致这里失败。
    assert covered_periods == list(range(1, 46))


# 4. 收入金额和互补随机收入的制度声明。
def test_income_amounts() -> None:
    assert spec.FIXED_INCOME == 125
    assert spec.HIGH_RANDOM_INCOME == 200
    assert spec.LOW_RANDOM_INCOME == 0


def test_pair_income_is_complementary_not_independent() -> None:
    assert spec.EXACTLY_ONE_MEMBER_RECEIVES_HIGH_RANDOM_INCOME_PER_PERIOD is True
    assert spec.OTHER_MEMBER_RECEIVES_LOW_RANDOM_INCOME_PER_PERIOD is True
    assert spec.PAIR_RANDOM_INCOME_SAMPLING_IS_INDEPENDENT is False


# 5. 保险合同及时间安排。
def test_insurance_contract_amounts() -> None:
    assert spec.INSURANCE_COST == 25
    assert spec.INSURANCE_PAYOUT_LOW_RANDOM_INCOME == 50
    assert spec.INSURANCE_PAYOUT_HIGH_RANDOM_INCOME == 0


def test_insurance_decision_and_active_periods() -> None:
    assert spec.INSURANCE_OPPORTUNITY_REVEALED_PERIOD == 16
    assert spec.INSURANCE_DECISION_PERIOD == 16
    assert spec.INSURANCE_DECISION_COUNT == 1
    assert spec.INSURANCE_DECISION_IS_ONCE_AND_FOR_ALL is True
    assert spec.INSURANCE_ACTIVE_START_PERIOD == 16
    assert spec.INSURANCE_ACTIVE_END_PERIOD == 30
    assert spec.INSURANCE_REMOVAL_REVEALED_PERIOD == 31


def test_insurance_timeline_matches_phases() -> None:
    assert (
        spec.INSURANCE_OPPORTUNITY_REVEALED_PERIOD
        == spec.INSURANCE_DECISION_PERIOD
        == spec.INSURANCE_ACTIVE_START_PERIOD
        == spec.PHASE_2_START
    )
    assert spec.INSURANCE_ACTIVE_END_PERIOD == spec.PHASE_2_END
    assert (
        spec.INSURANCE_REMOVAL_REVEALED_PERIOD
        == spec.INSURANCE_ACTIVE_END_PERIOD + 1
        == spec.PHASE_3_START
    )


def test_25v_decisions_are_private_until_both_submit() -> None:
    # “独立提交”指看不到对方尚未公开的选择，不指选择在统计上独立。
    assert spec.TREATMENT_25V_INSURANCE_DECISIONS_ARE_SIMULTANEOUS is True
    assert spec.TREATMENT_25V_INSURANCE_SUBMISSIONS_ARE_INDEPENDENT is True
    assert spec.TREATMENT_25V_PARTNER_DECISION_VISIBLE_BEFORE_OWN_SUBMISSION is False
    assert spec.TREATMENT_25V_INSURANCE_STATUS_VISIBLE_AFTER_BOTH_SUBMISSIONS is True


# 6. 转账制度，不在这里重新实现转账结算。
def test_transfer_institution() -> None:
    assert spec.MIN_TRANSFER == 0
    assert spec.HIGH_INCOME_MEMBER_CAN_TRANSFER is True
    assert spec.LOW_INCOME_MEMBER_CAN_TRANSFER is False
    assert spec.TRANSFER_MAXIMUM_IS_SENDER_CURRENT_INCOME is True


# 7. 参与者获知的结束信息，与后台实际结束规则分开检查。
def test_participant_facing_horizon() -> None:
    assert spec.PARTICIPANT_MINIMUM_PERIOD_COUNT == 45
    assert spec.PARTICIPANT_IS_TOLD_ENDING_IS_STOCHASTIC is True
    assert spec.PARTICIPANT_IS_TOLD_EXACT_ENDING_PROBABILITIES is False
    assert spec.PARTICIPANT_IS_TOLD_MAXIMUM_PERIOD_COUNT is False


def test_backend_termination_boundaries() -> None:
    assert spec.REAL_TERMINATION_PERIOD_1_47_ALLOWED is False
    assert spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD == 48
    assert spec.REAL_TERMINATION_FORCED_PERIOD == 50
    assert spec.REAL_MAXIMUM_PERIOD_COUNT == 50


def test_analysis_announced_and_actual_horizons_are_consistent() -> None:
    assert spec.ANALYSIS_LAST_PERIOD == spec.PARTICIPANT_MINIMUM_PERIOD_COUNT
    assert (
        spec.ANALYSIS_LAST_PERIOD
        < spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD
        < spec.REAL_TERMINATION_FORCED_PERIOD
    )
    assert spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD - 1 == 47
    assert spec.REAL_TERMINATION_FORCED_PERIOD - 1 == 49
    assert spec.REAL_TERMINATION_FORCED_PERIOD == spec.REAL_MAXIMUM_PERIOD_COUNT


# 统一核对收入概率、覆盖比例和分段结束概率。
@pytest.mark.parametrize(
    "constant_name, expected",
    [
        ("HIGH_INCOME_MEMBER_PROBABILITY", 0.5),
        ("INSURANCE_COVERAGE_FRACTION", 0.25),
        ("REAL_ENDING_PROBABILITY_PERIOD_1_47", 0.0),
        ("REAL_ENDING_PROBABILITY_PERIOD_48_49", 0.78),
        ("REAL_ENDING_PROBABILITY_PERIOD_50", 1.0),
    ],
)
def test_probability_and_fraction_constants(
    constant_name: str, expected: float
) -> None:
    actual = getattr(spec, constant_name)
    assert type(actual) in (int, float), f"{constant_name} 应为数值而非布尔值或字符串"
    assert 0.0 <= actual <= 1.0
    # 这里核对固定字面值，不是浮点运算的近似结果。
    assert actual == expected, f"{constant_name} 不符合当前协议"


# 8. 随机单轮支付的声明。
def test_payment_selects_one_random_period() -> None:
    assert spec.PAYMENT_SELECTED_PERIOD_COUNT == 1
    assert spec.PAYMENT_PERIOD_SELECTION_IS_RANDOM is True


# 9. 25F 的确定规则与暂未核实的参数。
def test_25f_forbids_both_members_being_uninsured() -> None:
    assert spec.F25_BOTH_UNINSURED_ALLOWED is False


def test_25f_assignment_probability_remains_explicitly_unresolved() -> None:
    # 临时保护：防止未经确认就填写分配概率。
    # 获得可靠依据并正式更新协议后，应同时更新本测试。
    # 此项通过不表示 25F 已具备可运行的随机分配算法。
    assert spec.F25_INSURANCE_ASSIGNMENT_PROBABILITY is None