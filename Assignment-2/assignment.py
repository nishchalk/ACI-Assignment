#!/usr/bin/env python3
"""
Assignment 2 – PS12: Bayesian Network for traffic monitoring.

Implements a reusable common-cause Bayesian Network with exact inference
via joint enumeration (not hard-coded per-query formulas).

Reads:  inputPS12.txt
Writes: outputPS12.txt
"""

from __future__ import annotations

import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROBLEM_SET_ID = "PS12"
INPUT_FILENAME = "inputPS12.txt"
OUTPUT_FILENAME = "outputPS12.txt"

SCENARIO_1 = "SCENARIO_1_ROAD_ACCIDENT"
SCENARIO_2 = "SCENARIO_2_TRAFFIC_SIGNAL_FAILURE"

SCENARIO_1_KEYS = (
    "P_A",
    "P_D_given_A",
    "P_D_given_notA",
    "P_E_given_A",
    "P_E_given_notA",
)

SCENARIO_2_KEYS = (
    "P_S",
    "P_C_given_S",
    "P_C_given_notS",
    "P_R_given_S",
    "P_R_given_notS",
)

INDEPENDENCE_TOLERANCE = 1e-9
JOINT_CAPACITY = 8  # binary common-cause network: 2^3 rows


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_prob(value: float, places: int = 4) -> str:
    """Format a probability with round-half-up to the given decimal places."""
    quant = Decimal("1").scaleb(-places)
    return str(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def bool_to_tf(flag: bool) -> str:
    return "T" if flag else "F"


# ---------------------------------------------------------------------------
# Bounded joint table (insert / delete with capacity messages)
# ---------------------------------------------------------------------------

class JointProbabilityTable:
    """
    Fixed-capacity store for joint assignments (X, Y, Z, probability).
    Insert/delete report when the table is full or empty.
    """

    def __init__(self, capacity: int = JOINT_CAPACITY) -> None:
        self.capacity = capacity
        self._rows: List[Tuple[bool, bool, bool, float]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def is_empty(self) -> bool:
        return len(self._rows) == 0

    def is_full(self) -> bool:
        return len(self._rows) >= self.capacity

    def insert(self, x: bool, y: bool, z: bool, probability: float) -> None:
        if self.is_full():
            raise ValueError(
                f"JointProbabilityTable insert failed: table is full "
                f"(capacity={self.capacity})."
            )
        self._rows.append((x, y, z, probability))

    def delete(self) -> Tuple[bool, bool, bool, float]:
        if self.is_empty():
            raise ValueError(
                "JointProbabilityTable delete failed: table is empty."
            )
        return self._rows.pop()

    def rows(self) -> List[Tuple[bool, bool, bool, float]]:
        return list(self._rows)

    def clear(self) -> None:
        self._rows.clear()


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

class InputParseError(Exception):
    """Raised when the input file is malformed or incomplete."""


def validate_probability(raw: str, key: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise InputParseError(
            f"Invalid probability for {key}: '{raw}' is not a number."
        ) from exc
    if not (0.0 <= value <= 1.0):
        raise InputParseError(
            f"Invalid probability for {key}: {value} is outside [0, 1]."
        )
    return value


def parse_input(path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse inputPS12.txt into scenario CPT dictionaries.

    Returns:
        { SCENARIO_1: {key: float, ...}, SCENARIO_2: {key: float, ...} }
    """
    if not os.path.isfile(path):
        raise InputParseError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]

    # Drop blanks but keep order of meaningful lines
    content = [line for line in lines if line]
    if not content:
        raise InputParseError("Input file is empty.")

    if content[0] != PROBLEM_SET_ID:
        raise InputParseError(
            f"Invalid problem-set identifier '{content[0]}'; "
            f"expected '{PROBLEM_SET_ID}'."
        )

    scenarios: Dict[str, Dict[str, float]] = {}
    current: Optional[str] = None
    index = 1

    known_headers = {SCENARIO_1, SCENARIO_2}

    while index < len(content):
        token = content[index]
        if token in known_headers:
            if token in scenarios:
                raise InputParseError(f"Duplicate scenario identifier: {token}")
            scenarios[token] = {}
            current = token
            index += 1
            continue

        if current is None:
            raise InputParseError(
                f"Unexpected line before any scenario header: '{token}'"
            )

        if "=" not in token:
            raise InputParseError(
                f"Malformed input line (expected KEY=value): '{token}'"
            )

        key, _, raw_value = token.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            raise InputParseError(
                f"Malformed input line (empty key or value): '{token}'"
            )

        if key in scenarios[current]:
            raise InputParseError(
                f"Duplicate key '{key}' in scenario {current}."
            )

        scenarios[current][key] = validate_probability(raw_value, key)
        index += 1

    required = {
        SCENARIO_1: SCENARIO_1_KEYS,
        SCENARIO_2: SCENARIO_2_KEYS,
    }
    for scenario_id, keys in required.items():
        if scenario_id not in scenarios:
            raise InputParseError(f"Missing scenario: {scenario_id}")
        missing = [k for k in keys if k not in scenarios[scenario_id]]
        if missing:
            raise InputParseError(
                f"Missing keys in {scenario_id}: {', '.join(missing)}"
            )
        extra = [k for k in scenarios[scenario_id] if k not in keys]
        if extra:
            raise InputParseError(
                f"Unknown keys in {scenario_id}: {', '.join(extra)}"
            )

    return scenarios


# ---------------------------------------------------------------------------
# Bayesian Network: common-cause structure X -> Y, X -> Z
# ---------------------------------------------------------------------------

class CommonCauseBN:
    """
    Bayesian Network with factorization:
        P(X, Y, Z) = P(X) * P(Y | X) * P(Z | X)
    """

    def __init__(
        self,
        p_x: float,
        p_y_given_x: float,
        p_y_given_not_x: float,
        p_z_given_x: float,
        p_z_given_not_x: float,
    ) -> None:
        self.p_x = p_x
        self.p_y_given_x = p_y_given_x
        self.p_y_given_not_x = p_y_given_not_x
        self.p_z_given_x = p_z_given_x
        self.p_z_given_not_x = p_z_given_not_x

    def _p_x(self, x: bool) -> float:
        return self.p_x if x else (1.0 - self.p_x)

    def _p_y_given_x(self, y: bool, x: bool) -> float:
        base = self.p_y_given_x if x else self.p_y_given_not_x
        return base if y else (1.0 - base)

    def _p_z_given_x(self, z: bool, x: bool) -> float:
        base = self.p_z_given_x if x else self.p_z_given_not_x
        return base if z else (1.0 - base)

    def joint_entry(self, x: bool, y: bool, z: bool) -> float:
        return (
            self._p_x(x)
            * self._p_y_given_x(y, x)
            * self._p_z_given_x(z, x)
        )

    def build_joint(self) -> JointProbabilityTable:
        """
        Build the complete joint table in sample order:
        for X in [T, F]:
          for Y in [T, F]:
            for Z in [T, F]:
        """
        table = JointProbabilityTable(capacity=JOINT_CAPACITY)
        for x in (True, False):
            for y in (True, False):
                for z in (True, False):
                    table.insert(x, y, z, self.joint_entry(x, y, z))
        return table


# ---------------------------------------------------------------------------
# Exact inference by joint enumeration
# ---------------------------------------------------------------------------

def _matches(
    row: Tuple[bool, bool, bool, float],
    evidence: Dict[str, bool],
    var_names: Sequence[str],
) -> bool:
    assignment = {
        var_names[0]: row[0],
        var_names[1]: row[1],
        var_names[2]: row[2],
    }
    for name, value in evidence.items():
        if assignment[name] != value:
            return False
    return True


def marginal(
    joint: JointProbabilityTable,
    evidence: Dict[str, bool],
    var_names: Sequence[str],
) -> float:
    """Sum joint probability mass consistent with evidence."""
    total = 0.0
    for row in joint.rows():
        if _matches(row, evidence, var_names):
            total += row[3]
    return total


def enumerate_prob(
    joint: JointProbabilityTable,
    query_var: str,
    query_value: bool,
    evidence: Dict[str, bool],
    var_names: Sequence[str],
) -> float:
    """
    Reusable exact inference: P(query_var = query_value | evidence)
    by enumerating (summing) joint entries and normalizing.
    """
    if query_var in evidence:
        return 1.0 if evidence[query_var] == query_value else 0.0

    numerator_ev = dict(evidence)
    numerator_ev[query_var] = query_value
    numerator = marginal(joint, numerator_ev, var_names)
    denominator = marginal(joint, evidence, var_names)
    if denominator == 0.0:
        raise ValueError(
            f"Cannot condition on evidence with zero probability: {evidence}"
        )
    return numerator / denominator


def are_marginally_independent(
    joint: JointProbabilityTable,
    y_name: str,
    z_name: str,
    var_names: Sequence[str],
    tol: float = INDEPENDENCE_TOLERANCE,
) -> bool:
    """Check Y ⊥ Z by comparing P(Y=T,Z=T) with P(Y=T)P(Z=T)."""
    p_yz = marginal(joint, {y_name: True, z_name: True}, var_names)
    p_y = marginal(joint, {y_name: True}, var_names)
    p_z = marginal(joint, {z_name: True}, var_names)
    return abs(p_yz - p_y * p_z) < tol


def are_conditionally_independent(
    joint: JointProbabilityTable,
    x_name: str,
    y_name: str,
    z_name: str,
    var_names: Sequence[str],
    tol: float = INDEPENDENCE_TOLERANCE,
) -> bool:
    """Check Y ⊥ Z | X for both values of X."""
    for x_val in (True, False):
        p_x = marginal(joint, {x_name: x_val}, var_names)
        if p_x == 0.0:
            continue
        p_yz_x = (
            marginal(
                joint,
                {x_name: x_val, y_name: True, z_name: True},
                var_names,
            )
            / p_x
        )
        p_y_x = marginal(joint, {x_name: x_val, y_name: True}, var_names) / p_x
        p_z_x = marginal(joint, {x_name: x_val, z_name: True}, var_names) / p_x
        if abs(p_yz_x - p_y_x * p_z_x) >= tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Scenario runners & output formatting
# ---------------------------------------------------------------------------

def format_joint_table(
    joint: JointProbabilityTable,
    labels: Sequence[str],
) -> List[str]:
    lines = [
        "Scenario 1 - Joint Probability Table",
        f"{labels[0]}\t{labels[1]}\t{labels[2]}\tProbability",
    ]
    for x, y, z, prob in joint.rows():
        lines.append(
            f"{bool_to_tf(x)}\t{bool_to_tf(y)}\t{bool_to_tf(z)}\t"
            f"{format_prob(prob)}"
        )
    return lines


def run_scenario_1(probs: Dict[str, float]) -> List[str]:
    bn = CommonCauseBN(
        p_x=probs["P_A"],
        p_y_given_x=probs["P_D_given_A"],
        p_y_given_not_x=probs["P_D_given_notA"],
        p_z_given_x=probs["P_E_given_A"],
        p_z_given_not_x=probs["P_E_given_notA"],
    )
    var_names = ("A", "D", "E")
    joint = bn.build_joint()

    p_a_d = enumerate_prob(joint, "A", True, {"D": True}, var_names)
    p_a_e = enumerate_prob(joint, "A", True, {"E": True}, var_names)
    p_a_de = enumerate_prob(
        joint, "A", True, {"D": True, "E": True}, var_names
    )

    marg_indep = are_marginally_independent(joint, "D", "E", var_names)
    cond_indep = are_conditionally_independent(
        joint, "A", "D", "E", var_names
    )

    lines = format_joint_table(joint, var_names)
    lines.append("Scenario 1: Road Accident Prediction")
    lines.append(f"P(A | D) = {format_prob(p_a_d)}")
    lines.append(f"P(A | E) = {format_prob(p_a_e)}")
    lines.append(f"P(A | D and E) = {format_prob(p_a_de)}")

    if marg_indep:
        lines.append(
            "Traffic Delay and Emergency Call are independent without evidence."
        )
    else:
        lines.append(
            "Traffic Delay and Emergency Call are not independent without evidence."
        )

    if cond_indep:
        lines.append(
            "Traffic Delay and Emergency Call are conditionally independent "
            "given Road Accident."
        )
    else:
        lines.append(
            "Traffic Delay and Emergency Call are not conditionally independent "
            "given Road Accident."
        )
    return lines


def run_scenario_2(probs: Dict[str, float]) -> List[str]:
    bn = CommonCauseBN(
        p_x=probs["P_S"],
        p_y_given_x=probs["P_C_given_S"],
        p_y_given_not_x=probs["P_C_given_notS"],
        p_z_given_x=probs["P_R_given_S"],
        p_z_given_not_x=probs["P_R_given_notS"],
    )
    var_names = ("S", "C", "R")
    joint = bn.build_joint()

    p_s_c = enumerate_prob(joint, "S", True, {"C": True}, var_names)
    p_s_r = enumerate_prob(joint, "S", True, {"R": True}, var_names)
    p_s_cr = enumerate_prob(
        joint, "S", True, {"C": True, "R": True}, var_names
    )

    marg_indep = are_marginally_independent(joint, "C", "R", var_names)
    cond_indep = are_conditionally_independent(
        joint, "S", "C", "R", var_names
    )

    lines = ["Scenario 2: Traffic Signal Failure Detection"]
    lines.append(f"P(S | C) = {format_prob(p_s_c)}")
    lines.append(f"P(S | R) = {format_prob(p_s_r)}")
    lines.append(f"P(S | C and R) = {format_prob(p_s_cr)}")

    if marg_indep:
        lines.append(
            "Camera Alert and Sensor Alert are independent without evidence."
        )
    else:
        lines.append(
            "Camera Alert and Sensor Alert are not independent without evidence."
        )

    if cond_indep:
        lines.append(
            "Camera Alert and Sensor Alert are conditionally independent "
            "given Traffic Signal Failure."
        )
    else:
        lines.append(
            "Camera Alert and Sensor Alert are not conditionally independent "
            "given Traffic Signal Failure."
        )
    return lines


def generate_output(scenarios: Dict[str, Dict[str, float]]) -> str:
    lines: List[str] = []
    lines.extend(run_scenario_1(scenarios[SCENARIO_1]))
    lines.extend(run_scenario_2(scenarios[SCENARIO_2]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, INPUT_FILENAME)
    output_path = os.path.join(base_dir, OUTPUT_FILENAME)

    try:
        scenarios = parse_input(input_path)
        output_text = generate_output(scenarios)
    except (InputParseError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(output_text)

    # Keep console quiet for submission-friendly runs; uncomment for local debug:
    # print(output_text, end="")
    # print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
