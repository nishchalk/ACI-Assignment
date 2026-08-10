# Design Document — Assignment 2 PS12 (Bayesian Network)

**Course:** AIMLCZG557 / AECLZG557 · S2_2025-2026  
**Group ID:** `<GROUP_ID>`  
**Members:** `<NAME_1> (<REG_1>)`, `<NAME_2> (<REG_2>)`, …  
**Python source:** `assignment.py`  
**Limit:** Convert/export this draft to `designPS12_<GROUP_ID>.pdf` (≤ 4 pages).

---

## 1. Task A — PEAS Description

The intelligent agent monitors road conditions and predicts traffic accidents (and related signal failures) under uncertainty.

| Component | Description |
|-----------|-------------|
| **Performance measure** | Calibrated probability of accident / signal failure; high recall for true emergencies; low false-alert rate; timely update of posteriors as new evidence arrives; operator usefulness of ranked risk. |
| **Environment** | Urban/highway road network with uncertain weather and congestion; partial observability; noisy, overlapping symptoms (delay, emergency calls, camera/sensor alerts); discrete events that may or may not co-occur. |
| **Actuators** | Emit risk posteriors and alerts to operators; trigger emergency-call prioritization; request signal inspection / traffic diversion; write structured logs for audit. |
| **Sensors** | Traffic-delay detectors; emergency-call feeds; roadside / camera alert units; signal-health sensors; (optional upstream) weather and congestion feeds if the network is extended. |

The agent is **partially observable** and must reason with incomplete evidence rather than waiting for a complete symptom set.

---

## 2. Task B — Bayesian Network Model & Inference

### 2.1 Network structures

**Scenario 1 — Road Accident Prediction**

```text
        A (Road Accident)
       / \
      v   v
     D     E
 (Delay) (Emergency Call)
```

Factorization: \(P(A,D,E)=P(A)\,P(D\mid A)\,P(E\mid A)\).  
No direct \(D\)–\(E\) edge ⇒ \(D \perp E \mid A\); shared cause ⇒ \(D \not\!\perp E\) marginally.

**Scenario 2 — Traffic Signal Failure Detection**

```text
        S (Signal Failure)
       / \
      v   v
     C     R
 (Camera) (Sensor)
```

Factorization: \(P(S,C,R)=P(S)\,P(C\mid S)\,P(R\mid S)\).  
Same conditional / marginal independence pattern for \(C,R\) given \(S\).

All variables are binary \(\{T,F\}\).

### 2.2 Inference method

Implementation uses a **reusable joint-enumeration** engine in `assignment.py`:

1. Build the full joint table (8 rows) from the factorization.  
2. For query \(P(Q=q\mid e)\), sum joint mass where \(Q=q\) and evidence \(e\) holds; divide by \(P(e)\).  
3. Independence tests compare \(P(Y,Z)\) vs \(P(Y)P(Z)\) and \(P(Y,Z\mid x)\) vs \(P(Y\mid x)P(Z\mid x)\).

This avoids hard-coded closed forms per query so the same API serves both scenarios and arbitrary evidence subsets.

### 2.3 Sample results (from `inputPS12.txt`)

| Query | Posterior |
|-------|-----------|
| \(P(A\mid D)\) | 0.2095 |
| \(P(A\mid E)\) | 0.3816 |
| \(P(A\mid D,E)\) | 0.6848 |
| \(P(S\mid C)\) | 0.2830 |
| \(P(S\mid R)\) | 0.3640 |
| \(P(S\mid C,R)\) | 0.8111 |

**Interpretation:** Combining two dependent symptoms sharply raises belief in the hidden cause (explaining-away / common-cause update). Marginal dependence of the two effects is confirmed numerically; conditional independence given the cause also holds, matching d-separation on the graph.

---

## 3. Task C — Comparative Analysis (BN vs rule-based)

1. **Probabilistic vs IF–THEN** — Rules fire on crisp boolean antecedents (`IF delay AND call THEN accident`). BNs maintain graded beliefs and remain useful when only one symptom is observed.  
2. **Uncertain / incomplete / overlapping symptoms** — Enumeration conditions on any evidence subset. Rules need hand-authored patterns for every partial combination and struggle with noisy overlap.  
3. **Conditional dependencies & accuracy** — Treating alerts as independent underestimates joint evidence; sample posteriors jump from ~0.28–0.36 (single alert) to **0.81** (both), which fixed rules typically cannot express as calibrated confidence.  
4. **Evidence propagation** — Each new observation renormalizes the joint; posteriors move continuously rather than flipping a single boolean flag.  
5. **Conditional independence & d-separation** — Given \(A\) (or \(S\)), effects factorize → fewer CPT parameters and an 8-atom joint instead of an unstructured full table; d-separation justifies skipping edges between effects.

---

## 4. Alternate modelling approach & performance

**Alternative:** Deterministic multi-rule expert system (priority list of IF–THEN rules over the same sensors), optionally with ad-hoc confidence scores.

| Aspect | Common-cause BN (chosen) | Rule-based alternative |
|--------|--------------------------|------------------------|
| Uncertainty | Native posteriors | Brittle / heuristic scores |
| Partial evidence | Any subset via enumeration | Many rule variants required |
| Extending variables | Add nodes/CPTs; cost \(O(2^n)\) naive enum | Rule explosion |
| Runtime (3 binaries) | Negligible (8 rows) | Negligible |
| Larger networks | Prefer variable elimination / junction tree | Hard to keep consistent |

For this assignment size, enumeration is exact and cheap. For wider traffic graphs (weather, congestion, multiple segments), structured BN inference scales better than maintaining a consistent rule base, while pure enumeration would need replacement by factor algorithms.

---

## 5. Implementation notes

- Single module: `assignment.py`.  
- CPTs loaded from `inputPS12.txt` (no hard-coded evaluation values).  
- Output written to `outputPS12.txt` with round-half-up 4-decimal formatting.  
- `JointProbabilityTable` supports bounded insert/delete with empty/full messages.  
- Parser rejects missing/duplicate keys, invalid probabilities, and unknown scenario IDs.

**Reference:** Russell & Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. (Bayesian networks; exact inference by enumeration).

---

*Replace all `<...>` placeholders before exporting to PDF. Keep final PDF ≤ 4 pages.*
