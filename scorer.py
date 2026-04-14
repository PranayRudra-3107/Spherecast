"""
scorer.py
---------
Weighted multi-criteria scoring engine for substitute candidates.

Scoring dimensions (from Executive Summary):
  A. Spec similarity     35%  — numeric distance across 15+ attributes
  B. Compliance          20%  — RoHS/REACH/FDA pass/fail/unknown (hard gate)
  C. Price & lead time   15%  — cost delta, MOQ, lead time
  D. Quality/reliability 15%  — supplier scorecard + compliance completeness
  E. Business priority   10%  — category match, preferred suppliers
  F. Traceability         5%  — provenance completeness (new)

Compliance is a HARD GATE: any "fail" = disqualified before scoring.
Feedback adjustments (+/- 0.2) are applied to the blended total.
ESG hints (recycled, sustainable) add a small bonus to business priority.
"""

import math
from typing import Optional
from knowledge_store import get_conn, get_specs_map, get_material_full, get_feedback_score

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "spec_similarity":  0.35,
    "compliance":       0.20,
    "price_lead_time":  0.15,
    "quality":          0.15,
    "business_priority":0.10,
    "traceability":     0.05,
}

# ESG keywords that signal sustainable / lower-risk sourcing
ESG_POSITIVE_KEYWORDS = [
    "recycled", "recyclable", "bio-based", "biobased", "sustainable",
    "renewable", "low-carbon", "conflict-free", "drc conflict free",
    "iso 14001", "fsc certified",
]

# Specs used for similarity scoring and their relative importance
SPEC_IMPORTANCE = {
    "tensile_strength": 1.0,
    "yield_strength": 0.9,
    "elastic_modulus": 0.8,
    "density": 0.7,
    "max_operating_temp": 0.8,
    "thermal_conductivity": 0.6,
    "elongation_at_break": 0.6,
    "melting_point": 0.5,
    "coefficient_thermal_expansion": 0.5,
    "hardness_vickers": 0.5,
    "glass_transition_temp": 0.7,
    "flexural_strength": 0.7,
    "impact_strength": 0.6,
    "purity_percent": 0.8,
}

PREFERRED_SUPPLIERS = set()  # populate from config in production


# ---------------------------------------------------------------------------
# Hard compliance gate
# ---------------------------------------------------------------------------

def compliance_gate(material_id: str) -> tuple[bool, list[str]]:
    """
    Returns (passes, [reasons_for_failure]).
    Any "fail" status on a compliance record disqualifies the candidate.
    """
    with get_conn() as conn:
        records = conn.execute(
            "SELECT standard, status, notes FROM compliance WHERE material_id = ?",
            (material_id,),
        ).fetchall()

    failures = [
        f"{r['standard']}: {r['notes'] or 'non-compliant'}"
        for r in records
        if r["status"] == "fail"
    ]
    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Dimension A: Spec similarity
# ---------------------------------------------------------------------------

def score_spec_similarity(query_specs: dict, candidate_specs: dict) -> tuple[float, dict]:
    """
    Returns (score 0-1, detail_dict).
    Uses weighted normalized distance across shared numeric specs.
    """
    total_weight = 0.0
    weighted_similarity = 0.0
    details = {}

    for attr, importance in SPEC_IMPORTANCE.items():
        q = query_specs.get(attr)
        c = candidate_specs.get(attr)

        if q is None or c is None:
            # Missing spec — partial penalty
            if q is not None:  # query has it, candidate doesn't
                weighted_similarity += 0.3 * importance  # partial credit
                details[attr] = {"status": "missing_in_candidate", "similarity": 0.3}
            total_weight += importance
            continue

        q_val = q.get("value")
        c_val = c.get("value")

        if q_val is None or c_val is None or q_val == 0:
            total_weight += importance
            continue

        # Normalized similarity: 1 - |delta| / max(q, c)
        delta_ratio = abs(q_val - c_val) / max(abs(q_val), abs(c_val))
        # Apply a smooth decay: perfect match = 1.0, 50% difference = 0.5, 100%+ = near 0
        similarity = math.exp(-2.0 * delta_ratio)
        weighted_similarity += similarity * importance
        total_weight += importance
        details[attr] = {
            "query": q_val,
            "candidate": c_val,
            "unit": q.get("unit"),
            "delta_pct": round(delta_ratio * 100, 1),
            "similarity": round(similarity, 3),
        }

    score = weighted_similarity / total_weight if total_weight > 0 else 0.5
    return round(score, 4), details


# ---------------------------------------------------------------------------
# Dimension B: Compliance
# ---------------------------------------------------------------------------

def score_compliance(material_id: str) -> tuple[float, dict]:
    """
    Returns (score 0-1, detail_dict).
    pass=1.0, unknown=0.6, fail=0.0 (but fail should have been caught by gate).
    """
    with get_conn() as conn:
        records = conn.execute(
            "SELECT standard, status FROM compliance WHERE material_id = ?",
            (material_id,),
        ).fetchall()

    status_scores = {"pass": 1.0, "unknown": 0.6, "fail": 0.0}
    details = {}
    scores = []

    for r in records:
        s = status_scores.get(r["status"], 0.6)
        scores.append(s)
        details[r["standard"]] = r["status"]

    score = sum(scores) / len(scores) if scores else 0.6  # neutral if no data
    return round(score, 4), details


# ---------------------------------------------------------------------------
# Dimension C: Price & lead time
# ---------------------------------------------------------------------------

def score_price_lead_time(
    query_price: Optional[dict],
    candidate_price: Optional[dict],
) -> tuple[float, dict]:
    """
    Score based on: price ratio, lead time, MOQ.
    Lower price = better. Shorter lead time = better.
    """
    if not candidate_price:
        return 0.5, {"note": "no price data"}

    score = 0.5
    details = {}

    # Price score (40% of this dimension)
    if query_price and query_price.get("price_usd") and candidate_price.get("price_usd"):
        q_price = query_price["price_usd"]
        c_price = candidate_price["price_usd"]
        ratio = c_price / q_price
        # Cheaper is better. ratio < 1 = cheaper. ratio > 2 = expensive.
        price_score = max(0.0, min(1.0, 1.5 - ratio))
        score = price_score * 0.4 + 0.6  # bias toward neutral when price unknown
        details["price_usd"] = c_price
        details["price_vs_query_pct"] = round((ratio - 1) * 100, 1)
        details["price_score"] = round(price_score, 3)
    else:
        details["price_usd"] = candidate_price.get("price_usd")

    # Lead time score (35% of this dimension)
    lt = candidate_price.get("lead_time_days")
    if lt is not None:
        # <14 days = excellent, 14-30 = good, 30-60 = fair, >60 = poor
        if lt <= 14:
            lt_score = 1.0
        elif lt <= 30:
            lt_score = 0.8
        elif lt <= 60:
            lt_score = 0.5
        else:
            lt_score = 0.3
        score = (score + lt_score * 0.35) / 1.35
        details["lead_time_days"] = lt

    # MOQ factor (25% — high MOQ = less flexible)
    moq = candidate_price.get("moq")
    if moq is not None:
        moq_score = 1.0 if moq <= 50 else (0.7 if moq <= 200 else 0.4)
        score = (score + moq_score * 0.25) / 1.25
        details["moq"] = moq

    return round(min(score, 1.0), 4), details


# ---------------------------------------------------------------------------
# Dimension D: Quality
# ---------------------------------------------------------------------------

def score_quality(material_id: str) -> tuple[float, dict]:
    """
    Demo: heuristic based on compliance completeness + supplier known status.
    Production: plug in supplier scorecard + ERP defect rates.
    """
    with get_conn() as conn:
        mat = conn.execute("SELECT supplier FROM materials WHERE id = ?", (material_id,)).fetchone()
        comp_count = conn.execute(
            "SELECT COUNT(*) as c FROM compliance WHERE material_id = ? AND status = 'pass'",
            (material_id,),
        ).fetchone()["c"]

    supplier = mat["supplier"] if mat else ""
    details = {"supplier": supplier, "compliance_passes": comp_count}

    # Base quality from compliance completeness
    base = 0.5 + min(comp_count * 0.1, 0.4)

    # Preferred supplier bonus
    if supplier and supplier in PREFERRED_SUPPLIERS:
        base = min(base + 0.15, 1.0)
        details["preferred_supplier"] = True

    return round(base, 4), details


# ---------------------------------------------------------------------------
# Dimension E: Business priority
# ---------------------------------------------------------------------------

def score_business_priority(query_category: Optional[str], candidate_id: str) -> tuple[float, dict]:
    """Category match + preferred supplier + ESG sustainability signals."""
    with get_conn() as conn:
        mat = conn.execute(
            "SELECT category, supplier FROM materials WHERE id = ?", (candidate_id,)
        ).fetchone()
        prov_rows = conn.execute(
            "SELECT raw_text FROM provenance WHERE material_id = ?", (candidate_id,)
        ).fetchall()

    if not mat:
        return 0.5, {}

    details = {"category": mat["category"]}
    score = 0.5

    # Category match
    if query_category and mat["category"] and query_category.lower() == mat["category"].lower():
        score += 0.3
        details["category_match"] = True

    # Preferred supplier
    if mat["supplier"] and mat["supplier"] in PREFERRED_SUPPLIERS:
        score += 0.2
        details["preferred_supplier"] = True

    # ESG bonus — scan provenance text for sustainability signals
    full_text = " ".join(r["raw_text"] or "" for r in prov_rows).lower()
    esg_hits = [kw for kw in ESG_POSITIVE_KEYWORDS if kw in full_text]
    if esg_hits:
        esg_bonus = min(len(esg_hits) * 0.05, 0.15)
        score = min(score + esg_bonus, 1.0)
        details["esg_signals"] = esg_hits

    return round(min(score, 1.0), 4), details


# ---------------------------------------------------------------------------
# Dimension F: Traceability (new)
# ---------------------------------------------------------------------------

def score_traceability(candidate_id: str) -> tuple[float, dict]:
    """
    Score based on provenance completeness.
    More sources (PDF, website, ERP) = higher traceability = higher trust.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source_type FROM provenance WHERE material_id = ?", (candidate_id,)
        ).fetchall()

    source_types = {r["source_type"] for r in rows}
    count = len(rows)

    # Diversity bonus: having multiple source types is better than one
    diversity_bonus = min(len(source_types) * 0.1, 0.3)
    base = min(count / 3.0, 0.7)
    score = min(base + diversity_bonus, 1.0)

    details = {"source_count": count, "source_types": list(source_types)}
    return round(score, 4), details


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_candidate(
    query_material: dict,
    candidate_id: str,
    vector_score: float = 0.5,
    query_material_ref: dict = None,
) -> Optional[dict]:
    """
    Score a single candidate against a query material.
    Returns None if candidate fails compliance gate.
    """
    # Hard gate first
    passes, failures = compliance_gate(candidate_id)
    if not passes:
        return {
            "material_id": candidate_id,
            "disqualified": True,
            "disqualification_reason": failures,
        }

    query_specs = {s["attribute"]: {"value": s["value"], "unit": s["unit"]}
                   for s in query_material.get("specs", []) if s.get("value") is not None}
    candidate_specs = get_specs_map(candidate_id)

    # Score each dimension
    spec_score, spec_detail = score_spec_similarity(query_specs, candidate_specs)
    comp_score, comp_detail = score_compliance(candidate_id)
    price_score, price_detail = score_price_lead_time(
        query_material.get("price"),
        get_material_full(candidate_id).get("price"),
    )
    quality_score, quality_detail = score_quality(candidate_id)
    business_score, business_detail = score_business_priority(
        query_material.get("category"), candidate_id
    )

    trace_score, trace_detail = score_traceability(candidate_id)

    # Weighted composite (6 dimensions)
    total = (
        spec_score    * WEIGHTS["spec_similarity"]
        + comp_score  * WEIGHTS["compliance"]
        + price_score * WEIGHTS["price_lead_time"]
        + quality_score * WEIGHTS["quality"]
        + business_score * WEIGHTS["business_priority"]
        + trace_score * WEIGHTS["traceability"]
    )

    # Blend with vector similarity (soft signal, 15% influence)
    blended = total * 0.85 + vector_score * 0.15

    # Feedback adjustment from historical user approvals/rejections
    feedback_adj = get_feedback_score(
        query_material.get("id", ""),
        candidate_id,
    )
    blended = round(min(max(blended + feedback_adj, 0.0), 1.0), 4)

    return {
        "material_id": candidate_id,
        "disqualified": False,
        "scores": {
            "total":            blended,
            "spec_similarity":  round(spec_score, 4),
            "compliance":       round(comp_score, 4),
            "price_lead_time":  round(price_score, 4),
            "quality":          round(quality_score, 4),
            "business_priority":round(business_score, 4),
            "traceability":     round(trace_score, 4),
            "vector_similarity":round(vector_score, 4),
            "feedback_adj":     feedback_adj,
        },
        "details": {
            "spec":       spec_detail,
            "compliance": comp_detail,
            "price":      price_detail,
            "quality":    quality_detail,
            "business":   business_detail,
            "traceability": trace_detail,
        },
    }


def rank_candidates(query_material: dict, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score and rank all candidates. Filters disqualified. Returns top_k.
    candidates: list of {id, name, vector_score}
    """
    scored = []
    disqualified = []

    for c in candidates:
        result = score_candidate(query_material, c["id"], c.get("vector_score", 0.5))
        if result is None:
            continue
        if result["disqualified"]:
            disqualified.append({**c, **result})
        else:
            scored.append({**c, **result})

    scored.sort(key=lambda x: x["scores"]["total"], reverse=True)
    return scored[:top_k], disqualified
