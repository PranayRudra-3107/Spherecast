"""
compliance_engine.py
--------------------
Rule-based regulatory compliance engine for Agnes.
Covers: RoHS 2, REACH SVHC, Conflict Minerals (3TG / Dodd-Frank), FDA food contact.
No LLM required — all deterministic text-scanning against provenance records.

Design (from Executive Summary):
  "Tools should flag if a substitute lacks required certifications."
  "Compliance is a hard gate: any fail = disqualified."
"""

import re
from knowledge_store import get_material_full, get_conn

# ---------------------------------------------------------------------------
# Regulatory databases (representative subsets)
# ---------------------------------------------------------------------------

# RoHS 2 (EU Directive 2011/65/EU + amendment 2015/863/EU) — 10 restricted substances
ROHS_RESTRICTED = {
    "lead":                           {"cas": "7439-92-1",  "max_ppm": 1000, "symbol": "pb"},
    "mercury":                        {"cas": "7439-97-6",  "max_ppm": 1000, "symbol": "hg"},
    "cadmium":                        {"cas": "7440-43-9",  "max_ppm": 100,  "symbol": "cd"},
    "hexavalent chromium":            {"cas": "18540-29-9", "max_ppm": 1000, "symbol": "cr(vi)"},
    "polybrominated biphenyls":       {"cas": "various",    "max_ppm": 1000, "symbol": "pbb"},
    "polybrominated diphenyl ethers": {"cas": "various",    "max_ppm": 1000, "symbol": "pbde"},
    "bis(2-ethylhexyl) phthalate":    {"cas": "117-81-7",   "max_ppm": 1000, "symbol": "dehp"},
    "benzyl butyl phthalate":         {"cas": "85-68-7",    "max_ppm": 1000, "symbol": "bbp"},
    "dibutyl phthalate":              {"cas": "84-74-2",    "max_ppm": 1000, "symbol": "dbp"},
    "diisobutyl phthalate":           {"cas": "84-69-5",    "max_ppm": 1000, "symbol": "dibp"},
}

# REACH — Substances of Very High Concern (sample; official EU list has 240+)
REACH_SVHC = [
    {"name": "chromium trioxide",      "cas": "1333-82-0"},
    {"name": "arsenic trioxide",       "cas": "1327-53-3"},
    {"name": "asbestos",               "cas": "various"},
    {"name": "benzene",                "cas": "71-43-2"},
    {"name": "formaldehyde",           "cas": "50-00-0"},
    {"name": "boric acid",             "cas": "10043-35-3"},
    {"name": "sodium dichromate",      "cas": "10588-01-9"},
    {"name": "ammonium dichromate",    "cas": "7789-09-5"},
    {"name": "cobalt sulphate",        "cas": "10124-43-3"},
    {"name": "lead chromate",          "cas": "7758-97-6"},
    {"name": "nickel compounds",       "cas": "various"},
    {"name": "trichloroethylene",      "cas": "79-01-6"},
    {"name": "perchloroethylene",      "cas": "127-18-4"},
    {"name": "dibutyltin dilaurate",   "cas": "77-58-7"},
]

# Conflict Minerals — Dodd-Frank Section 1502 (3TG from DRC conflict zones)
CONFLICT_MINERALS = {
    "tin":      ["tin", " sn ", "cassiterite", "stannite"],
    "tantalum": ["tantalum", " ta ", "coltan", "columbite-tantalite"],
    "tungsten": ["tungsten", " w ", "wolframite", "scheelite", "wolfram"],
    "gold":     ["gold", " au ", "aurum"],
}

# FDA 21 CFR — substances requiring explicit food-contact clearance
FDA_CLEARANCE_REQUIRED = [
    "bisphenol a", "bpa", "styrene monomer", "acrylonitrile", "vinyl chloride",
    "melamine", "epoxy resin", "pvc additive", "phthalate plasticizer",
]

# Negation contexts that cancel a substance match (reduce false positives)
NEGATION_CONTEXT = [
    "free", "compliant", "no ", "not ", "without", "exempt", "below",
    "< ", "≤ ", "lead-free", "cadmium-free", "mercury-free",
    "rohs compliant", "reach compliant", "rohs: pass", "reach: pass",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_substance(text: str, term: str) -> bool:
    """Return True if term appears in text outside a negation context."""
    idx = text.find(term)
    if idx == -1:
        return False
    window_start = max(0, idx - 70)
    window_end   = min(len(text), idx + len(term) + 70)
    window       = text[window_start:window_end]
    return not any(neg in window for neg in NEGATION_CONTEXT)


def _get_provenance_text(material_id: str) -> str:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT raw_text FROM provenance WHERE material_id = ?", (material_id,)
        ).fetchall()
    return " ".join(r["raw_text"] or "" for r in rows).lower()


# ---------------------------------------------------------------------------
# Core compliance checker
# ---------------------------------------------------------------------------

def check_compliance(material_id: str) -> dict:
    """
    Run full regulatory compliance analysis.
    Returns a structured report with risk_level, disqualified flag, and evidence.

    Fields returned:
      material_id, material_name, risk_level (LOW|MEDIUM|HIGH|CRITICAL),
      disqualified (bool), compliance_score (0-1), traceability_score (0-1),
      passed (list), flags (list of CRITICAL/HIGH issues),
      warnings (list of INFO/WARNING notices),
      conflict_minerals (list of detected 3TG metals),
      existing_declarations (dict standard→status).
    """
    mat = get_material_full(material_id)
    if not mat:
        return {"error": "Material not found", "material_id": material_id,
                "disqualified": False, "compliance_score": 0.5}

    text     = _get_provenance_text(material_id)
    existing = {c["standard"]: c["status"] for c in mat.get("compliance", [])}
    category = (mat.get("category") or "").lower()

    flags    = []   # severity CRITICAL or HIGH — used in disqualification logic
    warnings = []   # severity WARNING or INFO
    passed   = []   # standards explicitly confirmed OK

    # ── RoHS ─────────────────────────────────────────────────────────────────
    rohs = existing.get("RoHS", "unknown")
    if rohs == "pass":
        passed.append({"standard": "RoHS", "message": "Explicitly declared RoHS compliant."})
    elif rohs == "fail":
        flags.append({"standard": "RoHS", "severity": "CRITICAL",
                      "message": "Material explicitly FAILS RoHS — contains restricted substance."})
    else:
        detected = [
            s for s, info in ROHS_RESTRICTED.items()
            if _has_substance(text, s) or _has_substance(text, info["symbol"])
        ]
        if detected:
            warnings.append({"standard": "RoHS", "severity": "WARNING",
                              "message": f"Possible restricted substances in spec text: "
                                         f"{', '.join(detected)}. Verify RoHS declaration."})
        else:
            warnings.append({"standard": "RoHS", "severity": "INFO",
                              "message": "RoHS status not declared — assume unknown."})

    # ── REACH ────────────────────────────────────────────────────────────────
    reach = existing.get("REACH", "unknown")
    if reach == "pass":
        passed.append({"standard": "REACH", "message": "Explicitly declared REACH compliant."})
    elif reach == "fail":
        flags.append({"standard": "REACH", "severity": "CRITICAL",
                      "message": "Material explicitly FAILS REACH — SVHC present above 0.1% w/w threshold."})
    else:
        svhc = [s["name"] for s in REACH_SVHC if s["name"] in text]
        if svhc:
            flags.append({"standard": "REACH", "severity": "HIGH",
                           "message": f"REACH SVHC detected in provenance text: {', '.join(svhc)}"})
        else:
            warnings.append({"standard": "REACH", "severity": "INFO",
                              "message": "REACH status not declared."})

    # ── Conflict Minerals (3TG / Dodd-Frank §1502) ───────────────────────────
    conflict_free = any(p in text for p in
                        ["conflict-free", "conflict free", "drc conflict free", "3tg free"])
    cm_detected = {}
    for mineral, aliases in CONFLICT_MINERALS.items():
        if any(a in text for a in aliases) and not conflict_free:
            cm_detected[mineral] = True
    if cm_detected:
        warnings.append({
            "standard": "Conflict Minerals (3TG)",
            "severity": "WARNING",
            "message": f"Contains {list(cm_detected.keys())}. "
                       "Dodd-Frank Section 1502 conflict-minerals declaration may be required.",
        })
    elif any(a in text for mineral_aliases in CONFLICT_MINERALS.values() for a in mineral_aliases):
        passed.append({"standard": "Conflict Minerals", "message": "Conflict-free declaration found."})

    # ── FDA food-contact (applicable to chemicals, polymers, or food-related text) ──
    needs_fda = category in ("chemical", "polymer", "composite") or "food" in text or "fda" in text
    fda = existing.get("FDA", "unknown")
    if needs_fda:
        if fda == "pass":
            passed.append({"standard": "FDA", "message": "FDA food-contact approval declared."})
        elif fda == "fail":
            flags.append({"standard": "FDA", "severity": "CRITICAL",
                           "message": "Fails FDA food-contact requirements."})
        else:
            fda_subs = [s for s in FDA_CLEARANCE_REQUIRED if s in text]
            if fda_subs:
                flags.append({"standard": "FDA", "severity": "HIGH",
                               "message": f"FDA-clearance-required substances detected: {fda_subs}"})
            else:
                warnings.append({"standard": "FDA", "severity": "INFO",
                                  "message": "FDA food-contact status not declared."})

    # ── California Prop 65 hint ───────────────────────────────────────────────
    if "prop 65" in text or "proposition 65" in text:
        if "compliant" in text or "no warning required" in text:
            passed.append({"standard": "CA Prop 65",
                           "message": "Prop 65 compliance declared."})
        else:
            warnings.append({"standard": "CA Prop 65", "severity": "WARNING",
                              "message": "Prop 65 mentioned in spec — verify warning requirements."})

    # ── Scoring ───────────────────────────────────────────────────────────────
    critical_count = sum(1 for f in flags if f["severity"] == "CRITICAL")
    high_count     = sum(1 for f in flags if f["severity"] == "HIGH")
    disqualified   = critical_count > 0

    risk = (
        "CRITICAL" if critical_count > 0 else
        "HIGH"     if high_count > 0     else
        "MEDIUM"   if warnings           else
        "LOW"
    )

    total_applicable = 3 + (1 if needs_fda else 0)
    compliance_score = len(passed) / total_applicable if total_applicable > 0 else 0.5
    if disqualified:
        compliance_score = 0.0
    elif high_count > 0:
        compliance_score = min(compliance_score, 0.4)

    # Traceability: ratio of provenance records to target (3 = good)
    with get_conn() as conn:
        prov_count = conn.execute(
            "SELECT COUNT(*) as c FROM provenance WHERE material_id=?", (material_id,)
        ).fetchone()["c"]
    traceability_score = min(1.0, prov_count / 3.0)

    return {
        "material_id":          material_id,
        "material_name":        mat["name"],
        "risk_level":           risk,
        "disqualified":         disqualified,
        "compliance_score":     round(compliance_score, 3),
        "traceability_score":   round(traceability_score, 3),
        "passed":               passed,
        "flags":                flags,
        "warnings":             warnings,
        "conflict_minerals":    list(cm_detected.keys()),
        "existing_declarations": existing,
    }


def get_compliance_score(material_id: str) -> float:
    """Quick 0–1 compliance score for use in scorer.py."""
    result = check_compliance(material_id)
    return result.get("compliance_score", 0.5) if "error" not in result else 0.5


def batch_check(material_ids: list) -> dict:
    """Run check_compliance for a list of IDs. Returns {id: report}."""
    return {mid: check_compliance(mid) for mid in material_ids}
