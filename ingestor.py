"""
ingestor.py
-----------
Ingests raw material spec text (or PDF text) and uses HuggingFace Inference API
(google/gemma-4-26B-A4B-it) to extract structured specs, compliance hints, and metadata.
"""

import json
import re
import os
from typing import Optional
from huggingface_hub import InferenceClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from knowledge_store import (
    upsert_material,
    insert_specs,
    insert_compliance,
    insert_provenance,
    insert_price,
    init_db,
)

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_LLM_MODEL = os.environ.get("AGNES_LLM_MODEL", "google/gemma-4-26B-A4B-it")

EXTRACTION_SYSTEM = """You are a materials science data extraction engine for a supply-chain intelligence system.

Given raw text from a material specification sheet, extract structured data.

RULES:
- Convert all units to SI base: MPa (not psi/ksi), mm (not inches), g/cm3 (not lb/in3), degC (not degF).
- If a value is a range (e.g. "120-140 MPa"), use the midpoint as value and note the range in raw_value.
- If a spec is missing, omit it from the specs array. Never invent values.
- For compliance, only assert "pass" if explicitly stated. Otherwise use "unknown".
- Confidence: 1.0 = explicitly stated, 0.7 = inferred, 0.5 = uncertain.
- Return ONLY valid JSON. No markdown fences, no explanation, no extra text before or after.

Output schema:
{
  "name": "material name",
  "part_number": "part number or null",
  "category": "metal|polymer|chemical|ceramic|composite|other",
  "supplier": "supplier name or null",
  "specs": [
    {"attribute": "tensile_strength", "value": 250.0, "unit": "MPa", "raw_value": "250 MPa", "confidence": 1.0}
  ],
  "compliance": [
    {"standard": "RoHS", "status": "pass|fail|unknown", "notes": ""},
    {"standard": "REACH", "status": "pass|fail|unknown", "notes": ""},
    {"standard": "FDA", "status": "pass|fail|unknown", "notes": ""}
  ],
  "price_usd": null,
  "moq": null,
  "lead_time_days": null
}

Common spec attributes to extract when present:
tensile_strength, yield_strength, elongation_at_break, hardness_vickers,
density, melting_point, glass_transition_temp, thermal_conductivity,
coefficient_thermal_expansion, elastic_modulus, flexural_strength,
max_operating_temp, purity_percent, molecular_weight"""


def _call_hf(prompt: str, system: str) -> str:
    """Call HuggingFace Inference API via InferenceClient. Returns raw text response."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable not set.\n"
            "Fix: export HF_TOKEN=hf_your_token_here"
        )
    try:
        client = InferenceClient(HF_LLM_MODEL, token=HF_TOKEN)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"HuggingFace Inference API error: {e}")


def _extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM output, handling fences and leading text."""
    raw = re.sub(r"```json\s*|```\s*", "", raw, flags=re.IGNORECASE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find outermost { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response.\n"
        f"First 400 chars: {raw[:400]}"
    )


def extract_specs_from_text(raw_text: str) -> dict:
    """Send raw spec text to Gemma via HF Inference API for structured extraction."""
    prompt = (
        "Extract structured material data from the specification text below. "
        "Return ONLY valid JSON — no explanation, no markdown.\n\n"
        f"SPECIFICATION TEXT:\n{raw_text}"
    )
    response = _call_hf(prompt, EXTRACTION_SYSTEM)
    return _extract_json(response)


def ingest_text(
    raw_text: str,
    source_type: str = "manual",
    source_name: str = "manual_input",
    price_usd: Optional[float] = None,
    moq: Optional[int] = None,
    lead_time_days: Optional[int] = None,
) -> dict:
    """Full pipeline: extract → store → return summary."""
    print(f"[Ingestor] Extracting via {HF_LLM_MODEL}: {source_name}")
    extracted = extract_specs_from_text(raw_text)

    material_id = upsert_material(
        name=extracted["name"],
        part_number=extracted.get("part_number"),
        category=extracted.get("category"),
        supplier=extracted.get("supplier"),
    )

    insert_specs(material_id, extracted.get("specs", []))
    insert_compliance(material_id, extracted.get("compliance", []))
    insert_provenance(material_id, source_type, source_name, raw_text[:2000])

    _price = price_usd or extracted.get("price_usd")
    _moq   = moq or extracted.get("moq") or 1
    _lt    = lead_time_days or extracted.get("lead_time_days") or 30
    if _price:
        insert_price(material_id, float(_price), int(_moq), int(_lt))

    print(f"[Ingestor] Stored '{extracted['name']}' → {material_id}")
    return {
        "material_id": material_id,
        "name": extracted["name"],
        "specs_extracted": len(extracted.get("specs", [])),
        "compliance_records": len(extracted.get("compliance", [])),
    }


def ingest_url(url: str, source_name: str = None, **kwargs) -> dict:
    """
    Fetch a supplier/spec webpage, extract visible text, and ingest it.
    Requires: pip install beautifulsoup4 requests
    """
    try:
        import requests as _requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install dependencies: pip install beautifulsoup4 requests")

    headers = {"User-Agent": "Agnes-MaterialIntelligence/1.0 (supply-chain research bot)"}
    resp = _requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n", strip=True)
    # Trim to 8000 chars to stay within LLM context
    raw_text = raw_text[:8000]

    return ingest_text(
        raw_text,
        source_type="website",
        source_name=source_name or url,
        **kwargs,
    )


def ingest_bom_csv(csv_text: str, default_source: str = "bom_import") -> list[dict]:
    """
    Ingest a Bill-of-Materials CSV.
    Expected columns (flexible, any order): name, part_number, supplier,
    category, tensile_strength, density, price_usd, moq, lead_time_days,
    rohs, reach, fda, notes/description/spec_text.

    Returns list of ingest result dicts.
    """
    import io
    import csv

    reader = csv.DictReader(io.StringIO(csv_text))
    results = []

    for row in reader:
        # Build a spec text blob from the CSV row
        lines = []
        name = row.get("name") or row.get("material_name") or row.get("Name") or ""
        if not name.strip():
            continue

        lines.append(f"Material: {name}")
        for field in ["part_number", "supplier", "category"]:
            val = row.get(field) or row.get(field.replace("_", " ")) or ""
            if val.strip():
                lines.append(f"{field.replace('_', ' ').title()}: {val}")

        # Numeric specs
        for spec in ["tensile_strength", "yield_strength", "density", "elastic_modulus",
                     "max_operating_temp", "melting_point", "thermal_conductivity",
                     "elongation_at_break", "hardness_vickers", "purity_percent"]:
            val = row.get(spec) or row.get(spec.replace("_", " ")) or ""
            if val.strip():
                lines.append(f"{spec.replace('_', ' ').title()}: {val}")

        # Compliance hints
        for std in ["rohs", "reach", "fda"]:
            val = row.get(std) or row.get(std.upper()) or ""
            if val.strip():
                lines.append(f"{std.upper()} Compliant: {val}")

        # Free-text notes
        for note_col in ["notes", "description", "spec_text", "comments"]:
            val = row.get(note_col) or ""
            if val.strip():
                lines.append(val)

        spec_text = "\n".join(lines)

        # Parse numeric price/moq/lead_time
        def _num(key, typ=float):
            try:
                return typ(row.get(key, "") or 0) or None
            except (ValueError, TypeError):
                return None

        try:
            result = ingest_text(
                spec_text,
                source_type="erp_bom",
                source_name=default_source,
                price_usd=_num("price_usd"),
                moq=_num("moq", int),
                lead_time_days=_num("lead_time_days", int),
            )
            results.append(result)
        except Exception as e:
            results.append({"name": name, "error": str(e)})

    return results


def ingest_pdf(pdf_path: str, **kwargs) -> dict:
    """Extract text from a PDF and ingest it. Requires pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
            for table in page.extract_tables():
                for row in table:
                    text_parts.append(" | ".join(str(c) for c in row if c))

    raw_text = "\n".join(text_parts)
    return ingest_text(
        raw_text,
        source_type="pdf",
        source_name=os.path.basename(pdf_path),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Demo seed data
# ---------------------------------------------------------------------------

DEMO_MATERIALS = [
    {
        "text": """
        Material: Aluminum 6061-T6
        Part Number: AL-6061-T6
        Supplier: MetalsCo Inc.
        Category: Metal
        Tensile Strength: 310 MPa
        Yield Strength: 276 MPa
        Elongation at Break: 12%
        Density: 2.70 g/cm3
        Thermal Conductivity: 167 W/m·K
        Elastic Modulus: 68.9 GPa
        Melting Point: 617 degC
        Max Operating Temp: 150 degC
        Coefficient Thermal Expansion: 23.6 um/m·degC
        RoHS Compliant: Yes
        REACH Compliant: Yes
        Price: $4.20/kg, MOQ 50kg, Lead time 14 days
        """,
        "price_usd": 4.20, "moq": 50, "lead_time_days": 14,
    },
    {
        "text": """
        Material: Aluminum 7075-T6
        Part Number: AL-7075-T6
        Supplier: AeroAlloys Ltd.
        Category: Metal
        Tensile Strength: 572 MPa
        Yield Strength: 503 MPa
        Elongation at Break: 11%
        Density: 2.81 g/cm3
        Thermal Conductivity: 130 W/m·K
        Elastic Modulus: 71.7 GPa
        Melting Point: 556 degC
        Max Operating Temp: 120 degC
        Coefficient Thermal Expansion: 23.4 um/m·degC
        RoHS Compliant: Yes
        REACH Compliant: Yes
        Price: $6.80/kg, MOQ 25kg, Lead time 21 days
        """,
        "price_usd": 6.80, "moq": 25, "lead_time_days": 21,
    },
    {
        "text": """
        Material: Aluminum 2024-T4
        Part Number: AL-2024-T4
        Supplier: StructuralMetals Corp.
        Category: Metal
        Tensile Strength: 469 MPa
        Yield Strength: 324 MPa
        Elongation at Break: 19%
        Density: 2.78 g/cm3
        Thermal Conductivity: 121 W/m·K
        Elastic Modulus: 73.1 GPa
        Melting Point: 570 degC
        Max Operating Temp: 130 degC
        Coefficient Thermal Expansion: 23.2 um/m·degC
        RoHS Compliant: Yes
        REACH: Contains trace Cr(VI) - under investigation
        Price: $5.50/kg, MOQ 100kg, Lead time 28 days
        """,
        "price_usd": 5.50, "moq": 100, "lead_time_days": 28,
    },
    {
        "text": """
        Material: Stainless Steel 316L
        Part Number: SS-316L
        Supplier: SteelWorks Global
        Category: Metal
        Tensile Strength: 485 MPa
        Yield Strength: 170 MPa
        Elongation at Break: 40%
        Density: 8.0 g/cm3
        Thermal Conductivity: 16.3 W/m·K
        Elastic Modulus: 193 GPa
        Melting Point: 1385 degC
        Max Operating Temp: 870 degC
        Coefficient Thermal Expansion: 15.9 um/m·degC
        RoHS Compliant: Yes
        REACH Compliant: Yes
        FDA Approved for food contact: Yes
        Price: $8.90/kg, MOQ 200kg, Lead time 35 days
        """,
        "price_usd": 8.90, "moq": 200, "lead_time_days": 35,
    },
    {
        "text": """
        Material: Titanium Grade 5 (Ti-6Al-4V)
        Part Number: TI-6AL4V-G5
        Supplier: TitaniumPro Ltd.
        Category: Metal
        Tensile Strength: 950 MPa
        Yield Strength: 880 MPa
        Elongation at Break: 14%
        Density: 4.43 g/cm3
        Thermal Conductivity: 6.7 W/m·K
        Elastic Modulus: 113.8 GPa
        Melting Point: 1632 degC
        Max Operating Temp: 300 degC
        Coefficient Thermal Expansion: 8.6 um/m·degC
        RoHS Compliant: Yes
        REACH Compliant: Yes
        Price: $42.00/kg, MOQ 10kg, Lead time 45 days
        """,
        "price_usd": 42.00, "moq": 10, "lead_time_days": 45,
    },
    {
        "text": """
        Material: Nylon 66 (Polyamide 66)
        Part Number: PA66-NAT
        Supplier: PolymerSource Inc.
        Category: Polymer
        Tensile Strength: 82 MPa
        Yield Strength: 82 MPa
        Elongation at Break: 60%
        Density: 1.14 g/cm3
        Thermal Conductivity: 0.23 W/m·K
        Elastic Modulus: 2.8 GPa
        Glass Transition Temp: 70 degC
        Max Operating Temp: 120 degC
        Melting Point: 255 degC
        RoHS Compliant: Yes
        REACH Compliant: Yes
        FDA food contact: Yes
        Price: $3.20/kg, MOQ 500kg, Lead time 21 days
        """,
        "price_usd": 3.20, "moq": 500, "lead_time_days": 21,
    },
    {
        "text": """
        Material: PEEK (Polyether Ether Ketone)
        Part Number: PEEK-450G
        Supplier: HighPerf Polymers
        Category: Polymer
        Tensile Strength: 100 MPa
        Yield Strength: 91 MPa
        Elongation at Break: 40%
        Density: 1.32 g/cm3
        Thermal Conductivity: 0.25 W/m·K
        Elastic Modulus: 3.6 GPa
        Glass Transition Temp: 143 degC
        Max Operating Temp: 260 degC
        Melting Point: 343 degC
        RoHS Compliant: Yes
        REACH Compliant: Yes
        Price: $89.00/kg, MOQ 25kg, Lead time 30 days
        """,
        "price_usd": 89.00, "moq": 25, "lead_time_days": 30,
    },
    {
        "text": """
        Material: PTFE (Teflon)
        Part Number: PTFE-GR-VIRGIN
        Supplier: FluoroPoly Corp.
        Category: Polymer
        Tensile Strength: 31 MPa
        Elongation at Break: 400%
        Density: 2.15 g/cm3
        Thermal Conductivity: 0.25 W/m·K
        Elastic Modulus: 0.5 GPa
        Max Operating Temp: 260 degC
        Melting Point: 327 degC
        Chemical Resistance: Excellent
        RoHS Compliant: Yes
        REACH Compliant: Yes
        FDA Approved: Yes
        Price: $18.50/kg, MOQ 100kg, Lead time 25 days
        """,
        "price_usd": 18.50, "moq": 100, "lead_time_days": 25,
    },
]


def seed_demo_data():
    """Populate the DB with sample materials."""
    init_db()
    print(f"[Ingestor] Seeding {len(DEMO_MATERIALS)} materials via {HF_LLM_MODEL}...")
    print("[Ingestor] Estimated time: ~2-4 min (LLM inference per material)\n")
    for i, mat in enumerate(DEMO_MATERIALS, 1):
        print(f"  [{i}/{len(DEMO_MATERIALS)}] Ingesting...")
        result = ingest_text(
            mat["text"],
            source_type="demo",
            source_name="demo_seed",
            price_usd=mat.get("price_usd"),
            moq=mat.get("moq"),
            lead_time_days=mat.get("lead_time_days"),
        )
        print(f"  ✓ {result['name']} ({result['specs_extracted']} specs)")
    print("\n[Ingestor] Seed complete.")


if __name__ == "__main__":
    seed_demo_data()
