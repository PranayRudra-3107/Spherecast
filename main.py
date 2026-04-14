"""
main.py
-------
Agnes Raw-Material Intelligence API — FastAPI application.
Powered by HuggingFace Inference API (Gemma + sentence-transformers).

Endpoints:
  POST /ingest/text          — ingest raw spec text
  POST /ingest/pdf           — ingest a PDF spec sheet
  GET  /materials            — list all materials
  GET  /materials/{id}       — get material details
  POST /substitutes          — find and rank substitutes
  POST /explain              — natural-language explanation
  POST /build-index          — rebuild FAISS vector index
  GET  /health               — health + HF API connectivity check
"""

import os
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

from knowledge_store import (
    init_db, get_material_full, list_all_materials,
    add_feedback, list_feedback, list_audit_log, get_graph_neighbors,
)
from ingestor import ingest_text, ingest_pdf, ingest_url, ingest_bom_csv, HF_TOKEN, HF_LLM_MODEL
from vector_index import search_by_material_id, search, build_index, EMBED_MODEL
from scorer import rank_candidates
from compliance_engine import check_compliance, batch_check

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agnes — Raw Material Intelligence API",
    description="Substitute component finder. Powered by HuggingFace + FAISS.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    print(f"[Agnes] Started. LLM={HF_LLM_MODEL}, Embeddings={EMBED_MODEL}, HF_TOKEN={'set' if HF_TOKEN else 'NOT SET'}")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class IngestTextRequest(BaseModel):
    text: str
    source_name: str = "api_input"
    price_usd: Optional[float] = None
    moq: Optional[int] = None
    lead_time_days: Optional[int] = None


class SubstituteRequest(BaseModel):
    material_id: Optional[str] = None
    query_text: Optional[str] = None
    top_k: int = 5
    require_same_category: bool = False


class ExplainRequest(BaseModel):
    query_material_id: str
    substitute_material_id: str
    scores: Optional[dict] = None


class AgentRequest(BaseModel):
    material_id: Optional[str] = None
    query_text: Optional[str] = None
    top_k: int = 5


class IngestUrlRequest(BaseModel):
    url: str
    source_name: Optional[str] = None
    price_usd: Optional[float] = None
    moq: Optional[int] = None
    lead_time_days: Optional[int] = None


class IngestBomRequest(BaseModel):
    csv_text: str
    source_name: str = "bom_import"


class FeedbackRequest(BaseModel):
    query_material_id: str
    substitute_material_id: str
    approved: bool
    comment: Optional[str] = None
    score_override: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Check API health and HuggingFace connectivity."""
    hf_ok = False
    error = None
    try:
        client = InferenceClient(token=HF_TOKEN)
        client.feature_extraction("test", model=EMBED_MODEL)
        hf_ok = True
    except Exception as e:
        error = str(e)

    return {
        "status": "ok",
        "version": "0.3.0",
        "huggingface": {
            "connected": hf_ok,
            "token_set": bool(HF_TOKEN),
            "llm_model": HF_LLM_MODEL,
            "embed_model": EMBED_MODEL,
            **({"error": error} if error else {}),
        },
    }


@app.get("/materials")
def list_materials_endpoint():
    return {"materials": list_all_materials()}


@app.get("/materials/{material_id}")
def get_material(material_id: str):
    mat = get_material_full(material_id)
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    return mat


@app.post("/ingest/text")
def ingest_text_endpoint(req: IngestTextRequest):
    try:
        result = ingest_text(
            req.text,
            source_type="api",
            source_name=req.source_name,
            price_usd=req.price_usd,
            moq=req.moq,
            lead_time_days=req.lead_time_days,
        )
        try:
            build_index()
        except Exception as e:
            print(f"[Agnes] Index rebuild warning: {e}")
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    tmp_path = Path(f"/tmp/{file.filename}")
    try:
        tmp_path.write_bytes(await file.read())
        result = ingest_pdf(str(tmp_path))
        try:
            build_index()
        except Exception as e:
            print(f"[Agnes] Index rebuild warning: {e}")
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/substitutes")
def find_substitutes(req: SubstituteRequest):
    """Find and rank substitutes for a material."""
    if not req.material_id and not req.query_text:
        raise HTTPException(status_code=400, detail="Provide material_id or query_text")

    # Resolve query material
    if req.material_id:
        query_material = get_material_full(req.material_id)
        if not query_material:
            raise HTTPException(status_code=404, detail="Query material not found")
        candidates_raw = search_by_material_id(req.material_id, top_k=req.top_k * 3)
    else:
        query_material = {
            "id": "query",
            "name": req.query_text,
            "category": None,
            "specs": [],
            "compliance": [],
            "price": None,
        }
        candidates_raw = search(req.query_text, top_k=req.top_k * 3)

    # Optional category filter
    if req.require_same_category and query_material.get("category"):
        from knowledge_store import get_conn
        with get_conn() as conn:
            candidates_raw = [
                c for c in candidates_raw
                if conn.execute(
                    "SELECT category FROM materials WHERE id = ?", (c["id"],)
                ).fetchone()["category"] == query_material["category"]
            ]

    if not candidates_raw:
        return {
            "query_material": query_material.get("name"),
            "substitutes": [],
            "message": "No candidates found. Ingest more materials or rebuild the index.",
        }

    ranked, disqualified = rank_candidates(query_material, candidates_raw, top_k=req.top_k)

    results = []
    for r in ranked:
        mat = get_material_full(r["id"])
        results.append({
            "rank": len(results) + 1,
            "material_id": r["id"],
            "name": r["name"],
            "supplier": mat.get("supplier"),
            "part_number": mat.get("part_number"),
            "scores": r["scores"],
            "score_details": r["details"],
            "price": mat.get("price"),
            "compliance_summary": {
                c["standard"]: c["status"] for c in mat.get("compliance", [])
            },
        })

    return {
        "query_material_id": req.material_id,
        "query_material_name": query_material.get("name"),
        "substitutes": results,
        "disqualified_count": len(disqualified),
        "disqualified": [
            {"name": d["name"], "reason": d.get("disqualification_reason")}
            for d in disqualified
        ],
    }


@app.post("/explain")
def explain_substitution(req: ExplainRequest):
    """Generate a natural-language explanation via Gemma on HuggingFace."""
    query_mat = get_material_full(req.query_material_id)
    sub_mat   = get_material_full(req.substitute_material_id)

    if not query_mat:
        raise HTTPException(status_code=404, detail="Query material not found")
    if not sub_mat:
        raise HTTPException(status_code=404, detail="Substitute material not found")

    def mat_summary(mat):
        specs_str = "\n".join(
            f"  {s['attribute']}: {s['value']} {s['unit'] or ''}"
            for s in mat.get("specs", []) if s.get("value") is not None
        )
        comp_str  = ", ".join(f"{c['standard']}={c['status']}" for c in mat.get("compliance", []))
        price     = mat.get("price")
        price_str = f"${price['price_usd']}/kg, MOQ {price['moq']}, {price['lead_time_days']} days" if price else "unknown"
        return (
            f"Name: {mat['name']}\n"
            f"Supplier: {mat.get('supplier', 'N/A')}\n"
            f"Category: {mat.get('category', 'N/A')}\n"
            f"Specs:\n{specs_str}\n"
            f"Compliance: {comp_str or 'none recorded'}\n"
            f"Price: {price_str}"
        )

    scores_text = ""
    if req.scores:
        scores_text = (
            f"\nScoring summary:\n"
            f"  Overall: {req.scores.get('total', 'N/A')}\n"
            f"  Spec similarity: {req.scores.get('spec_similarity', 'N/A')} (40%)\n"
            f"  Compliance: {req.scores.get('compliance', 'N/A')} (20%)\n"
            f"  Price/Lead time: {req.scores.get('price_lead_time', 'N/A')} (15%)\n"
            f"  Quality: {req.scores.get('quality', 'N/A')} (15%)\n"
            f"  Business priority: {req.scores.get('business_priority', 'N/A')} (10%)"
        )

    prompt = (
        "You are Agnes, an expert supply-chain materials intelligence assistant.\n"
        "Write a clear, concise, technically accurate explanation (3-5 sentences) "
        "of why this substitution is or is not recommended.\n"
        "Focus on key spec differences, compliance concerns, and cost/lead-time trade-offs.\n"
        "Be specific about numbers. Flag any risks clearly.\n\n"
        f"ORIGINAL MATERIAL:\n{mat_summary(query_mat)}\n\n"
        f"PROPOSED SUBSTITUTE:\n{mat_summary(sub_mat)}\n"
        f"{scores_text}\n\n"
        "Write your explanation:"
    )

    try:
        client = InferenceClient(HF_LLM_MODEL, token=HF_TOKEN)
        resp = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        explanation = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"HuggingFace Inference API error: {str(e)}")

    return {
        "query_material": query_mat["name"],
        "substitute_material": sub_mat["name"],
        "explanation": explanation,
        "scores": req.scores,
        "evidence_paths": [
            {"source_type": p["source_type"], "source_name": p["source_name"], "ingested_at": p["ingested_at"]}
            for p in sub_mat.get("provenance", [])
        ],
        "compliance_flags": {
            c["standard"]: {"status": c["status"], "notes": c.get("notes")}
            for c in sub_mat.get("compliance", [])
        },
    }


@app.post("/agent/find")
def agent_find(req: AgentRequest):
    """
    Agentic multi-step substitute finder.
    Runs: search → compliance filter → scoring → LLM explanation → audit log.
    Returns ranked substitutes + step-by-step agent_trace.
    """
    if not req.material_id and not req.query_text:
        raise HTTPException(status_code=400, detail="Provide material_id or query_text")
    try:
        from agent import SubstituteAgent
        ag = SubstituteAgent()
        result = ag.run(
            material_id=req.material_id,
            query_text=req.query_text,
            top_k=req.top_k,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/{material_id}")
def compliance_report(material_id: str):
    """
    Full regulatory compliance report for a material.
    Covers RoHS, REACH SVHC, Conflict Minerals (3TG), FDA food contact.
    """
    mat = get_material_full(material_id)
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    return check_compliance(material_id)


@app.post("/compliance/batch")
def compliance_batch(body: dict):
    """Compliance check for a list of material IDs."""
    ids = body.get("material_ids", [])
    if not ids:
        raise HTTPException(status_code=400, detail="Provide material_ids list")
    return batch_check(ids)


@app.post("/ingest/url")
def ingest_url_endpoint(req: IngestUrlRequest):
    """Fetch a supplier/spec webpage and ingest its content via Gemma extraction."""
    try:
        result = ingest_url(
            req.url,
            source_name=req.source_name,
            price_usd=req.price_usd,
            moq=req.moq,
            lead_time_days=req.lead_time_days,
        )
        try:
            build_index()
        except Exception as e:
            print(f"[Agnes] Index rebuild warning: {e}")
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/bom")
def ingest_bom_endpoint(req: IngestBomRequest):
    """
    Ingest a Bill-of-Materials CSV.
    Expected columns: name, part_number, supplier, category, tensile_strength,
    density, price_usd, moq, lead_time_days, rohs, reach, fda, notes.
    """
    try:
        results = ingest_bom_csv(req.csv_text, default_source=req.source_name)
        try:
            build_index()
        except Exception as e:
            print(f"[Agnes] Index rebuild warning: {e}")
        ok  = [r for r in results if "error" not in r]
        err = [r for r in results if "error" in r]
        return {"ingested": len(ok), "failed": len(err), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit approval/rejection feedback for a substitution suggestion."""
    query_mat = get_material_full(req.query_material_id)
    sub_mat   = get_material_full(req.substitute_material_id)
    if not query_mat:
        raise HTTPException(status_code=404, detail="Query material not found")
    if not sub_mat:
        raise HTTPException(status_code=404, detail="Substitute material not found")
    fid = add_feedback(
        query_material_id=req.query_material_id,
        substitute_material_id=req.substitute_material_id,
        approved=req.approved,
        comment=req.comment,
        score_override=req.score_override,
    )
    return {"feedback_id": fid, "status": "recorded",
            "approved": req.approved, "comment": req.comment}


@app.get("/feedback")
def get_feedback(limit: int = 100):
    """List recent user feedback entries."""
    return {"feedback": list_feedback(limit=limit)}


@app.get("/audit-log")
def get_audit_log(limit: int = 50):
    """Return the audit trail of agent queries and ingestion events."""
    return {"audit_log": list_audit_log(limit=limit)}


@app.get("/graph/neighbors/{material_id}")
def graph_neighbors(material_id: str, edge_type: Optional[str] = None):
    """Return all KG edges connected to a material node."""
    mat = get_material_full(material_id)
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    edges = get_graph_neighbors(material_id, edge_type=edge_type)
    return {"material_id": material_id, "material_name": mat["name"], "edges": edges}


@app.post("/build-index")
def rebuild_index():
    """Rebuild the FAISS vector index over all materials."""
    try:
        build_index()
        return {"status": "ok", "materials_indexed": len(list_all_materials())}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
