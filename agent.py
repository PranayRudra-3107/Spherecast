"""
agent.py
--------
Agnes Substitute-Finder Agent.

Multi-step agentic orchestration (from the Executive Summary):
  "Build agentic workflows that gather and connect information from supplier
   websites, spec sheets, and internal procurement records."

Pipeline:
  1. Resolve query material (ID or free text)
  2. Search — vector similarity + graph neighbor traversal
  3. Compliance pre-filter — compliance_engine hard gate
  4. Score & rank — scorer multi-criteria engine
  5. Explain top result — LLM narrative (Gemma via HF)
  6. Audit log — write to knowledge store

Returns a rich result dict including agent_trace for UI step visualisation.
"""

import os
import json
from datetime import datetime
from huggingface_hub import InferenceClient

HF_TOKEN    = os.environ.get("HF_TOKEN")
HF_LLM_MODEL = os.environ.get("AGNES_LLM_MODEL", "google/gemma-4-26B-A4B-it")


class SubstituteAgent:
    """
    Deterministic multi-step agent for substitute finding.
    Each step is explicit, traceable, and produces structured output.
    """

    def __init__(self):
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set. Run: export HF_TOKEN=hf_...")
        self.client = InferenceClient(HF_LLM_MODEL, token=HF_TOKEN)
        self.trace: list[dict] = []

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def _log(self, step: int, action: str, detail: str, data: dict = None):
        self.trace.append({
            "step":   step,
            "action": action,
            "detail": detail,
            "data":   data or {},
            "ts":     datetime.utcnow().isoformat(),
        })

    # ------------------------------------------------------------------
    # Step 1 — Resolve query material
    # ------------------------------------------------------------------

    def _resolve(self, material_id: str = None, query_text: str = None):
        from knowledge_store import get_material_full
        if material_id:
            mat = get_material_full(material_id)
            if mat:
                self._log(1, "resolve_material",
                          f"Resolved material from DB: {mat['name']}")
                return mat, mat["name"]

        stub = {
            "id": "query_stub", "name": query_text or "Unknown",
            "specs": [], "compliance": [], "category": None, "price": None,
        }
        self._log(1, "resolve_text", f"Free-text query: {query_text}")
        return stub, (query_text or "Unknown")

    # ------------------------------------------------------------------
    # Step 2 — Search (vector + graph neighbours)
    # ------------------------------------------------------------------

    def _search(self, material_id: str = None, query_text: str = None) -> list:
        from vector_index import search_by_material_id, search
        from knowledge_store import get_graph_neighbors, list_all_materials

        candidates = []
        seen_ids   = set()

        # Vector / FAISS search
        try:
            if material_id:
                vec_results = search_by_material_id(material_id, top_k=20)
            else:
                vec_results = search(query_text or "", top_k=20)

            for c in vec_results:
                if c["id"] not in seen_ids:
                    candidates.append(c)
                    seen_ids.add(c["id"])
            self._log(2, "vector_search",
                      f"Vector search returned {len(vec_results)} candidates")
        except Exception as e:
            self._log(2, "vector_search_error", str(e))

        # Graph neighbour expansion (substitutedBy / relatedTo edges)
        if material_id:
            try:
                edges = get_graph_neighbors(material_id, edge_type="substitutedBy")
                edges += get_graph_neighbors(material_id, edge_type="relatedTo")
                graph_ids = {
                    e["target_id"] if e["source_id"] == material_id else e["source_id"]
                    for e in edges
                }
                for gid in graph_ids:
                    if gid not in seen_ids and gid != material_id:
                        candidates.append({"id": gid, "name": gid, "vector_score": 0.6,
                                           "source": "graph_edge"})
                        seen_ids.add(gid)
                if graph_ids:
                    self._log(2, "graph_expansion",
                              f"Graph expansion added {len(graph_ids)} neighbours")
            except Exception as e:
                self._log(2, "graph_expansion_error", str(e))

        # Exclude the query material itself
        candidates = [c for c in candidates if c["id"] != material_id]
        self._log(2, "search_complete",
                  f"Total unique candidates: {len(candidates)}",
                  {"count": len(candidates)})
        return candidates

    # ------------------------------------------------------------------
    # Step 3 — Compliance pre-filter
    # ------------------------------------------------------------------

    def _compliance_filter(self, candidates: list) -> tuple[list, list]:
        from compliance_engine import check_compliance

        passed       = []
        disqualified = []

        for c in candidates:
            report = check_compliance(c["id"])
            c["_compliance"] = report
            if report.get("disqualified"):
                reasons = [f["message"] for f in report.get("flags", [])
                           if f["severity"] == "CRITICAL"]
                disqualified.append({
                    **c,
                    "disqualification_reason": "; ".join(reasons) or "Compliance failure",
                })
                self._log(3, "disqualified",
                          f"  ✗ {c.get('name','?')} — {report.get('risk_level','?')} risk")
            else:
                passed.append(c)

        self._log(3, "compliance_filter",
                  f"{len(passed)} passed, {len(disqualified)} disqualified",
                  {"passed": len(passed), "disqualified": len(disqualified)})
        return passed, disqualified

    # ------------------------------------------------------------------
    # Step 4 — Score & rank
    # ------------------------------------------------------------------

    def _score(self, query_material: dict, candidates: list, top_k: int) -> tuple[list, list]:
        from scorer import rank_candidates

        try:
            ranked, score_disq = rank_candidates(query_material, candidates, top_k=top_k)

            # Attach compliance engine data to ranked results
            for r in ranked:
                comp = next((c["_compliance"] for c in candidates if c["id"] == r["id"]), {})
                if comp:
                    r.setdefault("scores", {})
                    r["scores"]["compliance_engine"] = comp.get("compliance_score", 0.5)
                    r["scores"]["traceability"]      = comp.get("traceability_score", 0.5)
                    r["scores"]["risk_level"]        = comp.get("risk_level", "MEDIUM")
                    r["compliance_report"]           = {
                        "risk_level":      comp.get("risk_level"),
                        "flags":           comp.get("flags", []),
                        "warnings":        comp.get("warnings", []),
                        "passed":          comp.get("passed", []),
                        "conflict_minerals": comp.get("conflict_minerals", []),
                    }

            top_name = ranked[0]["name"] if ranked else "none"
            self._log(4, "scoring_complete",
                      f"Ranked {len(ranked)} materials. Top: {top_name}",
                      {"ranked_count": len(ranked)})
            return ranked, score_disq
        except Exception as e:
            self._log(4, "scoring_error", str(e))
            return [], []

    # ------------------------------------------------------------------
    # Step 5 — LLM explanation for top result
    # ------------------------------------------------------------------

    def _explain(self, query_material: dict, top_substitute: dict) -> str:
        from knowledge_store import get_material_full

        sub_full = get_material_full(top_substitute["id"])
        if not sub_full:
            return "Explanation unavailable — substitute material not found."

        def summarize(mat: dict) -> str:
            specs = "\n".join(
                f"  {s['attribute']}: {s['value']} {s.get('unit','')}"
                for s in (mat.get("specs") or [])[:8]
                if s.get("value") is not None
            ) or "  (no specs)"
            comp = ", ".join(
                f"{c['standard']}={c['status']}" for c in (mat.get("compliance") or [])
            ) or "none declared"
            price = mat.get("price")
            price_str = (f"${price['price_usd']}/kg · MOQ {price['moq']} · {price['lead_time_days']}d"
                         if price else "N/A")
            return (f"Name: {mat['name']}  |  Category: {mat.get('category','N/A')}\n"
                    f"Specs:\n{specs}\nCompliance: {comp}\nPrice: {price_str}")

        scores_text = json.dumps(top_substitute.get("scores", {}), indent=2)
        comp_report = top_substitute.get("compliance_report", {})
        risk = comp_report.get("risk_level", "MEDIUM")

        prompt = (
            "You are Agnes, an expert supply-chain materials intelligence assistant.\n"
            "Write a concise 3-5 sentence technical justification for this substitution.\n"
            "Cite specific spec differences, compliance status (risk level: "
            f"{risk}), and cost/lead-time trade-offs. Flag any risks clearly.\n\n"
            f"ORIGINAL MATERIAL:\n{summarize(query_material)}\n\n"
            f"PROPOSED SUBSTITUTE:\n{summarize(sub_full)}\n\n"
            f"MULTI-CRITERIA SCORES:\n{scores_text}\n\n"
            "Write your explanation:"
        )
        try:
            resp = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )
            explanation = resp.choices[0].message.content.strip()
            self._log(5, "explanation_generated",
                      f"Explanation for {sub_full['name']} generated.")
            return explanation
        except Exception as e:
            self._log(5, "explanation_error", str(e))
            return f"LLM explanation failed: {e}"

    # ------------------------------------------------------------------
    # Step 6 — Audit log
    # ------------------------------------------------------------------

    def _audit(self, query_material: dict, query_text: str, ranked_count: int):
        try:
            from knowledge_store import log_audit
            log_audit(
                action="agent_query",
                query_material_id=query_material.get("id") if query_material else None,
                query_text=query_text,
                result_count=ranked_count,
                metadata={"model": HF_LLM_MODEL},
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, material_id: str = None, query_text: str = None,
            top_k: int = 5) -> dict:
        """
        Execute the full agentic pipeline.

        Returns:
          query, query_material (name), substitutes (ranked list),
          disqualified (list), top_explanation (str), agent_trace (list), stats (dict)
        """
        self.trace = []

        # 1. Resolve
        query_material, q_text = self._resolve(material_id, query_text)

        # 2. Search
        candidates = self._search(material_id, q_text)
        if not candidates:
            return {
                "query":           q_text,
                "query_material":  query_material.get("name", q_text),
                "substitutes":     [],
                "disqualified":    [],
                "top_explanation": None,
                "agent_trace":     self.trace,
                "stats":           {"candidates_found": 0},
                "message":         "No candidates found. Ingest more materials or rebuild the index.",
            }

        # 3. Compliance filter
        compliant, disqualified = self._compliance_filter(candidates)
        if not compliant:
            return {
                "query":           q_text,
                "query_material":  query_material.get("name", q_text),
                "substitutes":     [],
                "disqualified":    disqualified,
                "top_explanation": None,
                "agent_trace":     self.trace,
                "stats":           {"candidates_found": len(candidates),
                                    "compliance_failed": len(disqualified)},
                "message":         "All candidates disqualified by compliance engine.",
            }

        # 4. Score
        ranked, score_disq = self._score(query_material, compliant, top_k)

        # 5. Explain top result (only when querying by material ID for context)
        explanation = None
        if ranked and material_id:
            explanation = self._explain(query_material, ranked[0])

        # 6. Audit
        self._audit(query_material, q_text, len(ranked))

        return {
            "query":           q_text,
            "query_material":  query_material.get("name", q_text),
            "substitutes":     ranked,
            "disqualified":    disqualified + score_disq,
            "top_explanation": explanation,
            "agent_trace":     self.trace,
            "stats": {
                "candidates_found":  len(candidates),
                "compliance_passed": len(compliant),
                "compliance_failed": len(disqualified),
                "final_ranked":      len(ranked),
            },
        }
