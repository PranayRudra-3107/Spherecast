"""
ui.py — Agnes Material Intelligence UI
Run: streamlit run ui.py
Requires the FastAPI backend running at http://localhost:8000
"""

import requests
import streamlit as st
import pandas as pd

API = "http://localhost:8000"

st.set_page_config(
    page_title="Agnes — Material Intelligence",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 4px 0; }
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
}
.rank-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path: str):
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach Agnes API at `localhost:8000`. Is it running?"
    except Exception as e:
        return None, str(e)


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach Agnes API at `localhost:8000`. Is it running?"
    except Exception as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail


def score_color(score: float) -> str:
    if score >= 0.75:
        return "#16a34a"
    if score >= 0.5:
        return "#d97706"
    return "#dc2626"


def score_bar(score: float) -> str:
    pct = int(score * 100)
    color = score_color(score)
    return f"""
    <div style="display:flex;align-items:center;gap:8px">
      <div style="flex:1;background:#e2e8f0;border-radius:4px;height:8px">
        <div style="width:{pct}%;background:{color};border-radius:4px;height:8px"></div>
      </div>
      <span style="font-size:13px;font-weight:600;color:{color};width:40px">{score:.2f}</span>
    </div>"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔩 Agnes")
    st.markdown("*Raw Material Intelligence*")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🔍 Find Substitutes",
            "🤖 Agent Query",
            "🔬 Compliance",
            "📥 Ingest Material",
            "🌐 Ingest from URL",
            "📋 BOM Import",
            "📦 Browse Materials",
            "⭐ Feedback",
            "📜 Audit Log",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    # API status indicator
    health, err = api_get("/health")
    if health:
        hf = health.get("huggingface", {})
        dot = "🟢" if hf.get("connected") else "🟡"
        st.markdown(f"{dot} **API Online**")
        st.caption(f"LLM: `{hf.get('llm_model','').split('/')[-1]}`")
        st.caption(f"Embed: `{hf.get('embed_model','').split('/')[-1]}`")
    else:
        st.markdown("🔴 **API Offline**")
        st.caption("Start: `uvicorn main:app --reload`")


# ---------------------------------------------------------------------------
# Helper: risk badge
# ---------------------------------------------------------------------------
def risk_badge(level: str) -> str:
    colors = {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7c3aed"}
    color = colors.get(level, "#64748b")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700">{level}</span>'


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.title("Agnes — Raw Material Intelligence")
    st.markdown("Substitute component finder for supply-chain teams. Powered by **Gemma** + **FAISS** + **SQLite**.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    materials, err = api_get("/materials")
    count = len(materials.get("materials", [])) if materials else 0

    col1.metric("Materials in DB", count)
    col2.metric("LLM", "Gemma 4 26B")
    col3.metric("Vector Index", "FAISS · MiniLM-L6")

    st.divider()
    st.subheader("How it works")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**1. Ingest**\nPaste spec text or upload a PDF — Gemma extracts structured specs automatically.")
    c2.info("**2. Index**\nSpecs are embedded via sentence-transformers and stored in a FAISS vector index.")
    c3.info("**3. Search**\nQuery by material ID or free text — vector search returns the closest candidates.")
    c4.info("**4. Score**\nCandidates are ranked by spec similarity, compliance, price, quality, and business fit.")

    st.divider()
    st.subheader("New in v2 — Complex Architecture")
    r1, r2, r3, r4 = st.columns(4)
    r1.success("**🤖 Agent**\nMulti-step orchestration: search → compliance → score → explain")
    r2.success("**🔬 Compliance**\nRoHS · REACH SVHC · Conflict Minerals · FDA — full regulatory scan")
    r3.success("**🌐 URL Ingest**\nScrape supplier pages automatically")
    r4.success("**📋 BOM Import**\nCSV Bill-of-Materials bulk ingestion")

    if count == 0:
        st.warning("No materials found. Run `python setup_and_demo.py` to seed demo data, then rebuild the index.")
    else:
        st.success(f"Ready — {count} materials indexed. Go to **Agent Query** or **Find Substitutes**.")


# ---------------------------------------------------------------------------
# Page: Find Substitutes
# ---------------------------------------------------------------------------
elif page == "🔍 Find Substitutes":
    st.title("🔍 Find Substitutes")

    materials_resp, err = api_get("/materials")
    if err:
        st.error(err)
        st.stop()

    materials = materials_resp.get("materials", [])
    if not materials:
        st.warning("No materials in the database. Seed demo data first with `python setup_and_demo.py`.")
        st.stop()

    mat_options = {f"{m['name']} ({m.get('part_number') or 'no PN'})": m["id"] for m in materials}

    col_left, col_right = st.columns([2, 1])

    with col_left:
        mode = st.radio("Query by", ["Select a material", "Free text"], horizontal=True)

    with col_right:
        top_k = st.slider("Top results", 1, 10, 5)
        same_cat = st.checkbox("Same category only")

    if mode == "Select a material":
        selected_label = st.selectbox("Material", list(mat_options.keys()))
        query_payload = {
            "material_id": mat_options[selected_label],
            "top_k": top_k,
            "require_same_category": same_cat,
        }
    else:
        query_text = st.text_input("Describe the material", placeholder="lightweight high-strength aluminum for aerospace bracket")
        query_payload = {
            "query_text": query_text,
            "top_k": top_k,
            "require_same_category": same_cat,
        }

    if st.button("Find Substitutes", type="primary", use_container_width=True):
        if mode == "Free text" and not query_payload.get("query_text", "").strip():
            st.warning("Enter a description first.")
            st.stop()

        with st.spinner("Searching and scoring candidates..."):
            result, err = api_post("/substitutes", query_payload)

        if err:
            st.error(err)
            st.stop()

        subs = result.get("substitutes", [])
        disq = result.get("disqualified", [])

        st.divider()
        st.subheader(f"Results for: *{result.get('query_material_name')}*")

        if not subs:
            st.info("No substitutes found. Try rebuilding the index via `POST /build-index`.")
        else:
            # Score table
            table_data = []
            for s in subs:
                sc = s["scores"]
                comp = s.get("compliance_summary", {})
                comp_str = " · ".join(f"{k}={v}" for k, v in comp.items()) or "—"
                price = s.get("price")
                price_str = f"${price['price_usd']}/kg" if price else "—"
                table_data.append({
                    "Rank": s["rank"],
                    "Name": s["name"],
                    "Supplier": s.get("supplier") or "—",
                    "Score": sc["total"],
                    "Spec": sc["spec_similarity"],
                    "Compliance": sc["compliance"],
                    "Price/LT": sc["price_lead_time"],
                    "Quality": sc["quality"],
                    "Price": price_str,
                    "Compliance Notes": comp_str,
                })

            df = pd.DataFrame(table_data)

            # Color-coded score columns
            def style_scores(val):
                if isinstance(val, float):
                    color = score_color(val)
                    return f"color: {color}; font-weight: 600"
                return ""

            styled = df.style.applymap(
                style_scores, subset=["Score", "Spec", "Compliance", "Price/LT", "Quality"]
            ).format({
                "Score": "{:.3f}", "Spec": "{:.3f}", "Compliance": "{:.3f}",
                "Price/LT": "{:.3f}", "Quality": "{:.3f}",
            }).hide(axis="index")

            st.dataframe(styled, use_container_width=True)

            # Expandable detail cards
            st.subheader("Details")
            for s in subs:
                sc = s["scores"]
                with st.expander(f"#{s['rank']} — {s['name']}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Supplier:** {s.get('supplier') or '—'}")
                        st.markdown(f"**Part #:** {s.get('part_number') or '—'}")
                        price = s.get("price")
                        if price:
                            st.markdown(f"**Price:** ${price['price_usd']}/kg · MOQ {price['moq']} · {price['lead_time_days']} days")
                    with c2:
                        comp = s.get("compliance_summary", {})
                        for std, status in comp.items():
                            icon = "✅" if status == "pass" else "❌" if status == "fail" else "❓"
                            st.markdown(f"{icon} **{std}:** {status}")

                    st.markdown("**Scores**")
                    for label, key in [
                        ("Overall", "total"), ("Spec Similarity", "spec_similarity"),
                        ("Compliance", "compliance"), ("Price / Lead Time", "price_lead_time"),
                        ("Quality", "quality"), ("Business Priority", "business_priority"),
                    ]:
                        val = sc.get(key)
                        if val is not None:
                            st.markdown(f"*{label}*")
                            st.markdown(score_bar(val), unsafe_allow_html=True)

                    # Explain button
                    mat_id = query_payload.get("material_id")
                    if mat_id:
                        if st.button(f"Explain this substitution", key=f"explain_{s['material_id']}"):
                            with st.spinner("Asking Gemma for an explanation..."):
                                exp_result, exp_err = api_post("/explain", {
                                    "query_material_id": mat_id,
                                    "substitute_material_id": s["material_id"],
                                    "scores": sc,
                                })
                            if exp_err:
                                st.error(exp_err)
                            else:
                                st.info(exp_result.get("explanation", "No explanation returned."))

        if disq:
            with st.expander(f"Disqualified candidates ({len(disq)})"):
                for d in disq:
                    st.markdown(f"- **{d['name']}** — {d.get('reason', 'unknown reason')}")


# ---------------------------------------------------------------------------
# Page: Ingest Material
# ---------------------------------------------------------------------------
elif page == "📥 Ingest Material":
    st.title("📥 Ingest Material")
    st.markdown("Paste raw spec text (any format). Gemma will extract structured specs automatically.")

    with st.form("ingest_form"):
        text = st.text_area(
            "Specification text",
            height=280,
            placeholder=(
                "Material: Aluminum 6061-T6\n"
                "Tensile Strength: 310 MPa\n"
                "Yield Strength: 276 MPa\n"
                "Density: 2.70 g/cm3\n"
                "RoHS Compliant: Yes\n"
                "Price: $4.20/kg, MOQ 50kg, Lead time 14 days"
            ),
        )
        c1, c2, c3, c4 = st.columns(4)
        source_name = c1.text_input("Source name", value="manual_input")
        price_usd = c2.number_input("Price ($/kg)", min_value=0.0, value=0.0, step=0.01)
        moq = c3.number_input("MOQ (kg)", min_value=0, value=0, step=1)
        lead_time = c4.number_input("Lead time (days)", min_value=0, value=30, step=1)
        submitted = st.form_submit_button("Ingest Material", type="primary", use_container_width=True)

    if submitted:
        if not text.strip():
            st.warning("Paste some spec text first.")
        else:
            payload = {
                "text": text,
                "source_name": source_name,
                "price_usd": price_usd if price_usd > 0 else None,
                "moq": moq if moq > 0 else None,
                "lead_time_days": lead_time if lead_time > 0 else None,
            }
            with st.spinner("Extracting specs with Gemma — this may take 20–60s..."):
                result, err = api_post("/ingest/text", payload)

            if err:
                st.error(err)
            else:
                st.success(f"Ingested **{result['name']}** — {result['specs_extracted']} specs, {result['compliance_records']} compliance records extracted.")
                st.code(f"material_id: {result['material_id']}", language="text")

    st.divider()
    st.subheader("Upload PDF")
    uploaded = st.file_uploader("PDF spec sheet", type=["pdf"])
    if uploaded:
        if st.button("Ingest PDF", type="primary"):
            with st.spinner("Extracting text from PDF and running Gemma..."):
                try:
                    r = requests.post(
                        f"{API}/ingest/pdf",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=180,
                    )
                    r.raise_for_status()
                    result = r.json()
                    st.success(f"Ingested **{result['name']}** — {result['specs_extracted']} specs extracted.")
                    st.code(f"material_id: {result['material_id']}", language="text")
                except Exception as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------
# Page: Browse Materials
# ---------------------------------------------------------------------------
elif page == "📦 Browse Materials":
    st.title("📦 Browse Materials")

    materials_resp, err = api_get("/materials")
    if err:
        st.error(err)
        st.stop()

    materials = materials_resp.get("materials", [])
    if not materials:
        st.info("No materials yet. Seed demo data with `python setup_and_demo.py`.")
        st.stop()

    st.markdown(f"**{len(materials)} materials** in the database.")

    # Search filter
    search_q = st.text_input("Filter by name or supplier", placeholder="aluminum, MetalsCo, PEEK...")
    if search_q:
        q = search_q.lower()
        materials = [m for m in materials if q in m.get("name", "").lower() or q in (m.get("supplier") or "").lower()]

    col_a, col_b = st.columns([1, 2])

    with col_a:
        mat_names = [f"{m['name']}" for m in materials]
        selected_idx = st.radio("Select material", range(len(mat_names)), format_func=lambda i: mat_names[i])

    with col_b:
        selected_id = materials[selected_idx]["id"]
        detail, err = api_get(f"/materials/{selected_id}")
        if err:
            st.error(err)
        elif detail:
            st.subheader(detail["name"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Category", detail.get("category") or "—")
            c2.metric("Supplier", detail.get("supplier") or "—")
            c3.metric("Part #", detail.get("part_number") or "—")

            price = detail.get("price")
            if price:
                p1, p2, p3 = st.columns(3)
                p1.metric("Price", f"${price['price_usd']}/kg")
                p2.metric("MOQ", f"{price['moq']} kg")
                p3.metric("Lead Time", f"{price['lead_time_days']} days")

            specs = detail.get("specs", [])
            if specs:
                st.markdown("**Specifications**")
                spec_df = pd.DataFrame([
                    {"Attribute": s["attribute"], "Value": s["value"], "Unit": s.get("unit") or "", "Confidence": s.get("confidence", 1.0)}
                    for s in specs if s.get("value") is not None
                ])
                st.dataframe(spec_df.style.format({"Confidence": "{:.0%}"}), use_container_width=True, hide_index=True)

            compliance = detail.get("compliance", [])
            if compliance:
                st.markdown("**Compliance**")
                cols = st.columns(len(compliance))
                for col, c in zip(cols, compliance):
                    icon = "✅" if c["status"] == "pass" else "❌" if c["status"] == "fail" else "❓"
                    col.metric(c["standard"], f"{icon} {c['status']}")

            st.markdown(f"**ID:** `{selected_id}`")


# ---------------------------------------------------------------------------
# Page: Agent Query
# ---------------------------------------------------------------------------
elif page == "🤖 Agent Query":
    st.title("🤖 Agent Query")
    st.markdown(
        "Multi-step agentic substitute finder: **search → compliance filter → score → explain**. "
        "Each step is shown in the trace below."
    )

    materials_resp, err = api_get("/materials")
    materials = (materials_resp or {}).get("materials", [])
    mat_options = {f"{m['name']} ({m.get('part_number') or 'no PN'})": m["id"] for m in materials}

    mode = st.radio("Query by", ["Select a material", "Free text"], horizontal=True)
    top_k = st.slider("Top results", 1, 10, 5)

    if mode == "Select a material" and mat_options:
        sel = st.selectbox("Material", list(mat_options.keys()))
        payload = {"material_id": mat_options[sel], "top_k": top_k}
    else:
        qt = st.text_input("Describe the material", placeholder="high-strength aluminum for aerospace")
        payload = {"query_text": qt, "top_k": top_k}

    if st.button("Run Agent", type="primary", use_container_width=True):
        if mode == "Free text" and not payload.get("query_text", "").strip():
            st.warning("Enter a description first.")
            st.stop()
        with st.spinner("Agent running — this may take 30–90s (LLM explanation step)..."):
            result, err = api_post("/agent/find", payload)
        if err:
            st.error(err)
            st.stop()

        st.divider()

        # Stats
        stats = result.get("stats", {})
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Candidates Found", stats.get("candidates_found", "—"))
        s2.metric("Passed Compliance", stats.get("compliance_passed", "—"))
        s3.metric("Compliance Failed", stats.get("compliance_failed", "—"))
        s4.metric("Final Ranked", stats.get("final_ranked", "—"))

        # Agent trace
        st.subheader("Agent Trace")
        for t in result.get("agent_trace", []):
            icon = {"resolve_material": "🔎", "vector_search": "🔍", "graph_expansion": "🕸",
                    "compliance_filter": "🔬", "disqualified": "✗", "scoring_complete": "📊",
                    "explanation_generated": "💬"}.get(t["action"], "▸")
            st.markdown(f"**Step {t['step']}** `{t['action']}` {icon} — {t['detail']}")

        # Top explanation
        exp = result.get("top_explanation")
        if exp:
            st.divider()
            st.subheader("Top Substitute — Explanation")
            subs = result.get("substitutes", [])
            if subs:
                st.markdown(f"**{subs[0].get('name')}**")
            st.info(exp)

        # Results table
        subs = result.get("substitutes", [])
        if subs:
            st.divider()
            st.subheader(f"Ranked Substitutes — {result.get('query_material')}")
            rows = []
            for s in subs:
                sc = s.get("scores", {})
                comp_r = s.get("compliance_report", {})
                rows.append({
                    "Rank": s.get("rank", "—"),
                    "Name": s["name"],
                    "Score": sc.get("total", "—"),
                    "Spec": sc.get("spec_similarity", "—"),
                    "Compliance": sc.get("compliance", "—"),
                    "Traceability": sc.get("traceability", "—"),
                    "Risk": comp_r.get("risk_level", "—"),
                    "Feedback Adj": sc.get("feedback_adj", 0.0),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Disqualified
        disq = result.get("disqualified", [])
        if disq:
            with st.expander(f"Disqualified ({len(disq)})"):
                for d in disq:
                    st.markdown(f"- **{d.get('name','?')}** — {d.get('disqualification_reason','?')}")


# ---------------------------------------------------------------------------
# Page: Compliance
# ---------------------------------------------------------------------------
elif page == "🔬 Compliance":
    st.title("🔬 Compliance Report")
    st.markdown(
        "Full regulatory scan per material: **RoHS · REACH SVHC · Conflict Minerals (3TG) · FDA · CA Prop 65**"
    )

    materials_resp, err = api_get("/materials")
    materials = (materials_resp or {}).get("materials", [])
    if not materials:
        st.info("No materials. Seed demo data first.")
        st.stop()

    mat_options = {m["name"]: m["id"] for m in materials}

    col_l, col_r = st.columns([1, 2])
    with col_l:
        sel_name = st.radio("Select material", list(mat_options.keys()))
    with col_r:
        mat_id = mat_options[sel_name]
        report, err = api_get(f"/compliance/{mat_id}")
        if err:
            st.error(err)
            st.stop()

        risk = report.get("risk_level", "UNKNOWN")
        disq = report.get("disqualified", False)

        # Header
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Risk Level**<br>{risk_badge(risk)}", unsafe_allow_html=True)
        c2.metric("Compliance Score", f"{report.get('compliance_score', 0):.0%}")
        c3.metric("Traceability",     f"{report.get('traceability_score', 0):.0%}")

        if disq:
            st.error("DISQUALIFIED — this material has critical compliance failures.")

        # Passed
        passed = report.get("passed", [])
        if passed:
            st.markdown("**Passed Standards**")
            for p in passed:
                st.markdown(f"✅ **{p['standard']}** — {p['message']}")

        # Flags
        flags = report.get("flags", [])
        if flags:
            st.markdown("**Flags (Critical / High)**")
            for f in flags:
                icon = "🚨" if f["severity"] == "CRITICAL" else "⚠️"
                st.markdown(f"{icon} **{f['standard']}** `{f['severity']}` — {f['message']}")

        # Warnings
        warns = report.get("warnings", [])
        if warns:
            with st.expander(f"Warnings & Info ({len(warns)})"):
                for w in warns:
                    icon = "⚠️" if w["severity"] == "WARNING" else "ℹ️"
                    st.markdown(f"{icon} **{w['standard']}** — {w['message']}")

        # Conflict minerals
        cm = report.get("conflict_minerals", [])
        if cm:
            st.warning(f"Conflict Minerals detected: **{', '.join(cm)}** — Dodd-Frank declaration required.")

        # Existing declarations
        decl = report.get("existing_declarations", {})
        if decl:
            st.markdown("**Declared Standards**")
            cols = st.columns(min(len(decl), 4))
            for col, (std, status) in zip(cols, decl.items()):
                icon = "✅" if status == "pass" else "❌" if status == "fail" else "❓"
                col.metric(std, f"{icon} {status}")


# ---------------------------------------------------------------------------
# Page: Ingest from URL
# ---------------------------------------------------------------------------
elif page == "🌐 Ingest from URL":
    st.title("🌐 Ingest from URL")
    st.markdown("Fetch a supplier product page or spec sheet URL — Gemma extracts specs automatically.")

    with st.form("url_form"):
        url = st.text_input("URL", placeholder="https://supplier.com/product/datasheet")
        source_name = st.text_input("Source name", value="web_scrape")
        c1, c2, c3 = st.columns(3)
        price_usd     = c1.number_input("Price ($/kg)", min_value=0.0, value=0.0, step=0.01)
        moq           = c2.number_input("MOQ", min_value=0, value=0)
        lead_time     = c3.number_input("Lead time (days)", min_value=0, value=30)
        submitted = st.form_submit_button("Fetch & Ingest", type="primary", use_container_width=True)

    if submitted:
        if not url.strip():
            st.warning("Enter a URL.")
        else:
            payload = {
                "url": url, "source_name": source_name,
                "price_usd": price_usd or None,
                "moq": moq or None,
                "lead_time_days": lead_time or None,
            }
            with st.spinner("Fetching page and extracting specs with Gemma..."):
                result, err = api_post("/ingest/url", payload)
            if err:
                st.error(err)
            else:
                st.success(
                    f"Ingested **{result['name']}** — "
                    f"{result['specs_extracted']} specs, {result['compliance_records']} compliance records."
                )
                st.code(f"material_id: {result['material_id']}", language="text")


# ---------------------------------------------------------------------------
# Page: BOM Import
# ---------------------------------------------------------------------------
elif page == "📋 BOM Import":
    st.title("📋 BOM Import")
    st.markdown("Upload a Bill-of-Materials CSV to bulk-ingest materials.")

    st.subheader("Expected CSV columns")
    st.code(
        "name, part_number, supplier, category, tensile_strength, density, "
        "elastic_modulus, max_operating_temp, price_usd, moq, lead_time_days, "
        "rohs, reach, fda, notes",
        language="text",
    )
    st.caption("Only `name` is required. All other columns are optional.")

    bom_file = st.file_uploader("Upload CSV", type=["csv"])
    manual_csv = st.text_area("…or paste CSV directly", height=180,
                               placeholder="name,supplier,price_usd\nAluminum 6061,MetalsCo,4.20")
    source_name = st.text_input("Source name", value="bom_import")

    csv_text = None
    if bom_file:
        csv_text = bom_file.read().decode("utf-8")
    elif manual_csv.strip():
        csv_text = manual_csv

    if csv_text and st.button("Import BOM", type="primary", use_container_width=True):
        with st.spinner("Ingesting BOM via Gemma — this may take several minutes..."):
            result, err = api_post("/ingest/bom", {"csv_text": csv_text, "source_name": source_name})
        if err:
            st.error(err)
        else:
            st.success(f"Ingested {result['ingested']} materials. Failed: {result['failed']}.")
            rows = []
            for r in result.get("results", []):
                rows.append({
                    "Name": r.get("name", "?"),
                    "Status": "✅ OK" if "error" not in r else f"❌ {r['error'][:60]}",
                    "Specs": r.get("specs_extracted", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Feedback
# ---------------------------------------------------------------------------
elif page == "⭐ Feedback":
    st.title("⭐ Substitution Feedback")
    st.markdown(
        "Rate substitution suggestions. Feedback adjusts future scores "
        "(approved → +0.2, rejected → −0.2 on the blended total)."
    )

    tab_submit, tab_history = st.tabs(["Submit Feedback", "Feedback History"])

    with tab_submit:
        materials_resp, _ = api_get("/materials")
        materials = (materials_resp or {}).get("materials", [])
        mat_options = {m["name"]: m["id"] for m in materials}
        if not mat_options:
            st.info("No materials yet.")
        else:
            with st.form("feedback_form"):
                query_name = st.selectbox("Original material (query)", list(mat_options.keys()))
                sub_name   = st.selectbox("Proposed substitute", list(mat_options.keys()))
                approved   = st.radio("Decision", ["Approve ✅", "Reject ❌"], horizontal=True)
                comment    = st.text_input("Comment (optional)")
                submitted  = st.form_submit_button("Submit", type="primary")

            if submitted:
                if query_name == sub_name:
                    st.warning("Query and substitute cannot be the same material.")
                else:
                    payload = {
                        "query_material_id":     mat_options[query_name],
                        "substitute_material_id": mat_options[sub_name],
                        "approved":              approved.startswith("Approve"),
                        "comment":               comment or None,
                    }
                    result, err = api_post("/feedback", payload)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"Feedback recorded (ID: `{result['feedback_id']}`)")

    with tab_history:
        fb_resp, err = api_get("/feedback?limit=200")
        if err:
            st.error(err)
        else:
            fb = fb_resp.get("feedback", [])
            if not fb:
                st.info("No feedback submitted yet.")
            else:
                rows = []
                for f in fb:
                    rows.append({
                        "Query":      f.get("query_name") or f.get("query_material_id","?")[:8],
                        "Substitute": f.get("sub_name") or f.get("substitute_material_id","?")[:8],
                        "Decision":   "✅ Approved" if f.get("approved") else "❌ Rejected",
                        "Comment":    f.get("comment") or "—",
                        "Date":       (f.get("created_at") or "")[:10],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Audit Log
# ---------------------------------------------------------------------------
elif page == "📜 Audit Log":
    st.title("📜 Audit Log")
    st.markdown("Traceability trail of all agent queries and ingestion events.")

    audit_resp, err = api_get("/audit-log?limit=100")
    if err:
        st.error(err)
        st.stop()

    logs = audit_resp.get("audit_log", [])
    if not logs:
        st.info("No audit entries yet. Run an Agent Query or ingest a material.")
    else:
        rows = []
        for entry in logs:
            rows.append({
                "Action":    entry.get("action", "—"),
                "Material":  entry.get("material_name") or entry.get("query_material_id", "—"),
                "Query":     (entry.get("query_text") or "—")[:60],
                "Results":   entry.get("result_count", "—"),
                "Timestamp": (entry.get("created_at") or "")[:19].replace("T", " "),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(logs)} entries shown.")
