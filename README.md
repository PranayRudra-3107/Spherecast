# Agnes — Raw Material Intelligence API

Substitute component finder for supply-chain intelligence. Built on Gemma (HuggingFace) + FAISS + SQLite.

## Architecture

```
Data Sources → Ingestor (Gemma extraction) → SQLite KG + FAISS Vector Index
                                                      ↓
                                      Substitute Finder (vector + KG merge)
                                                      ↓
                                       Scorer (spec / compliance / price)
                                                      ↓
                              FastAPI  →  /substitutes  →  Agnes
                                                      ↓
                                       Streamlit UI  (ui.py)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your HuggingFace API key

Copy the example env file and add your token:
```bash
cp .env.example .env
# Then edit .env and set HF_TOKEN=hf_your_token_here
```

Get a free token at https://huggingface.co/settings/tokens

> **Never commit `.env` to git.** It is already in `.gitignore`.

### 3. Seed demo data and verify
```bash
python setup_and_demo.py
```
This seeds 8 demo materials (aluminum alloys, stainless steel, titanium, nylon, PEEK, PTFE), builds the FAISS index, and runs a sample query.

### 4. Start the API
```bash
uvicorn main:app --reload --port 8000
```

Interactive docs at: http://localhost:8000/docs

### 5. Start the UI (new tab)
```bash
streamlit run ui.py
```

UI opens at: http://localhost:8501

---

## UI Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | Material count, system status, HF API health indicator |
| 🔍 Find Substitutes | Query by material or free text → ranked results table with score bars + Gemma explanations |
| 📥 Ingest Material | Paste spec text or upload a PDF — Gemma extracts specs automatically |
| 📦 Browse Materials | Filterable list with full specs, compliance, and price per material |

---

## Models Used

| Role | Model |
|------|-------|
| LLM (extraction + explanation) | `google/gemma-4-26B-A4B-it` via HF Inference API |
| Embeddings (vector search) | `sentence-transformers/all-MiniLM-L6-v2` via HF Inference API |

Override either model via environment variables:
```bash
export AGNES_LLM_MODEL=google/gemma-4-26B-A4B-it
export AGNES_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## API Endpoints

### `GET /health`
Check API status and HuggingFace connectivity.

### `GET /materials`
List all ingested materials.

### `GET /materials/{id}`
Full material record: specs, compliance, provenance, price.

### `POST /ingest/text`
Ingest raw spec text. Gemma extracts structured specs.
```json
{
  "text": "Material: 316L Stainless Steel\nTensile Strength: 485 MPa...",
  "source_name": "supplier_datasheet",
  "price_usd": 8.90,
  "moq": 200,
  "lead_time_days": 35
}
```

### `POST /ingest/pdf`
Upload a PDF spec sheet (multipart/form-data).

### `POST /substitutes`
Find ranked substitutes for a material.
```json
{
  "material_id": "uuid-of-material",
  "top_k": 5,
  "require_same_category": false
}
```
Or by free text:
```json
{
  "query_text": "lightweight high-strength aluminum for aerospace bracket",
  "top_k": 5
}
```

**Response includes:**
- Ranked substitutes with composite scores
- Per-dimension scores (spec, compliance, price, quality, business)
- Disqualified candidates with reasons
- Compliance summary per substitute

### `POST /explain`
Get a natural-language explanation for a substitution.
```json
{
  "query_material_id": "uuid",
  "substitute_material_id": "uuid",
  "scores": { "total": 0.82, "spec_similarity": 0.91 }
}
```

### `POST /build-index`
Rebuild the FAISS vector index (run after bulk ingestion).

---

## Scoring Weights

| Dimension         | Weight | Notes |
|------------------|--------|-------|
| Spec similarity   | 40%    | Numeric distance across 15+ attributes |
| Compliance        | 20%    | RoHS / REACH / FDA — hard gate: any fail = disqualified |
| Price & lead time | 15%    | Cost ratio, MOQ, delivery days |
| Quality           | 15%    | Supplier scorecards (heuristic in demo) |
| Business priority | 10%    | Category match, preferred suppliers |

Vector similarity blended in at 15% weight on final score.

---

## Adding Your Own Materials

```python
from ingestor import ingest_text

result = ingest_text("""
  Material: Your Material Name
  Tensile Strength: 300 MPa
  Density: 7.8 g/cm³
  ...any spec text...
  RoHS: Compliant
""", source_name="my_supplier_sheet", price_usd=12.50)

print(result)  # {'material_id': '...', 'name': '...', 'specs_extracted': N}
```

Then rebuild the index:
```bash
curl -X POST http://localhost:8000/build-index
```

---

## Production Upgrade Path

| Component | Demo | Production |
|-----------|------|------------|
| LLM | HF Inference API (Gemma) | Dedicated HF Endpoint / fine-tuned model |
| KG store | SQLite | Neo4j / Amazon Neptune |
| Vector DB | FAISS (local) | Milvus / Weaviate / Pinecone |
| PDF ingestion | pdfplumber | AWS Textract / Azure Form Recognizer |
| Supplier scraping | manual input | Playwright + structured extractors |
| Quality scores | heuristic | ERP defect rate integration |
| Auth | none | API key / OAuth2 |
| Audit log | SQLite provenance | append-only event log |
