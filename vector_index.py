"""
vector_index.py
---------------
FAISS-backed vector index using HuggingFace Inference API embeddings
(sentence-transformers/all-MiniLM-L6-v2).
"""

import os
import pickle
import numpy as np
from pathlib import Path
from huggingface_hub import InferenceClient

INDEX_PATH = Path("agnes_faiss.index")
META_PATH  = Path("agnes_faiss_meta.pkl")

HF_TOKEN    = os.environ.get("HF_TOKEN")
EMBED_MODEL = os.environ.get("AGNES_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_text(text: str) -> np.ndarray:
    """Get a single embedding from HuggingFace Inference API."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable not set.\n"
            "Fix: export HF_TOKEN=hf_your_token_here"
        )
    try:
        client = InferenceClient(token=HF_TOKEN)
        vec = client.feature_extraction(text, model=EMBED_MODEL)
        arr = np.array(vec, dtype=np.float32)
        # Handle (seq_len, dim) output — mean-pool to (dim,)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr
    except Exception as e:
        raise RuntimeError(f"HuggingFace embedding error: {e}")


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, dim) array."""
    vecs = [embed_text(t) for t in texts]
    return np.vstack(vecs)


def build_material_text(material_id: str) -> str:
    """Build a rich text representation of a material for embedding."""
    from knowledge_store import get_conn
    with get_conn() as conn:
        mat   = conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        specs = conn.execute(
            "SELECT attribute, value, unit FROM specs WHERE material_id = ? AND confidence > 0.5",
            (material_id,),
        ).fetchall()
        comp  = conn.execute(
            "SELECT standard, status FROM compliance WHERE material_id = ?",
            (material_id,),
        ).fetchall()

    if not mat:
        return ""

    parts = [
        f"Material: {mat['name']}",
        f"Category: {mat['category'] or 'unknown'}",
        f"Supplier: {mat['supplier'] or 'unknown'}",
    ]
    if mat["part_number"]:
        parts.append(f"Part number: {mat['part_number']}")

    if specs:
        parts.append("Specifications:")
        for s in specs:
            if s["value"] is not None:
                parts.append(f"  {s['attribute']}: {s['value']} {s['unit'] or ''}")

    if comp:
        parts.append("Compliance: " + ", ".join(f"{c['standard']}={c['status']}" for c in comp))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def build_index():
    """Build FAISS index over all materials in the DB. Saves to disk."""
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu")

    from knowledge_store import list_all_materials
    materials = list_all_materials()
    if not materials:
        print("[VectorIndex] No materials found — run ingestor first.")
        return

    print(f"[VectorIndex] Embedding {len(materials)} materials with {EMBED_MODEL}...")

    embeddings, meta = [], []
    for mat in materials:
        text = build_material_text(mat["id"])
        if not text:
            continue
        vec = embed_text(text)
        embeddings.append(vec)
        meta.append({"id": mat["id"], "name": mat["name"], "text": text})
        print(f"  ✓ {mat['name']}")

    matrix = np.vstack(embeddings).astype(np.float32)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])  # cosine via normalized inner product
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"[VectorIndex] Built: {len(meta)} vectors, dim={matrix.shape[1]}")


def _load_index():
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu")

    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found — run build_index() first.")

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query_text: str, top_k: int = 10, exclude_ids: list[str] = None) -> list[dict]:
    """Search by text query. Returns [{id, name, vector_score, text}]."""
    import faiss
    if not INDEX_PATH.exists():
        print("[VectorIndex] Index missing — building now...")
        build_index()

    index, meta = _load_index()
    exclude_ids = set(exclude_ids or [])

    q_vec = embed_text(query_text).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec) + 1e-9
    q_vec = q_vec.reshape(1, -1)

    k = min(top_k + len(exclude_ids) + 5, len(meta))
    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        m = meta[idx]
        if m["id"] in exclude_ids:
            continue
        results.append({
            "id": m["id"],
            "name": m["name"],
            "vector_score": float(score),
            "text": m["text"],
        })
        if len(results) >= top_k:
            break

    return results


def search_by_material_id(material_id: str, top_k: int = 10) -> list[dict]:
    """Find similar materials to an existing material."""
    text = build_material_text(material_id)
    if not text:
        return []
    return search(text, top_k=top_k + 1, exclude_ids=[material_id])


if __name__ == "__main__":
    build_index()
    print("\nTest search: 'high strength aluminum structural'")
    for r in search("high strength aluminum structural", top_k=3):
        print(f"  {r['name']} → {r['vector_score']:.3f}")
