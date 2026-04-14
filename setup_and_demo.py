#!/usr/bin/env python3
"""
setup_and_demo.py
-----------------
1. Check HuggingFace API token and connectivity
2. Initialize the database
3. Seed 8 demo materials (LLM extraction per material)
4. Build the FAISS vector index (sentence-transformers/all-MiniLM-L6-v2)
5. Run a sample substitution query and print results

Usage:
    export HF_TOKEN=hf_your_token_here
    python setup_and_demo.py
"""

import sys
import os
from huggingface_hub import InferenceClient

HF_TOKEN     = os.environ.get("HF_TOKEN")
LLM_MODEL    = os.environ.get("AGNES_LLM_MODEL", "google/gemma-4-26B-A4B-it")
EMBED_MODEL  = os.environ.get("AGNES_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def check_hf():
    print("[1/5] Checking HuggingFace API...")
    if not HF_TOKEN:
        print("\n  ERROR: HF_TOKEN environment variable not set.")
        print("  Fix: export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    try:
        client = InferenceClient(token=HF_TOKEN)
        client.feature_extraction("connectivity test", model=EMBED_MODEL)
        print(f"  ✓ HF API reachable")
        print(f"  ✓ LLM model: {LLM_MODEL}")
        print(f"  ✓ Embed model: {EMBED_MODEL}")
    except Exception as e:
        print(f"\n  ERROR: Cannot reach HuggingFace API: {e}")
        print("  Check your HF_TOKEN and internet connection.")
        sys.exit(1)


def init():
    print("\n[2/5] Initializing database...")
    from knowledge_store import init_db
    init_db()
    print("  ✓ agnes.db ready")


def seed():
    print("\n[3/5] Seeding demo materials (LLM extraction — ~2-4 min)...")
    from ingestor import seed_demo_data
    seed_demo_data()


def build():
    print("\n[4/5] Building FAISS vector index...")
    from vector_index import build_index
    build_index()
    print("  ✓ Index built")


def demo_query():
    print("\n[5/5] Running demo substitution query for Aluminum 6061-T6...")
    from knowledge_store import list_all_materials, get_material_full
    from vector_index import search_by_material_id
    from scorer import rank_candidates

    materials = list_all_materials()
    al6061 = next((m for m in materials if "6061" in m["name"]), None)
    if not al6061:
        print("  ERROR: 6061 not found — seed may have failed.")
        return

    query_mat      = get_material_full(al6061["id"])
    candidates_raw = search_by_material_id(al6061["id"], top_k=15)
    ranked, disq   = rank_candidates(query_mat, candidates_raw, top_k=5)

    print(f"\n  Query: {query_mat['name']}")
    print(f"  {'Rank':<5} {'Name':<35} {'Total':>7} {'Spec':>7} {'Comp':>7} {'Price':>7}")
    print(f"  {'-'*65}")
    for i, r in enumerate(ranked, 1):
        s = r["scores"]
        print(f"  {i:<5} {r['name']:<35} {s['total']:>7.3f} {s['spec_similarity']:>7.3f} {s['compliance']:>7.3f} {s['price_lead_time']:>7.3f}")

    if disq:
        print(f"\n  Disqualified: {', '.join(d['name'] for d in disq)}")

    print(f"\n{'='*60}")
    print("Setup complete! Start the API with:")
    print("  uvicorn main:app --reload --port 8000")
    print("\nAPI docs:  http://localhost:8000/docs")
    print("Health:    http://localhost:8000/health")
    print(f"\nExample query:")
    print(f"  curl -X POST http://localhost:8000/substitutes \\")
    print(f"    -H 'Content-Type: application/json' \\")
    mat_id = al6061["id"]
    print(f"    -d '{{\"material_id\": \"{mat_id}\", \"top_k\": 3}}'")
    print("="*60)


if __name__ == "__main__":
    print("=" * 60)
    print("Agnes — Setup (HuggingFace Inference API)")
    print("=" * 60)
    check_hf()
    init()
    seed()
    build()
    demo_query()
