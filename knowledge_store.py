"""
knowledge_store.py
------------------
SQLite-backed knowledge store for materials, specs, compliance, and provenance.
Serves as the KG stand-in for the demo. Swap for Neo4j in production.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("agnes.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS materials (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            part_number TEXT,
            category    TEXT,
            supplier    TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS specs (
            id           TEXT PRIMARY KEY,
            material_id  TEXT NOT NULL,
            attribute    TEXT NOT NULL,   -- e.g. "tensile_strength"
            value        REAL,            -- numeric value (normalized)
            unit         TEXT,            -- normalized unit e.g. "MPa"
            raw_value    TEXT,            -- original string from source
            confidence   REAL DEFAULT 1.0,
            FOREIGN KEY (material_id) REFERENCES materials(id)
        );

        CREATE TABLE IF NOT EXISTS compliance (
            id           TEXT PRIMARY KEY,
            material_id  TEXT NOT NULL,
            standard     TEXT NOT NULL,   -- "RoHS", "REACH", "FDA", etc.
            status       TEXT NOT NULL,   -- "pass", "fail", "unknown"
            notes        TEXT,
            FOREIGN KEY (material_id) REFERENCES materials(id)
        );

        CREATE TABLE IF NOT EXISTS provenance (
            id           TEXT PRIMARY KEY,
            material_id  TEXT NOT NULL,
            source_type  TEXT NOT NULL,   -- "pdf", "website", "erp", "manual"
            source_name  TEXT NOT NULL,
            ingested_at  TEXT NOT NULL,
            raw_text     TEXT,
            FOREIGN KEY (material_id) REFERENCES materials(id)
        );

        CREATE TABLE IF NOT EXISTS price_records (
            id           TEXT PRIMARY KEY,
            material_id  TEXT NOT NULL,
            price_usd    REAL,
            currency     TEXT DEFAULT 'USD',
            moq          INTEGER,
            lead_time_days INTEGER,
            region       TEXT,
            recorded_at  TEXT NOT NULL,
            FOREIGN KEY (material_id) REFERENCES materials(id)
        );

        CREATE INDEX IF NOT EXISTS idx_specs_material ON specs(material_id);
        CREATE INDEX IF NOT EXISTS idx_compliance_material ON compliance(material_id);
        CREATE INDEX IF NOT EXISTS idx_specs_attribute ON specs(attribute);

        -- Graph layer: typed edges between any two nodes (materials, suppliers, certs)
        CREATE TABLE IF NOT EXISTS graph_edges (
            id          TEXT PRIMARY KEY,
            source_id   TEXT NOT NULL,
            target_id   TEXT NOT NULL,
            edge_type   TEXT NOT NULL,   -- suppliedBy | certifiedBy | substitutedBy | contains | relatedTo
            weight      REAL DEFAULT 1.0,
            properties  TEXT,            -- JSON blob for extra metadata
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_type   ON graph_edges(edge_type);

        -- User feedback on substitution suggestions
        CREATE TABLE IF NOT EXISTS feedback (
            id                    TEXT PRIMARY KEY,
            query_material_id     TEXT NOT NULL,
            substitute_material_id TEXT NOT NULL,
            approved              INTEGER NOT NULL,   -- 1=approved, 0=rejected
            score_override        REAL,               -- optional manual score
            comment               TEXT,
            created_at            TEXT NOT NULL
        );

        -- Audit trail for all agent queries and ingestion events
        CREATE TABLE IF NOT EXISTS audit_log (
            id                TEXT PRIMARY KEY,
            action            TEXT NOT NULL,          -- agent_query | ingest | compliance_check
            query_material_id TEXT,
            query_text        TEXT,
            result_count      INTEGER,
            metadata          TEXT,                   -- JSON
            created_at        TEXT NOT NULL
        );
        """)
    print(f"[KS] Database initialized at {DB_PATH}")


def upsert_material(
    name: str,
    part_number: Optional[str] = None,
    category: Optional[str] = None,
    supplier: Optional[str] = None,
    material_id: Optional[str] = None,
) -> str:
    """Insert or return existing material. Returns material_id."""
    mid = material_id or str(uuid.uuid4())
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM materials WHERE name = ? AND supplier = ?",
            (name, supplier),
        ).fetchone()
        if existing:
            return existing["id"]
        conn.execute(
            "INSERT INTO materials (id, name, part_number, category, supplier, created_at) VALUES (?,?,?,?,?,?)",
            (mid, name, part_number, category, supplier, datetime.utcnow().isoformat()),
        )
    return mid


def insert_specs(material_id: str, specs: list[dict]):
    """
    specs: list of {attribute, value, unit, raw_value, confidence}
    """
    with get_conn() as conn:
        conn.execute("DELETE FROM specs WHERE material_id = ?", (material_id,))
        for s in specs:
            conn.execute(
                "INSERT INTO specs (id, material_id, attribute, value, unit, raw_value, confidence) VALUES (?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    material_id,
                    s["attribute"],
                    s.get("value"),
                    s.get("unit"),
                    s.get("raw_value"),
                    s.get("confidence", 1.0),
                ),
            )


def insert_compliance(material_id: str, records: list[dict]):
    """records: list of {standard, status, notes}"""
    with get_conn() as conn:
        conn.execute("DELETE FROM compliance WHERE material_id = ?", (material_id,))
        for r in records:
            conn.execute(
                "INSERT INTO compliance (id, material_id, standard, status, notes) VALUES (?,?,?,?,?)",
                (str(uuid.uuid4()), material_id, r["standard"], r["status"], r.get("notes")),
            )


def insert_provenance(material_id: str, source_type: str, source_name: str, raw_text: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO provenance (id, material_id, source_type, source_name, ingested_at, raw_text) VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), material_id, source_type, source_name, datetime.utcnow().isoformat(), raw_text),
        )


def insert_price(material_id: str, price_usd: float, moq: int = 1, lead_time_days: int = 30, region: str = "global"):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO price_records (id, material_id, price_usd, moq, lead_time_days, region, recorded_at) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), material_id, price_usd, moq, lead_time_days, region, datetime.utcnow().isoformat()),
        )


def get_material_full(material_id: str) -> Optional[dict]:
    """Return a material with all its specs, compliance, provenance, price."""
    with get_conn() as conn:
        mat = conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        if not mat:
            return None
        specs = conn.execute("SELECT * FROM specs WHERE material_id = ?", (material_id,)).fetchall()
        comp = conn.execute("SELECT * FROM compliance WHERE material_id = ?", (material_id,)).fetchall()
        prov = conn.execute("SELECT * FROM provenance WHERE material_id = ?", (material_id,)).fetchall()
        price = conn.execute(
            "SELECT * FROM price_records WHERE material_id = ? ORDER BY recorded_at DESC LIMIT 1",
            (material_id,),
        ).fetchone()
        return {
            "id": mat["id"],
            "name": mat["name"],
            "part_number": mat["part_number"],
            "category": mat["category"],
            "supplier": mat["supplier"],
            "specs": [dict(s) for s in specs],
            "compliance": [dict(c) for c in comp],
            "provenance": [dict(p) for p in prov],
            "price": dict(price) if price else None,
        }


def list_all_materials() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name, category, supplier FROM materials ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def add_graph_edge(source_id: str, target_id: str, edge_type: str,
                   weight: float = 1.0, properties: dict = None) -> str:
    """Add a typed edge to the knowledge graph. Returns edge id."""
    eid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO graph_edges "
            "(id, source_id, target_id, edge_type, weight, properties, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (eid, source_id, target_id, edge_type, weight,
             json.dumps(properties or {}), datetime.utcnow().isoformat()),
        )
    return eid


def get_graph_neighbors(material_id: str, edge_type: str = None) -> list[dict]:
    """Return all nodes connected to material_id via graph edges."""
    with get_conn() as conn:
        if edge_type:
            rows = conn.execute(
                "SELECT * FROM graph_edges WHERE source_id=? AND edge_type=?",
                (material_id, edge_type),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM graph_edges WHERE source_id=? OR target_id=?",
                (material_id, material_id),
            ).fetchall()
    return [dict(r) for r in rows]


def add_feedback(query_material_id: str, substitute_material_id: str,
                 approved: bool, comment: str = None, score_override: float = None) -> str:
    fid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback "
            "(id, query_material_id, substitute_material_id, approved, score_override, comment, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (fid, query_material_id, substitute_material_id,
             1 if approved else 0, score_override, comment, datetime.utcnow().isoformat()),
        )
    return fid


def get_feedback_score(query_material_id: str, substitute_material_id: str) -> float:
    """Return feedback-adjusted score modifier (-0.2 to +0.2). 0.0 if no feedback."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT approved FROM feedback WHERE query_material_id=? AND substitute_material_id=?",
            (query_material_id, substitute_material_id),
        ).fetchall()
    if not rows:
        return 0.0
    avg = sum(r["approved"] for r in rows) / len(rows)
    return round((avg - 0.5) * 0.4, 3)   # maps [0,1] → [-0.2, +0.2]


def list_feedback(limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT f.*, m1.name as query_name, m2.name as sub_name "
            "FROM feedback f "
            "LEFT JOIN materials m1 ON f.query_material_id = m1.id "
            "LEFT JOIN materials m2 ON f.substitute_material_id = m2.id "
            "ORDER BY f.created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def log_audit(action: str, query_material_id: str = None,
              query_text: str = None, result_count: int = 0, metadata: dict = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (id, action, query_material_id, query_text, result_count, metadata, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), action, query_material_id, query_text,
             result_count, json.dumps(metadata or {}), datetime.utcnow().isoformat()),
        )


def list_audit_log(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT a.*, m.name as material_name FROM audit_log a "
            "LEFT JOIN materials m ON a.query_material_id = m.id "
            "ORDER BY a.created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_specs_map(material_id: str) -> dict:
    """Return {attribute: {value, unit}} for quick lookup."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT attribute, value, unit FROM specs WHERE material_id = ?", (material_id,)
        ).fetchall()
    return {r["attribute"]: {"value": r["value"], "unit": r["unit"]} for r in rows}
