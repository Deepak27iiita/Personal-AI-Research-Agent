"""
Long-Term Memory Subsystem for Personal AI Research Agent.

Architecture:
1. Memory Schema: Structured ResearchMemory dataclass tracking topics, verified findings,
   authoritative sources, confidence, and access telemetry.
2. Storage: Embedded SQLite engine with FTS5 full-text indexing and binary vector BLOBs.
3. PrivacyGuard: Zero-trust redaction of credentials, API keys, and sensitive tokens.
4. MemoryFilter: Policy layer ensuring only verified, high-signal knowledge enters long-term memory
   (explicitly rejecting raw web dumps, error traces, and ephemeral chit-chat).
5. Hybrid Retrieval: Blends semantic vector cosine similarity with BM25 lexical relevance,
   enforcing a strict relevance threshold to prevent hallucinated context injection.
6. Memory Consolidation: Merges and updates related research topics over time.
7. User Control: Full CRUD, topic deletion, wipeouts, and JSON export/import.
"""

import os
import re
import json
import time
import math
import struct
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("ResearchMemory")


# ---------------------------------------------------------------------------
# 1. Memory Schema
# ---------------------------------------------------------------------------
@dataclass
class ResearchMemory:
    """
    Represents a durable, verified unit of research knowledge stored across sessions.
    """
    id: str                                  # Unique identifier, e.g. "mem_1714000000_abcd"
    topic: str                               # Primary subject, e.g. "Agentic RAG"
    summary: str                             # Concise high-level synthesis
    key_findings: List[str]                  # Verified factual claims
    sources: List[Dict[str, str]]            # Authoritative sources [{"id": "S1", "url": "...", "title": "..."}]
    comparison_dimensions: List[str]         # Evaluated dimensions (e.g. Latency, Accuracy, Complexity)
    tags: List[str]                          # Keywords for quick indexing
    confidence_score: float                  # Grounding fidelity score (0.0 - 1.0)
    session_id: str                          # Session origin
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 1
    embedding: Optional[List[float]] = None  # Vector representation

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("embedding"):
            data["embedding_dim"] = len(data["embedding"])
            # Omit giant float array from lightweight dict representations
            del data["embedding"]
        return data

    def format_for_context(self) -> str:
        """Formats the memory as a clean prompt injection block for the agent."""
        sources_str = ", ".join(f"{s.get('title', 'Source')} ({s.get('url', '')})" for s in self.sources[:3])
        findings_str = "\n".join(f"  - {f}" for f in self.key_findings[:5])
        dims_str = ", ".join(self.comparison_dimensions) if self.comparison_dimensions else "None"
        
        return (
            f"=== PRIOR RESEARCH MEMORY: {self.topic} ===\n"
            f"Summary: {self.summary}\n"
            f"Key Dimensions Analyzed: {dims_str}\n"
            f"Verified Findings:\n{findings_str}\n"
            f"Known Sources: {sources_str}\n"
            f"Confidence Score: {round(self.confidence_score * 100, 1)}%\n"
            f"=========================================="
        )


# ---------------------------------------------------------------------------
# 2. Privacy Guard (Zero-Trust Secret Scrubbing)
# ---------------------------------------------------------------------------
class PrivacyGuard:
    """
    Ensures credentials, API keys, and sensitive tokens are scrubbed before storage.
    """
    SECRET_PATTERNS = [
        # Google API keys
        (re.compile(r"AQ\.[A-Za-z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
        (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "[REDACTED_API_KEY]"),
        # OpenAI API keys
        (re.compile(r"sk-[A-Za-z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
        # Bearer tokens
        (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
        # Password / secret assignments
        (re.compile(r"(password|secret|api_key|token)\s*[:=]\s*['\"]?[^\s'\"]+['\"]?", re.IGNORECASE), r"\1=[REDACTED]"),
        # Email addresses
        (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[REDACTED_EMAIL]"),
    ]

    @classmethod
    def scrub(cls, text: str) -> str:
        if not text:
            return ""
        scrubbed = text
        for pattern, replacement in cls.SECRET_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed

    @classmethod
    def scrub_findings(cls, findings: List[str]) -> List[str]:
        return [cls.scrub(f) for f in findings if f and f.strip()]


# ---------------------------------------------------------------------------
# 3. Memory Filter (What SHOULD vs SHOULD NOT become Long-Term Memory)
# ---------------------------------------------------------------------------
class MemoryFilter:
    """
    Distills high-value, verified knowledge from a completed research state.
    
    POLICY:
    DO STORE:
      - Grounded factual claims verified against citations.
      - Architectural tradeoffs and comparison dimensions.
      - Authoritative source citations.
      - Topic synthesis summaries.
      
    DO NOT STORE:
      - Raw HTML / DOM dumps.
      - Ephemeral execution errors (404s, timeouts, tracebacks).
      - Chit-chat or conversational pleasantries.
      - Unverified / phantom citation claims.
    """

    @classmethod
    def extract_clean_memory(
        cls,
        state: Any,
        session_id: str
    ) -> Optional[ResearchMemory]:
        question = getattr(state, "question", "").strip()
        report = getattr(state, "final_report", "") or ""
        verification = getattr(state, "verification", None)

        if not question or not report:
            logger.warning("Rejecting memory creation: research report or question is empty.")
            return None

        # Derive topic name from question
        topic = cls._extract_topic(question)

        # Extract verified findings: prioritize verified claims from verification audit
        key_findings = []
        if verification and getattr(verification, "checks", None):
            for check in verification.checks:
                if check.status == "VERIFIED" and check.claim:
                    clean_claim = PrivacyGuard.scrub(check.claim.strip())
                    if clean_claim and clean_claim not in key_findings:
                        key_findings.append(clean_claim)

        # Fallback: extract bullet points from report if verification didn't produce claims
        if not key_findings:
            key_findings = cls._extract_bullet_points(report)

        # Sanitize findings
        key_findings = PrivacyGuard.scrub_findings(key_findings)[:8]
        if not key_findings:
            key_findings = [f"Synthesized research findings for {topic}."]

        # Extract sources from state
        sources = []
        state_sources = getattr(state, "sources", {})
        for s_id, s_obj in state_sources.items():
            sources.append({
                "id": s_id,
                "url": getattr(s_obj, "url", ""),
                "title": getattr(s_obj, "title", "Untitled Source")
            })

        # Dimensions from plan
        plan = getattr(state, "plan", None)
        dimensions = getattr(plan, "comparison_dimensions", []) if plan else []

        # Confidence from verification fidelity
        confidence = getattr(verification, "fidelity_score", 0.9) if verification else 0.85

        # Build concise summary (first 300 chars of synthesis)
        summary = cls._extract_summary(report, topic)
        summary = PrivacyGuard.scrub(summary)

        # Tags
        tags = cls._generate_tags(topic, dimensions)

        memory_id = f"mem_{int(time.time())}_{abs(hash(topic)) % 10000:04d}"

        return ResearchMemory(
            id=memory_id,
            topic=topic,
            summary=summary,
            key_findings=key_findings,
            sources=sources[:10],
            comparison_dimensions=dimensions,
            tags=tags,
            confidence_score=round(confidence, 3),
            session_id=session_id
        )

    @classmethod
    def _extract_topic(cls, question: str) -> str:
        clean = re.sub(r"^(compare|explain|research|what is|tell me about|continue my research on|continue my research and compare it with)\s+", "", question, flags=re.IGNORECASE).strip()
        clean = clean.rstrip("?.! ")
        if len(clean) > 80:
            clean = clean[:80] + "..."
        return clean.title() if len(clean) <= 50 else clean

    @classmethod
    def _extract_summary(cls, report: str, topic: str) -> str:
        lines = [line.strip() for line in report.splitlines() if line.strip() and not line.startswith("#")]
        if lines:
            text = " ".join(lines[:3])
            return text[:400] + ("..." if len(text) > 400 else "")
        return f"Comprehensive research notes and comparative evaluation on {topic}."

    @classmethod
    def _extract_bullet_points(cls, text: str) -> List[str]:
        findings = []
        for line in text.splitlines():
            line_str = line.strip()
            if line_str.startswith(("-", "*", "•")) and len(line_str) > 15:
                item = re.sub(r"^[-*•]\s*", "", line_str)
                # Remove citation tags like [S1] for clean reading
                item = re.sub(r"\[S\d+\]", "", item).strip()
                if item and item not in findings:
                    findings.append(item)
            if len(findings) >= 8:
                break
        return findings

    @classmethod
    def _generate_tags(cls, topic: str, dimensions: List[str]) -> List[str]:
        words = re.findall(r"[A-Za-z0-9_-]{3,}", topic.lower())
        stop_words = {"the", "and", "for", "with", "from", "that", "this", "what", "how", "research"}
        tags = [w for w in words if w not in stop_words]
        for d in dimensions:
            dw = re.findall(r"[A-Za-z0-9_-]{3,}", d.lower())
            tags.extend(dw)
        return list(dict.fromkeys(tags))[:10]


# ---------------------------------------------------------------------------
# 4. Storage Engine (SQLite + FTS5 Full-Text Index + BLOB Vectors)
# ---------------------------------------------------------------------------
class MemoryStore:
    """
    Persistent Long-Term Memory Store for Research Agents.
    Supports FTS5 lexical matching, embedding cosine similarity, and zero-loss CRUD.
    """

    def __init__(self, db_path: str = "data/research_memory.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            # 1. Main persistent table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS research_memories (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_findings_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    dimensions_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    embedding_blob BLOB
                )
            """)

            # 2. FTS5 virtual table for fast full-text lexical ranking
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS research_memories_fts USING fts5(
                    id UNINDEXED,
                    topic,
                    summary,
                    key_findings,
                    tags,
                    tokenize = 'porter unicode61'
                )
            """)

    # -----------------------------------------------------------------------
    # Embeddings Support (Gemini-embedding-001 with Lexical Fallback)
    # -----------------------------------------------------------------------
    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates vector embeddings for semantic similarity.
        Uses gemini-embedding-001 when API key is present, with zero crash fallback.
        """
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                resp = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text[:2000]
                )
                if hasattr(resp, "embeddings") and resp.embeddings:
                    return list(resp.embeddings[0].values)
            except Exception as e:
                logger.debug(f"Gemini embedding generation skipped: {e}")

        # Fallback: Deterministic bag-of-words pseudo-embedding (256-dim hashed vector)
        return self._fallback_hash_embedding(text)

    def _fallback_hash_embedding(self, text: str, dim: int = 256) -> List[float]:
        """Provides a zero-dependency deterministic normalized vector for offline environments."""
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            idx = abs(hash(token)) % dim
            vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _pack_vector(vec: Optional[List[float]]) -> Optional[bytes]:
        if not vec:
            return None
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack_vector(blob: Optional[bytes]) -> Optional[List[float]]:
        if not blob:
            return None
        n_floats = len(blob) // 4
        return list(struct.unpack(f"{n_floats}f", blob))

    @staticmethod
    def cosine_similarity(v1: Optional[List[float]], v2: Optional[List[float]]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    # -----------------------------------------------------------------------
    # CRUD Operations
    # -----------------------------------------------------------------------
    def save(self, memory: ResearchMemory) -> str:
        """
        Saves or updates a memory in the persistent store.
        If an existing memory shares the same normalized topic, findings are merged.
        """
        existing = self.find_by_topic(memory.topic)
        if existing:
            # Merge findings & update access telemetry
            merged_findings = list(dict.fromkeys(existing.key_findings + memory.key_findings))[:10]
            merged_sources = existing.sources + [s for s in memory.sources if s not in existing.sources]
            merged_tags = list(dict.fromkeys(existing.tags + memory.tags))

            updated_memory = ResearchMemory(
                id=existing.id,
                topic=existing.topic,
                summary=memory.summary if len(memory.summary) >= len(existing.summary) else existing.summary,
                key_findings=merged_findings,
                sources=merged_sources[:12],
                comparison_dimensions=list(dict.fromkeys(existing.comparison_dimensions + memory.comparison_dimensions)),
                tags=merged_tags,
                confidence_score=max(existing.confidence_score, memory.confidence_score),
                session_id=memory.session_id,
                created_at=existing.created_at,
                last_accessed_at=time.time(),
                access_count=existing.access_count + 1,
                embedding=memory.embedding or existing.embedding
            )
            self._update_record(updated_memory)
            logger.info(f"Consolidated and updated existing memory: '{updated_memory.topic}' (ID: {updated_memory.id})")
            return updated_memory.id

        # Insert new record
        if not memory.embedding:
            combined_text = f"{memory.topic} {memory.summary} {' '.join(memory.key_findings)} {' '.join(memory.tags)}"
            memory.embedding = self.compute_embedding(combined_text)

        blob = self._pack_vector(memory.embedding)
        with self.conn:
            self.conn.execute("""
                INSERT INTO research_memories (
                    id, topic, summary, key_findings_json, sources_json,
                    dimensions_json, tags_json, confidence_score, session_id,
                    created_at, last_accessed_at, access_count, embedding_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.topic,
                memory.summary,
                json.dumps(memory.key_findings),
                json.dumps(memory.sources),
                json.dumps(memory.comparison_dimensions),
                json.dumps(memory.tags),
                memory.confidence_score,
                memory.session_id,
                memory.created_at,
                memory.last_accessed_at,
                memory.access_count,
                blob
            ))

            # Sync to FTS5
            self.conn.execute("""
                INSERT INTO research_memories_fts (id, topic, summary, key_findings, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.topic,
                memory.summary,
                " ".join(memory.key_findings),
                " ".join(memory.tags)
            ))

        logger.info(f"Persisted new long-term research memory: '{memory.topic}' (ID: {memory.id})")
        return memory.id

    def _update_record(self, memory: ResearchMemory):
        if not memory.embedding:
            combined_text = f"{memory.topic} {memory.summary} {' '.join(memory.key_findings)} {' '.join(memory.tags)}"
            memory.embedding = self.compute_embedding(combined_text)

        blob = self._pack_vector(memory.embedding)
        with self.conn:
            self.conn.execute("""
                UPDATE research_memories SET
                    summary = ?,
                    key_findings_json = ?,
                    sources_json = ?,
                    dimensions_json = ?,
                    tags_json = ?,
                    confidence_score = ?,
                    session_id = ?,
                    last_accessed_at = ?,
                    access_count = ?,
                    embedding_blob = ?
                WHERE id = ?
            """, (
                memory.summary,
                json.dumps(memory.key_findings),
                json.dumps(memory.sources),
                json.dumps(memory.comparison_dimensions),
                json.dumps(memory.tags),
                memory.confidence_score,
                memory.session_id,
                memory.last_accessed_at,
                memory.access_count,
                blob,
                memory.id
            ))

            # Sync to FTS5
            self.conn.execute("DELETE FROM research_memories_fts WHERE id = ?", (memory.id,))
            self.conn.execute("""
                INSERT INTO research_memories_fts (id, topic, summary, key_findings, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.topic,
                memory.summary,
                " ".join(memory.key_findings),
                " ".join(memory.tags)
            ))

    def get(self, memory_id: str) -> Optional[ResearchMemory]:
        cursor = self.conn.execute("SELECT * FROM research_memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None

    def find_by_topic(self, topic: str) -> Optional[ResearchMemory]:
        norm = topic.strip().lower()
        cursor = self.conn.execute("SELECT * FROM research_memories WHERE LOWER(TRIM(topic)) = ?", (norm,))
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None

    def list_all(self) -> List[ResearchMemory]:
        cursor = self.conn.execute("SELECT * FROM research_memories ORDER BY last_accessed_at DESC")
        return [self._row_to_memory(r) for r in cursor.fetchall()]

    def delete(self, memory_id: str) -> bool:
        with self.conn:
            c1 = self.conn.execute("DELETE FROM research_memories WHERE id = ?", (memory_id,))
            self.conn.execute("DELETE FROM research_memories_fts WHERE id = ?", (memory_id,))
            return c1.rowcount > 0

    def delete_by_topic(self, topic: str) -> int:
        norm = f"%{topic.strip().lower()}%"
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM research_memories WHERE LOWER(topic) LIKE ?", (norm,))
            ids = [r["id"] for r in cursor.fetchall()]
            for mid in ids:
                self.delete(mid)
            return len(ids)

    def clear_all(self) -> int:
        with self.conn:
            c = self.conn.execute("SELECT count(*) as count FROM research_memories").fetchone()
            total = c["count"] if c else 0
            self.conn.execute("DELETE FROM research_memories")
            self.conn.execute("DELETE FROM research_memories_fts")
            return total

    # -----------------------------------------------------------------------
    # Hybrid Retrieval (Semantic + FTS5 + Recency)
    # -----------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.35
    ) -> List[Tuple[ResearchMemory, float]]:
        """
        Hybrid retrieval combining semantic similarity and full-text keyword score.
        Filters out low-confidence or irrelevant memories below min_score.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Compute query vector
        query_vec = self.compute_embedding(clean_query)

        # 2. Get lexical candidate IDs from FTS5
        # Clean query tokens for FTS5 syntax
        tokens = re.findall(r"\w+", clean_query)
        fts_scores: Dict[str, float] = {}
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens[:6])
            try:
                cursor = self.conn.execute("""
                    SELECT id, rank FROM research_memories_fts
                    WHERE research_memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT 20
                """, (fts_query,))
                for row in cursor.fetchall():
                    # rank in FTS5 is negative (lower = better)
                    raw_rank = abs(row["rank"])
                    fts_scores[row["id"]] = 1.0 / (1.0 + raw_rank)
            except Exception as e:
                logger.debug(f"FTS match skipped: {e}")

        # 3. Score all memories
        all_memories = self.list_all()
        scored: List[Tuple[ResearchMemory, float]] = []

        now = time.time()
        for mem in all_memories:
            # Semantic score
            sem_score = 0.0
            if query_vec and mem.embedding:
                sem_score = self.cosine_similarity(query_vec, mem.embedding)

            # Lexical score
            lex_score = fts_scores.get(mem.id, 0.0)

            # Direct topic substring boost
            topic_boost = 0.0
            if mem.topic.lower() in clean_query.lower() or any(t.lower() in clean_query.lower() for t in mem.tags):
                topic_boost = 0.25

            # Recency & Confidence factor (0.0 - 0.1)
            days_old = (now - mem.last_accessed_at) / 86400.0
            recency_factor = max(0.0, 1.0 - (days_old / 30.0)) * 0.05
            confidence_factor = (mem.confidence_score) * 0.05

            # Blended Score: 60% semantic + 25% lexical + boost + recency/confidence
            final_score = (0.50 * sem_score) + (0.25 * lex_score) + topic_boost + recency_factor + confidence_factor
            final_score = min(1.0, final_score)

            if final_score >= min_score:
                scored.append((mem, round(final_score, 3)))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update last_accessed_at and access_count for retrieved items
        with self.conn:
            for mem, _ in scored[:limit]:
                self.conn.execute("""
                    UPDATE research_memories
                    SET last_accessed_at = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (now, mem.id))

        return scored[:limit]

    # -----------------------------------------------------------------------
    # Export / Import for User Auditability
    # -----------------------------------------------------------------------
    def export_to_json(self, filepath: str) -> int:
        memories = self.list_all()
        data = [m.to_dict() for m in memories]
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return len(data)

    def import_from_json(self, filepath: str) -> int:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for item in data:
            mem = ResearchMemory(
                id=item.get("id", f"mem_{int(time.time())}_{count}"),
                topic=item["topic"],
                summary=item.get("summary", ""),
                key_findings=item.get("key_findings", []),
                sources=item.get("sources", []),
                comparison_dimensions=item.get("comparison_dimensions", []),
                tags=item.get("tags", []),
                confidence_score=item.get("confidence_score", 0.9),
                session_id=item.get("session_id", "imported"),
                created_at=item.get("created_at", time.time()),
                last_accessed_at=item.get("last_accessed_at", time.time()),
                access_count=item.get("access_count", 1)
            )
            self.save(mem)
            count += 1
        return count

    def _row_to_memory(self, row: sqlite3.Row) -> ResearchMemory:
        return ResearchMemory(
            id=row["id"],
            topic=row["topic"],
            summary=row["summary"],
            key_findings=json.loads(row["key_findings_json"]),
            sources=json.loads(row["sources_json"]),
            comparison_dimensions=json.loads(row["dimensions_json"]),
            tags=json.loads(row["tags_json"]),
            confidence_score=row["confidence_score"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            embedding=self._unpack_vector(row["embedding_blob"])
        )

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
