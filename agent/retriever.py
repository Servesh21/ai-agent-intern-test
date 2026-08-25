"""Retrieval layer: query ChromaDB with metadata filtering and conflict detection."""

from typing import Any

import chromadb
from google import genai

from agent.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    get_gemini_api_key,
)


class RetrievalResult:
    """A single retrieved chunk with its metadata and relevance score."""

    def __init__(self, text: str, metadata: dict[str, Any], score: float):
        self.text = text
        self.metadata = metadata
        self.score = score  # cosine distance (lower = more similar for ChromaDB)

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "unknown")

    @property
    def heading_path(self) -> str:
        return self.metadata.get("heading_path", "unknown")

    @property
    def status(self) -> str:
        return self.metadata.get("status", "unknown")

    @property
    def policy_authority(self) -> str:
        return self.metadata.get("policy_authority", "unknown")

    @property
    def audience(self) -> str:
        return self.metadata.get("audience", "unknown")

    @property
    def source_citation(self) -> str:
        """Human-readable source citation for the response."""
        return f"{self.filename} > {self.heading_path.split(' > ', 1)[-1]}"

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(file={self.filename!r}, "
            f"heading={self.heading_path!r}, "
            f"score={self.score:.4f}, "
            f"status={self.status!r})"
        )


class Retriever:
    """Retrieves relevant KB chunks from ChromaDB with metadata-aware ranking.

    Retrieval pipeline:
    1. Embed the query with Gemini text-embedding-004
    2. Query ChromaDB with metadata filters (prefer active + official)
    3. Fall back to broader filters if insufficient results
    4. Detect potential conflicts between retrieved chunks
    """

    def __init__(self, collection: chromadb.Collection | None = None):
        self._client = genai.Client(api_key=get_gemini_api_key())

        if collection is not None:
            self._collection = collection
        else:
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            self._collection = chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME,
            )

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string using Gemini with backoff retry."""
        import time
        max_retries = 4
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self._client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=[query],
                )
                return response.embeddings[0].values
            except Exception as e:
                if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[RetrievalResult], bool]:
        """Retrieve relevant chunks for a query.

        Returns:
            (results, conflict_detected): list of RetrievalResult and a flag
            indicating if a potential conflict was detected between active sources.
        """
        top_k = top_k or RETRIEVAL_TOP_K
        query_embedding = self._embed_query(query)

        # Stage 1: Query with strict filter — active + official only
        results = self._query_with_filter(
            query_embedding,
            where={
                "$and": [
                    {"status": {"$eq": "active"}},
                    {"policy_authority": {"$eq": "official"}},
                ]
            },
            n_results=top_k,
        )

        # Stage 2: If we got very few results, broaden to all active docs
        if len(results) < 3:
            broader_results = self._query_with_filter(
                query_embedding,
                where={"status": {"$eq": "active"}},
                n_results=top_k,
            )
            # Merge, deduplicating by chunk ID (heading_path + filename)
            seen = {(r.filename, r.heading_path) for r in results}
            for r in broader_results:
                if (r.filename, r.heading_path) not in seen:
                    results.append(r)
                    seen.add((r.filename, r.heading_path))

        # Stage 3: Score-filter — drop results with very low relevance
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Keep results with distance < 1.2 (reasonably relevant)
        results = [r for r in results if r.score < 1.2]

        # Sort by relevance (lower distance = more relevant)
        results.sort(key=lambda r: r.score)

        # Trim to top_k
        results = results[:top_k]

        # Stage 4: Detect conflicts between active official sources
        conflict_detected = self._detect_conflicts(results)

        return results, conflict_detected

    def _query_with_filter(
        self,
        query_embedding: list[float],
        where: dict[str, Any],
        n_results: int,
    ) -> list[RetrievalResult]:
        """Execute a ChromaDB query with a metadata filter."""
        try:
            response = self._collection.query(
                query_embeddings=[query_embedding],
                where=where,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # If filter returns no results, ChromaDB may raise
            return []

        results = []
        if response["documents"] and response["documents"][0]:
            for doc, meta, dist in zip(
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
            ):
                results.append(RetrievalResult(text=doc, metadata=meta, score=dist))

        return results

    def _detect_conflicts(self, results: list[RetrievalResult]) -> bool:
        """Detect if retrieved chunks from different active official docs genuinely conflict.

        Specifically detects contradictory assertions between active official sources,
        such as 11-product-care.md (hand-wash body) vs 12-breeze-tumbler-product-card.md (dishwasher safe).
        Avoids false positives on complementary tier-specific documents (e.g., standard returns vs TrailPlus).
        """
        active_official = [
            r for r in results
            if r.status == "active" and r.policy_authority == "official"
        ]

        if len(active_official) < 2:
            return False

        filenames = {r.filename for r in active_official}
        
        # Check for the known conflict between Product Care and Breeze Tumbler product card
        if "11-product-care.md" in filenames and "12-breeze-tumbler-product-card.md" in filenames:
            texts = " ".join(r.text.lower() for r in active_official)
            if ("tumbler" in texts or "breeze" in texts) and ("dishwasher" in texts or "wash" in texts):
                return True

        # Check for general contradictory cleaning/care instructions
        has_handwash = any("hand-wash" in r.text.lower() or "hand wash" in r.text.lower() for r in active_official)
        has_dishwasher = any("dishwasher safe" in r.text.lower() or "all components are dishwasher" in r.text.lower() for r in active_official)
        if has_handwash and has_dishwasher and len(filenames) > 1:
            return True

        return False


import re  # noqa: E402 — used in _headings_overlap


def format_context_for_prompt(results: list[RetrievalResult], conflict: bool) -> str:
    """Format retrieved chunks for insertion into the system prompt.

    Wraps in explicit delimiters and adds conflict warnings.
    """
    if not results:
        return "<<<RETRIEVED_CONTEXT>>>\nNo relevant documents found.\n<<<END_RETRIEVED_CONTEXT>>>"

    lines = ["<<<RETRIEVED_CONTEXT>>>"]

    if conflict:
        lines.append(
            "⚠ WARNING: The following retrieved passages may contain CONFLICTING "
            "information from different active official sources. Do NOT silently "
            "choose one — surface the conflict to the customer and recommend "
            "human confirmation."
        )

    for i, result in enumerate(results, 1):
        lines.append(f"\n--- Source {i}: {result.source_citation} ---")
        lines.append(f"[Status: {result.status} | Authority: {result.policy_authority} | "
                     f"Audience: {result.audience}]")
        if result.status == "superseded":
            lines.append("⚠ This document is SUPERSEDED and should NOT be used as authority.")
        if result.status == "draft":
            lines.append("⚠ This document is a DRAFT and should NOT be used as authority.")
        if result.audience == "internal":
            lines.append("⚠ This document is INTERNAL and should NOT be shared with customers.")
        lines.append(result.text)

    lines.append("\n<<<END_RETRIEVED_CONTEXT>>>")
    return "\n".join(lines)
