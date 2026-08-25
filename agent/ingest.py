"""Knowledge base ingestion: parse markdown files, chunk by heading, embed, store in ChromaDB."""

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
import frontmatter
from google import genai

from agent.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    get_gemini_api_key,
)


def _parse_frontmatter(filepath: Path) -> tuple[dict[str, Any], str]:
    """Parse YAML front matter and body from a markdown file.

    Returns:
        (metadata_dict, body_text)
    """
    post = frontmatter.load(str(filepath))
    metadata = dict(post.metadata)
    body = post.content
    return metadata, body


def _chunk_by_heading(body: str, metadata: dict[str, Any], filename: str) -> list[dict]:
    """Split markdown body into chunks at H2 (##) boundaries.

    Each chunk carries:
      - text: the chunk content (including the heading line)
      - heading_path: "Doc Title > Section Heading"
      - All metadata fields from front matter
      - filename for citation
    """
    # Extract H1 title (first # line) for the heading path prefix
    h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    doc_title = h1_match.group(1).strip() if h1_match else metadata.get("title", filename)

    # Split on H2 boundaries, keeping the heading with its content
    # Pattern: split right before a line starting with ##
    sections = re.split(r"(?=^##\s)", body, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract the H2 heading if present
        h2_match = re.match(r"^##\s+(.+)$", section, re.MULTILINE)
        if h2_match:
            heading = h2_match.group(1).strip()
            heading_path = f"{doc_title} > {heading}"
        else:
            # This is the intro text before any H2 (preamble)
            heading = "Introduction"
            heading_path = f"{doc_title} > Introduction"

        # Skip chunks that are just the H1 title with no content
        # (the split may produce a chunk that's just "# Title\n")
        content_without_headings = re.sub(r"^#+\s+.*$", "", section, flags=re.MULTILINE).strip()
        if not content_without_headings:
            continue

        chunk = {
            "text": section,
            "heading_path": heading_path,
            "filename": filename,
            **{k: _serialize_metadata_value(k, v) for k, v in metadata.items()},
        }
        chunks.append(chunk)

    return chunks


def _serialize_metadata_value(key: str, value: Any) -> Any:
    """Convert metadata values to ChromaDB-compatible types (str, int, float, bool)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    # Convert dates, lists, etc. to string
    return str(value)


def _generate_chunk_id(filename: str, heading_path: str) -> str:
    """Generate a deterministic chunk ID from filename + heading path."""
    raw = f"{filename}::{heading_path}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def parse_all_documents(kb_dir: Path | None = None) -> list[dict]:
    """Parse all markdown files in the knowledge base directory.

    Returns a list of chunk dicts ready for embedding and storage.
    """
    kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
    all_chunks = []

    for filepath in sorted(kb_dir.glob("*.md")):
        metadata, body = _parse_frontmatter(filepath)
        filename = filepath.name
        chunks = _chunk_by_heading(body, metadata, filename)
        all_chunks.extend(chunks)

    return all_chunks


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Embed chunk texts using Gemini embedding model with rate-limit retries."""
    import time
    client = genai.Client(api_key=get_gemini_api_key())
    texts = [chunk["text"] for chunk in chunks]

    embeddings = []
    batch_size = 10  # Smaller batch size to respect per-minute quotas
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        max_retries = 5
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                for emb in response.embeddings:
                    embeddings.append(emb.values)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    if attempt < max_retries - 1:
                        print(f"  [Quota 429] Rate limit hit, sleeping {delay:.1f}s before retry...")
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        raise
                else:
                    raise
        # Polite delay between batches to stay under RPM limits
        time.sleep(1.0)

    return embeddings


def build_index(force_rebuild: bool = False) -> chromadb.Collection:
    """Build or load the ChromaDB collection.

    If the collection already exists and force_rebuild is False, returns it as-is.
    Otherwise, parses all docs, embeds them, and stores in ChromaDB.

    Returns the ChromaDB collection.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Check if collection exists and has data
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing_collections and not force_rebuild:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        if collection.count() > 0:
            print(f"Using existing ChromaDB collection ({collection.count()} chunks)")
            return collection
        # Collection exists but is empty — rebuild
    elif CHROMA_COLLECTION_NAME in existing_collections and force_rebuild:
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)

    # Create fresh collection
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Parse and chunk all documents
    print("Parsing knowledge base documents...")
    chunks = parse_all_documents()
    print(f"  -> {len(chunks)} chunks from {len(list(KNOWLEDGE_BASE_DIR.glob('*.md')))} documents")

    # Embed all chunks
    print("Embedding chunks with Gemini text-embedding-004...")
    embeddings = embed_chunks(chunks)
    print(f"  -> {len(embeddings)} embeddings generated")

    # Store in ChromaDB
    print("Storing in ChromaDB...")
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = _generate_chunk_id(chunk["filename"], chunk["heading_path"])
        ids.append(chunk_id)
        documents.append(chunk["text"])

        # Separate text from metadata for ChromaDB storage
        meta = {k: v for k, v in chunk.items() if k != "text"}
        metadatas.append(meta)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"  -> {collection.count()} chunks indexed in ChromaDB")
    return collection


if __name__ == "__main__":
    """Run standalone to build/rebuild the index."""
    import sys

    force = "--force" in sys.argv
    collection = build_index(force_rebuild=force)
    print(f"\nDone. Collection has {collection.count()} chunks.")

    # Print a summary of what was indexed
    results = collection.get(include=["metadatas"])
    status_counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        status = meta.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Status breakdown: {status_counts}")
