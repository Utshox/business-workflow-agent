"""Vector memory store using ChromaDB for workflow context retrieval."""

from __future__ import annotations

import os

import chromadb
from langchain_core.documents import Document


class MemoryStore:
    """Persistent vector memory backed by ChromaDB."""

    def __init__(self, persist_dir: str | None = None, collection_name: str = "workflow_memory"):
        persist_dir = persist_dir or os.getenv("CHROMADB_PATH", "./data/chroma")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = self._collection.count()

    def add(self, text: str, metadata: dict | None = None) -> str:
        """Add a document to the memory store. Returns the document ID."""
        doc_id = f"mem_{self._counter}"
        self._collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata if metadata else {"source": "workflow_agent"}],
        )
        self._counter += 1
        return doc_id

    def search(self, query: str, k: int = 3) -> list[Document]:
        """Search for similar documents. Returns LangChain Document objects."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )

        docs = []
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = 0
