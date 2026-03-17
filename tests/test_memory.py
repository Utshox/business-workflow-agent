"""Tests for the vector memory store."""

import tempfile

from src.memory.vector_store import MemoryStore


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(persist_dir=tmpdir, collection_name="test")
        store.add("Customer C001 had a billing issue resolved by applying a credit.")
        store.add("API rate limiting was increased for enterprise customers.")
        store.add("Password reset flow was updated in Q4.")

        results = store.search("billing problem", k=2)
        assert len(results) > 0
        assert "billing" in results[0].page_content.lower()


def test_empty_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(persist_dir=tmpdir, collection_name="test_empty")
        results = store.search("anything")
        assert results == []


def test_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(persist_dir=tmpdir, collection_name="test_clear")
        store.add("test document")
        store.clear()
        results = store.search("test")
        assert results == []
