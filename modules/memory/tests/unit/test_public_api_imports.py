from __future__ import annotations

def test_memory_public_api_imports():
    from modules import memory as mem

    # Contracts
    assert hasattr(mem, "MemoryEntry")
    assert hasattr(mem, "Edge")
    assert hasattr(mem, "SearchFilters")
    # Service & factory
    assert hasattr(mem, "MemoryService")
    assert hasattr(mem, "create_service")
