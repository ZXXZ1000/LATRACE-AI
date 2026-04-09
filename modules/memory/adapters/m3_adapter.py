from __future__ import annotations

from typing import Any, Dict, List, Tuple
import uuid
from modules.memory.contracts.memory_models import MemoryEntry, Edge


def build_entries_from_m3(parsed: Dict[str, Any], *, profile: Dict[str, Any] | None = None) -> Tuple[List[MemoryEntry], List[Edge]]:
    """Map m3-agent outputs (faces/voices/episodic/semantic) into unified entries/edges.

    `parsed` is expected to contain keys like: faces, voices, episodic, semantic, clip_id, timestamp.
    This is a placeholder. Real mapper should translate VideoGraph-like nodes to entries and relations.
    """
    entries: List[MemoryEntry] = []
    edges: List[Edge] = []

    clip_id = parsed.get("clip_id")
    ts = parsed.get("timestamp")

    for s in parsed.get("semantic", []):
        entries.append(
            MemoryEntry(
                kind="semantic",
                modality="text",
                contents=[str(s)],
                metadata={"source": "m3", "clip_id": clip_id, "timestamp": ts},
            )
        )
    # episodic entries with deterministic ids to allow edge wiring inside this batch
    episodic_entries: List[MemoryEntry] = []
    for e in parsed.get("episodic", []):
        ep = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="episodic",
            modality="text",
            contents=[str(e)],
            metadata={"source": "m3", "clip_id": clip_id, "timestamp": ts},
        )
        episodic_entries.append(ep)
        entries.append(ep)

    # optional entities: faces, voices, room, device
    face_entries: List[MemoryEntry] = []
    for f in parsed.get("faces", []) or []:
        fe = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="semantic",
            modality="image",
            contents=[str(f)],
            metadata={"source": "m3", "clip_id": clip_id, "timestamp": ts, "entity_type": "face"},
        )
        face_entries.append(fe)
        entries.append(fe)

    voice_entries: List[MemoryEntry] = []
    for v in parsed.get("voices", []) or []:
        ve = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="semantic",
            modality="audio",
            contents=[str(v)],
            metadata={"source": "m3", "clip_id": clip_id, "timestamp": ts, "entity_type": "voice"},
        )
        voice_entries.append(ve)
        entries.append(ve)

    # Optional structured entities
    room_node: MemoryEntry | None = None
    room = parsed.get("room")
    if room:
        room_node = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="semantic",
            modality="structured",
            contents=[str(room)],
            metadata={"source": "m3", "entity_type": "room"},
        )
        entries.append(room_node)

    device_node: MemoryEntry | None = None
    device = parsed.get("device")
    if device:
        device_node = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="semantic",
            modality="structured",
            contents=[str(device)],
            metadata={"source": "m3", "entity_type": "device"},
        )
        entries.append(device_node)

    # Build edges for each episodic
    for ep in episodic_entries:
        for fe in face_entries:
            edges.append(Edge(src_id=fe.id, dst_id=ep.id, rel_type="appears_in", weight=1.0))
        for ve in voice_entries:
            edges.append(Edge(src_id=ve.id, dst_id=ep.id, rel_type="said_by", weight=1.0))
        if room_node is not None:
            edges.append(Edge(src_id=ep.id, dst_id=room_node.id, rel_type="located_in", weight=1.0))
        if device_node is not None:
            # action execution trace linking device to episode
            edges.append(Edge(src_id=device_node.id, dst_id=ep.id, rel_type="executed", weight=1.0))

    return entries, edges
