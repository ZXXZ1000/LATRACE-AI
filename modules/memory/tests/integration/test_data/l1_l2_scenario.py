"""
L1-L2 Coverage Test Scenario Data

This module defines a controlled "Day in Life" scenario covering all 8 L1-L2 benchmark questions:

L1 (Direct Lookup):
- Q1: "上周五我去了哪些地方？" → Time + Place
- Q2: "我在视频里提到'人工智能'是在什么时候？" → Text + Time
- Q3: "画面里出现过红色的杯子吗？" → Visual Object
- Q4: "昨天下午跟我开会的人是谁？" → Event Type + Participants

L2 (Temporal & State):
- Q5: "我回家后做的第一件事是什么？" → NEXT_EVENT chain
- Q6: "我昨天玩手机玩了多久？" → Duration aggregation
- Q7: "我的车钥匙现在在哪？" → State tracking
- Q8: "出门前我锁门了吗？" → Temporal constraint

Timeline (2024-12-20 Friday):
- 08:00 Wake up (bedroom)
- 09:00 Lock door and leave
- 10:00-11:30 Meeting with Alice at cafe (discuss AI project)
- 12:00 Arrive library
- 12:30-13:00 Use phone (30 min)
- 14:00 Arrive home
- 14:05 Put car key in hallway
- 14:10 Change clothes (first thing after home)
- 14:30 Move car key to living room
- 15:00-15:30 Use phone (30 min)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from modules.memory.contracts.graph_models import (
    Entity,
    Event,
    Evidence,
    GraphEdge,
    GraphUpsertRequest,
    Place,
    TimeSlice,
    UtteranceEvidence,
)
from modules.memory.contracts.memory_models import Edge, MemoryEntry


# =============================================================================
# Constants
# =============================================================================

TENANT_ID = "test_tenant_l1_l2"
USER_ID = "user_test_001"
MEMORY_DOMAIN = "home"

# Base timestamp: 2024-12-20 00:00:00 UTC
BASE_DATE = datetime(2024, 12, 20, 0, 0, 0, tzinfo=timezone.utc)


def _dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    """Create datetime on 2024-12-20 with given time."""
    return BASE_DATE.replace(hour=hour, minute=minute, second=second)


# =============================================================================
# Entities
# =============================================================================

TEST_ENTITIES: List[Entity] = [
    # People
    Entity(
        id="entity_me",
        type="Person",
        name="Me",
        cluster_label="user_test_001",
        tenant_id=TENANT_ID,
    ),
    Entity(
        id="entity_alice",
        type="Person",
        name="Alice",
        cluster_label="alice",
        tenant_id=TENANT_ID,
    ),
    # Objects
    Entity(
        id="entity_car_key",
        type="Object",
        name="车钥匙",
        cluster_label="car_key",
        tenant_id=TENANT_ID,
    ),
    Entity(
        id="entity_phone",
        type="Object",
        name="手机",
        cluster_label="phone",
        tenant_id=TENANT_ID,
    ),
    Entity(
        id="entity_door",
        type="Object",
        name="门",
        cluster_label="door",
        tenant_id=TENANT_ID,
    ),
    Entity(
        id="entity_red_cup",
        type="Object",
        name="红色杯子",
        cluster_label="red_cup",
        tenant_id=TENANT_ID,
    ),
]

# =============================================================================
# Places
# =============================================================================

TEST_PLACES: List[Place] = [
    Place(id="place_bedroom", name="卧室", area_type="indoor", tenant_id=TENANT_ID),
    Place(id="place_hallway", name="玄关", area_type="indoor", tenant_id=TENANT_ID),
    Place(id="place_living_room", name="客厅", area_type="indoor", tenant_id=TENANT_ID),
    Place(id="place_cafe", name="咖啡厅", area_type="outdoor", tenant_id=TENANT_ID),
    Place(id="place_library", name="图书馆", area_type="outdoor", tenant_id=TENANT_ID),
    Place(id="place_home", name="家", area_type="indoor", tenant_id=TENANT_ID),
]

# =============================================================================
# Time Slices
# =============================================================================

TEST_TIMESLICES: List[TimeSlice] = [
    # Day granularity
    TimeSlice(
        id="ts_day_20241220",
        kind="day",
        t_abs_start=_dt(0, 0, 0),
        t_abs_end=_dt(23, 59, 59),
        granularity_level=1,
        tenant_id=TENANT_ID,
    ),
    # Hour granularity for key hours
    TimeSlice(
        id="ts_hour_10",
        kind="hour",
        t_abs_start=_dt(10, 0, 0),
        t_abs_end=_dt(10, 59, 59),
        granularity_level=2,
        parent_id="ts_day_20241220",
        tenant_id=TENANT_ID,
    ),
    TimeSlice(
        id="ts_hour_14",
        kind="hour",
        t_abs_start=_dt(14, 0, 0),
        t_abs_end=_dt(14, 59, 59),
        granularity_level=2,
        parent_id="ts_day_20241220",
        tenant_id=TENANT_ID,
    ),
]

# =============================================================================
# Events
# =============================================================================

TEST_EVENTS: List[Event] = [
    # 08:00 Wake up
    Event(
        id="event_001_wake_up",
        summary="起床",
        event_type="routine",
        action="wake_up",
        t_abs_start=_dt(8, 0, 0),
        t_abs_end=_dt(8, 0, 0),
        tenant_id=TENANT_ID,
    ),
    # 09:00 Lock door and leave
    Event(
        id="event_002_lock_door",
        summary="锁门出门",
        event_type="routine",
        action="lock",
        t_abs_start=_dt(9, 0, 0),
        t_abs_end=_dt(9, 0, 0),
        tenant_id=TENANT_ID,
    ),
    # 09:30 Arrive at cafe
    Event(
        id="event_003_arrive_cafe",
        summary="到达咖啡厅",
        event_type="travel",
        action="arrive",
        t_abs_start=_dt(9, 30, 0),
        t_abs_end=_dt(9, 30, 0),
        tenant_id=TENANT_ID,
    ),
    # 10:00-11:30 Meeting with Alice
    Event(
        id="event_004_meeting",
        summary="与Alice在咖啡厅开会，讨论人工智能项目",
        event_type="meeting",
        action="meeting",
        t_abs_start=_dt(10, 0, 0),
        t_abs_end=_dt(11, 30, 0),
        tenant_id=TENANT_ID,
    ),
    # 12:00 Arrive at library
    Event(
        id="event_005_arrive_library",
        summary="到达图书馆",
        event_type="travel",
        action="arrive",
        t_abs_start=_dt(12, 0, 0),
        t_abs_end=_dt(12, 0, 0),
        tenant_id=TENANT_ID,
    ),
    # 12:30-13:00 Use phone (30 min)
    Event(
        id="event_006_phone_1",
        summary="在图书馆看手机",
        event_type="leisure",
        action="use_phone",
        t_abs_start=_dt(12, 30, 0),
        t_abs_end=_dt(13, 0, 0),
        tenant_id=TENANT_ID,
    ),
    # 14:00 Arrive home
    Event(
        id="event_007_arrive_home",
        summary="回到家",
        event_type="travel",
        action="arrive",
        t_abs_start=_dt(14, 0, 0),
        t_abs_end=_dt(14, 0, 0),
        tenant_id=TENANT_ID,
    ),
    # 14:05 Put car key in hallway
    Event(
        id="event_008_put_key",
        summary="把车钥匙放在玄关",
        event_type="action",
        action="put",
        t_abs_start=_dt(14, 5, 0),
        t_abs_end=_dt(14, 5, 0),
        tenant_id=TENANT_ID,
    ),
    # 14:10 Change clothes (first thing after home)
    Event(
        id="event_009_change_clothes",
        summary="换衣服",
        event_type="routine",
        action="change_clothes",
        t_abs_start=_dt(14, 10, 0),
        t_abs_end=_dt(14, 15, 0),
        tenant_id=TENANT_ID,
    ),
    # 14:30 Move car key to living room
    Event(
        id="event_010_move_key",
        summary="把车钥匙拿到客厅",
        event_type="action",
        action="move",
        t_abs_start=_dt(14, 30, 0),
        t_abs_end=_dt(14, 30, 0),
        tenant_id=TENANT_ID,
    ),
    # 15:00-15:30 Use phone (30 min)
    Event(
        id="event_011_phone_2",
        summary="在家看手机",
        event_type="leisure",
        action="use_phone",
        t_abs_start=_dt(15, 0, 0),
        t_abs_end=_dt(15, 30, 0),
        tenant_id=TENANT_ID,
    ),
]

# =============================================================================
# Utterance Evidence (ASR)
# =============================================================================

TEST_UTTERANCES: List[UtteranceEvidence] = [
    UtteranceEvidence(
        id="utt_001",
        raw_text="我们今天讨论一下人工智能项目的进展",
        t_media_start=1000.0,  # 10:05
        t_media_end=1010.0,
        speaker_track_id="entity_me",
        tenant_id=TENANT_ID,
    ),
    UtteranceEvidence(
        id="utt_002",
        raw_text="人工智能在这个领域的应用前景很广阔",
        t_media_start=1600.0,  # 10:15
        t_media_end=1615.0,
        speaker_track_id="entity_alice",
        tenant_id=TENANT_ID,
    ),
    UtteranceEvidence(
        id="utt_003",
        raw_text="我把车钥匙放在玄关了",
        t_media_start=2000.0,  # 14:05
        t_media_end=2005.0,
        speaker_track_id="entity_me",
        tenant_id=TENANT_ID,
    ),
]

# =============================================================================
# Visual Evidence
# =============================================================================

TEST_EVIDENCES: List[Evidence] = [
    Evidence(
        id="ev_red_cup",
        source_id="seg_cafe_1030",
        algorithm="yolo",
        algorithm_version="v8",
        confidence=0.92,
        text="red cup on table",
        subtype="object",
        extras={"name": "cup", "color": "red", "bbox": [100, 200, 150, 250]},
        tenant_id=TENANT_ID,
    ),
]

# =============================================================================
# Graph Edges
# =============================================================================

TEST_EDGES: List[GraphEdge] = [
    # INVOLVES: Event participants
    GraphEdge(src_id="event_004_meeting", dst_id="entity_me", rel_type="INVOLVES", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_004_meeting", dst_id="entity_alice", rel_type="INVOLVES", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_002_lock_door", dst_id="entity_door", rel_type="INVOLVES", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_008_put_key", dst_id="entity_car_key", rel_type="INVOLVES", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_010_move_key", dst_id="entity_car_key", rel_type="INVOLVES", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_006_phone_1", dst_id="entity_phone", rel_type="INVOLVES", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_011_phone_2", dst_id="entity_phone", rel_type="INVOLVES", tenant_id=TENANT_ID),
    
    # OCCURS_AT: Event location
    GraphEdge(src_id="event_001_wake_up", dst_id="place_bedroom", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_002_lock_door", dst_id="place_hallway", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_003_arrive_cafe", dst_id="place_cafe", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_004_meeting", dst_id="place_cafe", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_005_arrive_library", dst_id="place_library", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_006_phone_1", dst_id="place_library", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_007_arrive_home", dst_id="place_home", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_008_put_key", dst_id="place_hallway", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_009_change_clothes", dst_id="place_bedroom", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_010_move_key", dst_id="place_living_room", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_011_phone_2", dst_id="place_living_room", rel_type="OCCURS_AT", tenant_id=TENANT_ID),
    
    # NEXT_EVENT: Temporal chain (after arriving home)
    GraphEdge(src_id="event_007_arrive_home", dst_id="event_008_put_key", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_008_put_key", dst_id="event_009_change_clothes", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_009_change_clothes", dst_id="event_010_move_key", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_010_move_key", dst_id="event_011_phone_2", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    
    # Full day chain
    GraphEdge(src_id="event_001_wake_up", dst_id="event_002_lock_door", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_002_lock_door", dst_id="event_003_arrive_cafe", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_003_arrive_cafe", dst_id="event_004_meeting", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_004_meeting", dst_id="event_005_arrive_library", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_005_arrive_library", dst_id="event_006_phone_1", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_006_phone_1", dst_id="event_007_arrive_home", rel_type="NEXT_EVENT", tenant_id=TENANT_ID),
    
    # SUPPORTED_BY: Event evidence
    GraphEdge(src_id="event_004_meeting", dst_id="utt_001", rel_type="SUPPORTED_BY", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_004_meeting", dst_id="utt_002", rel_type="SUPPORTED_BY", tenant_id=TENANT_ID),
    GraphEdge(src_id="event_008_put_key", dst_id="utt_003", rel_type="SUPPORTED_BY", tenant_id=TENANT_ID),
    
    # SPOKEN_BY: Speaker attribution
    GraphEdge(src_id="utt_001", dst_id="entity_me", rel_type="SPOKEN_BY", tenant_id=TENANT_ID),
    GraphEdge(src_id="utt_002", dst_id="entity_alice", rel_type="SPOKEN_BY", tenant_id=TENANT_ID),
    GraphEdge(src_id="utt_003", dst_id="entity_me", rel_type="SPOKEN_BY", tenant_id=TENANT_ID),
]

# =============================================================================
# MemoryEntry format (for InMem store tests)
# =============================================================================


def create_memory_entries() -> List[MemoryEntry]:
    """Create MemoryEntry objects for InMem store testing."""
    entries: List[MemoryEntry] = []
    
    # Events as episodic memories
    for event in TEST_EVENTS:
        ts = event.t_abs_start.timestamp() if event.t_abs_start else 0.0
        entry = MemoryEntry(
            id=event.id,
            kind="episodic",
            modality="text",
            contents=[event.summary],
            metadata={
                "tenant_id": TENANT_ID,
                "user_id": [USER_ID],
                "memory_domain": MEMORY_DOMAIN,
                "timestamp": ts,
                "event_type": event.event_type,
                "action": event.action,
                "t_abs_start": event.t_abs_start.isoformat() if event.t_abs_start else None,
                "t_abs_end": event.t_abs_end.isoformat() if event.t_abs_end else None,
            },
        )
        entries.append(entry)
    
    # Utterances as episodic memories
    for utt in TEST_UTTERANCES:
        entry = MemoryEntry(
            id=utt.id,
            kind="episodic",
            modality="text",
            contents=[utt.raw_text],
            metadata={
                "tenant_id": TENANT_ID,
                "user_id": [USER_ID],
                "memory_domain": MEMORY_DOMAIN,
                "timestamp": utt.t_media_start,
                "speaker_track_id": utt.speaker_track_id,
                "source": "asr",
            },
        )
        entries.append(entry)
    
    # Entities as semantic memories
    for entity in TEST_ENTITIES:
        entry = MemoryEntry(
            id=entity.id,
            kind="semantic",
            modality="structured",
            contents=[entity.name or entity.id],
            metadata={
                "tenant_id": TENANT_ID,
                "user_id": [USER_ID],
                "memory_domain": MEMORY_DOMAIN,
                "entity_type": entity.type,
            },
        )
        entries.append(entry)
    
    # Places as semantic memories
    for place in TEST_PLACES:
        entry = MemoryEntry(
            id=place.id,
            kind="semantic",
            modality="structured",
            contents=[place.name],
            metadata={
                "tenant_id": TENANT_ID,
                "user_id": [USER_ID],
                "memory_domain": MEMORY_DOMAIN,
                "entity_type": "place",
                "area_type": place.area_type,
            },
        )
        entries.append(entry)
    
    return entries


def create_memory_edges() -> List[Edge]:
    """Create Edge objects for InMem store testing."""
    edges: List[Edge] = []
    for ge in TEST_EDGES:
        edge = Edge(
            src_id=ge.src_id,
            dst_id=ge.dst_id,
            rel_type=ge.rel_type.lower(),
            weight=ge.weight or 1.0,
        )
        edges.append(edge)
    return edges


def create_graph_upsert_request() -> GraphUpsertRequest:
    """Create a GraphUpsertRequest for Neo4j testing."""
    return GraphUpsertRequest(
        entities=TEST_ENTITIES,
        events=TEST_EVENTS,
        places=TEST_PLACES,
        time_slices=TEST_TIMESLICES,
        utterances=TEST_UTTERANCES,
        evidences=TEST_EVIDENCES,
        edges=TEST_EDGES,
    )


# =============================================================================
# Expected Results for Validation
# =============================================================================

EXPECTED_RESULTS: Dict[str, Any] = {
    # Q1: Places on Friday
    "q1_places": {"咖啡厅", "图书馆"},  # outdoor places
    "q1_all_places": {"咖啡厅", "图书馆", "家", "卧室", "玄关", "客厅"},
    
    # Q2: When "人工智能" was mentioned
    "q2_ai_mention_times": [
        {"text": "我们今天讨论一下人工智能项目的进展", "time": "10:05"},
        {"text": "人工智能在这个领域的应用前景很广阔", "time": "10:15"},
    ],
    
    # Q3: Red cup
    "q3_red_cup_exists": True,
    "q3_red_cup_location": "咖啡厅",
    
    # Q4: Meeting participants
    "q4_meeting_participants": ["Alice"],  # excluding "Me"
    
    # Q5: First thing after arriving home
    "q5_first_action_after_home": "把车钥匙放在玄关",
    
    # Q6: Phone usage duration
    "q6_phone_duration_minutes": 60,  # 30 + 30
    
    # Q7: Car key current location
    "q7_key_location": "客厅",  # last known location
    
    # Q8: Door locked before leaving
    "q8_door_locked": True,
}


__all__ = [
    "TENANT_ID",
    "USER_ID",
    "MEMORY_DOMAIN",
    "BASE_DATE",
    "TEST_ENTITIES",
    "TEST_PLACES",
    "TEST_TIMESLICES",
    "TEST_EVENTS",
    "TEST_UTTERANCES",
    "TEST_EVIDENCES",
    "TEST_EDGES",
    "EXPECTED_RESULTS",
    "create_memory_entries",
    "create_memory_edges",
    "create_graph_upsert_request",
]









