from datetime import datetime
from modules.memory.contracts.graph_models import (
    Provenance,
    Provenanced,
    UtteranceEvidence,
    SpatioTemporalRegion,
    State,
    Knowledge,
    GraphUpsertRequest,
    Entity,
)

def test_provenance_model():
    p = Provenance(source="test", confidence=0.9)
    assert p.source == "test"
    assert p.confidence == 0.9
    assert p.model_version is None

def test_provenanced_base():
    dt = datetime.now()
    p = Provenanced(
        tenant_id="t1",
        time_origin="media",
        provenance=Provenance(source="src"),
        ttl=3600.0,
        created_at=dt,
        memory_strength=1.5,
        forgetting_policy="normal",
    )
    assert p.tenant_id == "t1"
    assert p.provenance.source == "src"
    assert p.ttl == 3600.0
    assert p.created_at == dt
    assert p.memory_strength == 1.5
    assert p.forgetting_policy == "normal"

def test_utterance_evidence():
    utt = UtteranceEvidence(
        id="u1",
        raw_text="hello world",
        t_media_start=0.0,
        t_media_end=1.0,
        speaker_track_id="spk_1",
        lang="en",
        tenant_id="t1",
        memory_strength=2.0,
        forgetting_policy="persistent",
    )
    assert utt.id == "u1"
    assert utt.raw_text == "hello world"
    assert utt.speaker_track_id == "spk_1"
    assert utt.tenant_id == "t1"
    assert utt.memory_strength == 2.0
    assert utt.forgetting_policy == "persistent"

def test_spatiotemporal_region():
    region = SpatioTemporalRegion(
        id="reg1",
        name="Living Room",
        region_type="room",
        polygon="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        tenant_id="t1"
    )
    assert region.id == "reg1"
    assert region.name == "Living Room"
    assert region.region_type == "room"

def test_state():
    st = State(
        id="st1",
        subject_id="ent1",
        property="is_active",
        value="true",
        tenant_id="t1"
    )
    assert st.id == "st1"
    assert st.subject_id == "ent1"
    assert st.property == "is_active"
    assert st.value == "true"

def test_knowledge():
    k = Knowledge(
        id="k1",
        schema_version="v1",
        summary="A summary",
        data={"key": "value"},
        tenant_id="t1"
    )
    assert k.id == "k1"
    assert k.data["key"] == "value"
    assert k.summary == "A summary"

def test_graph_upsert_request_v03():
    req = GraphUpsertRequest(
        utterances=[
            UtteranceEvidence(id="u1", raw_text="hi", t_media_start=0, t_media_end=1)
        ],
        regions=[
            SpatioTemporalRegion(id="r1")
        ],
        states=[
            State(id="s1", subject_id="e1", property="p", value="v")
        ],
        knowledge=[
            Knowledge(id="k1")
        ],
        entities=[
            Entity(id="e1", type="Person")
        ]
    )
    assert len(req.utterances) == 1
    assert len(req.regions) == 1
    assert len(req.states) == 1
    assert len(req.knowledge) == 1
    assert len(req.entities) == 1
    assert req.utterances[0].raw_text == "hi"
