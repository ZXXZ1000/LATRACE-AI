from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.upsert_payload = None
        self.last_segments_args = None
        self.last_timeline_args = None
        self.last_events_args = None
        self.last_places_args = None
        self.last_timeslices_args = None
        self.last_search_args = None
        self.last_build_event_rel_args = None
        self.last_build_timeslices_args = None
        self.last_build_cooccurs_args = None
        self.last_build_cooccurs_event_args = None
        self.last_build_first_meetings_args = None
        self.last_purge_args = None
        self.last_event_detail_args = None
        self.last_place_detail_args = None
        self.last_first_meeting_args = None
        self.last_event_explain_args = None
        self.segment_response: list[dict] = [{"id": "seg-1"}]
        self.timeline_response: list[dict] = [{"segment_id": "seg-1", "evidence_id": "face-1"}]
        self.events_response: list[dict] = [{"id": "event-1"}]
        self.places_response: list[dict] = [{"id": "place-1"}]
        self.timeslices_response: list[dict] = [{"id": "ts-1"}]
        self.event_detail_response: dict = {"id": "event-1"}
        self.place_detail_response: dict = {"id": "place-1"}
        self.first_meeting_response: dict = {
            "found": False,
            "event_id": None,
            "t_abs_start": None,
            "place_id": None,
            "summary": None,
            "evidence_ids": [],
        }
        self.event_evidence_response: dict = {
            "event": None,
            "entities": [],
            "places": [],
            "timeslices": [],
            "evidences": [],
            "utterances": [],
            "utterance_speakers": [],
        }
        self.search_response: dict = {"query": "", "items": [{"event_id": "event-1"}]}
        self.purge_response: dict = {"segments": 1}

    async def upsert(self, req):
        self.upsert_payload = req

    async def search_events_v1(self, **kwargs):
        self.last_search_args = kwargs
        return dict(self.search_response)

    async def list_segments(self, **kwargs):
        self.last_segments_args = kwargs
        return list(self.segment_response)

    async def entity_timeline(self, **kwargs):
        self.last_timeline_args = kwargs
        return list(self.timeline_response)

    async def list_events(self, **kwargs):
        self.last_events_args = kwargs
        return list(self.events_response)

    async def list_places(self, **kwargs):
        self.last_places_args = kwargs
        return list(self.places_response)

    async def list_time_slices(self, **kwargs):
        self.last_timeslices_args = kwargs
        return list(self.timeslices_response)

    async def build_event_relations(self, **kwargs):
        self.last_build_event_rel_args = kwargs
        return {"next_event": 3, "causes": 1}

    async def build_time_slices_from_segments(self, **kwargs):
        self.last_build_timeslices_args = kwargs
        return {"timeslices": 2, "edges": 4}

    async def build_cooccurs_from_timeslices(self, **kwargs):
        self.last_build_cooccurs_args = kwargs
        return {"co_occurs": 5}

    async def build_cooccurs_from_events(self, **kwargs):
        self.last_build_cooccurs_event_args = kwargs
        return {"co_occurs": 7}

    async def build_first_meetings(self, **kwargs):
        self.last_build_first_meetings_args = kwargs
        return {"first_meet": 7}

    async def purge_source(self, **kwargs):
        self.last_purge_args = kwargs
        return dict(self.purge_response)

    async def event_detail(self, **kwargs):
        self.last_event_detail_args = kwargs
        return dict(self.event_detail_response)

    async def place_detail(self, **kwargs):
        self.last_place_detail_args = kwargs
        return dict(self.place_detail_response)

    async def explain_first_meeting(self, **kwargs):
        self.last_first_meeting_args = kwargs
        return dict(self.first_meeting_response)

    async def explain_event_evidence(self, **kwargs):
        self.last_event_explain_args = kwargs
        return dict(self.event_evidence_response)


def _setup(monkeypatch):
    from modules.memory.api import server as srv

    stub = _GraphStub()
    monkeypatch.setattr(srv, "graph_svc", stub)
    monkeypatch.setattr(
        srv,
        "_auth_settings",
        lambda: {
            "enabled": False,
            "header": "X-API-Token",
            "token": "",
            "tenant_id": "",
            "token_map": {},
        },
    )
    client = TestClient(srv.app)
    return srv, stub, client


def test_graph_upsert_requires_tenant_header(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    resp = client.post("/graph/v0/upsert", json={"segments": []})
    assert resp.status_code == 400
    assert stub.upsert_payload is None


def test_graph_upsert_injects_tenant_id(monkeypatch):
    from modules.memory.contracts.graph_models import GraphUpsertRequest

    _, stub, client = _setup(monkeypatch)
    body = {
        "segments": [
            {
                "id": "seg-a",
                "tenant_id": "legacy",
                "source_id": "demo.mp4",
                "t_media_start": 0.0,
                "t_media_end": 1.0,
            }
        ]
    }

    resp = client.post("/graph/v0/upsert", headers={"X-Tenant-ID": "tenant-a"}, json=body)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert isinstance(stub.upsert_payload, GraphUpsertRequest)
    assert all(seg.tenant_id == "tenant-a" for seg in stub.upsert_payload.segments)


def test_graph_upsert_uses_token_mapping(monkeypatch):
    from modules.memory.contracts.graph_models import GraphUpsertRequest
    from modules.memory.api import server as srv

    stub = _GraphStub()
    monkeypatch.setattr(srv, "graph_svc", stub)
    monkeypatch.setattr(
        srv,
        "_auth_settings",
        lambda: {
            "enabled": True,
            "header": "X-API-Token",
            "token": "legacy",
            "tenant_id": "",
            "token_map": {"token-1": "tenant-mapped"},
        },
    )
    client = TestClient(srv.app)
    body = {"segments": [{"id": "seg-a", "source_id": "demo.mp4", "t_media_start": 0.0, "t_media_end": 1.0}]}
    resp = client.post("/graph/v0/upsert", headers={"X-API-Token": "token-1"}, json=body)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert isinstance(stub.upsert_payload, GraphUpsertRequest)
    assert all(seg.tenant_id == "tenant-mapped" for seg in stub.upsert_payload.segments)

def test_graph_search_v1_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.search_response = {"query": "hello", "items": [{"event_id": "event-1", "score": 1.2}]}
    resp = client.post(
        "/graph/v1/search",
        headers={"X-Tenant-ID": "tenant-s"},
        json={"query": "hello", "topk": 5, "source_id": "demo.mp4", "include_evidence": True},
    )
    assert resp.status_code == 200
    assert resp.json() == stub.search_response
    assert stub.last_search_args == {
        "tenant_id": "tenant-s",
        "query": "hello",
        "topk": 5,
        "source_id": "demo.mp4",
        "include_evidence": True,
    }


def test_graph_admin_purge_source_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.purge_response = {"segments": 2, "events": 1}
    resp = client.post(
        "/graph/v0/admin/purge_source",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"source_id": "demo.mp4", "delete_orphans": True},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == stub.purge_response
    assert stub.last_purge_args == {
        "tenant_id": "tenant-x",
        "source_id": "demo.mp4",
        "delete_orphans": True,
    }


def test_graph_segments_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.segment_response = [
        {
            "id": "seg-a",
            "source_id": "demo.mp4",
            "t_media_start": 0.0,
            "t_media_end": 1.0,
        }
    ]
    resp = client.get(
        "/graph/v0/segments?source_id=demo.mp4&limit=5",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == stub.segment_response
    assert stub.last_segments_args == {
        "tenant_id": "tenant-b",
        "source_id": "demo.mp4",
        "start": None,
        "end": None,
        "modality": None,
        "limit": 5,
    }


def test_graph_entity_timeline_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.timeline_response = [
        {
            "segment_id": "seg-a",
            "source_id": "demo.mp4",
            "evidence_id": "face-1",
            "t_media_start": 0.0,
            "t_media_end": 1.0,
        }
    ]
    resp = client.get(
        "/graph/v0/entities/person_1/timeline",
        headers={"X-Tenant-ID": "tenant-c"},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == stub.timeline_response
    assert stub.last_timeline_args == {
        "tenant_id": "tenant-c",
        "entity_id": "person_1",
        "limit": 200,
    }


def test_graph_events_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.events_response = [
        {
            "id": "event-1",
            "segment_id": "seg-a",
            "summary": "Person enters room",
            "entity_ids": ["person_1"],
            "place_ids": ["place-1"],
        }
    ]
    resp = client.get(
        "/graph/v0/events?segment_id=seg-a&entity_id=person_1&place_id=place-1&source_id=demo.mp4&relation=CAUSES&layer=hypothesis&status=candidate&limit=50",
        headers={"X-Tenant-ID": "tenant-d"},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == stub.events_response
    assert stub.last_events_args == {
        "tenant_id": "tenant-d",
        "segment_id": "seg-a",
        "entity_id": "person_1",
        "place_id": "place-1",
        "source_id": "demo.mp4",
        "relation": "CAUSES",
        "layer": "hypothesis",
        "status": "candidate",
        "limit": 50,
    }


def test_graph_places_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.places_response = [
        {
            "id": "place-1",
            "name": "Lobby",
            "segment_ids": ["seg-a"],
            "source_ids": ["demo.mp4"],
        }
    ]
    resp = client.get(
        "/graph/v0/places?name=Lobby&segment_id=seg-a&covers_timeslice=ts-1&limit=75",
        headers={"X-Tenant-ID": "tenant-e"},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == stub.places_response
    assert stub.last_places_args == {
        "tenant_id": "tenant-e",
        "name": "Lobby",
        "segment_id": "seg-a",
        "covers_timeslice": "ts-1",
        "limit": 75,
    }


def test_graph_event_detail_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.event_detail_response = {
        "id": "event-123",
        "segments": [{"id": "seg-a"}],
        "entity_ids": ["entity-1"],
        "evidence_ids": ["ev-1"],
        "timeslice_ids": ["ts-1"],
        "relations": [{"type": "NEXT_EVENT", "target_event_id": "event-124"}],
    }
    resp = client.get(
        "/graph/v0/events/event-123",
        headers={"X-Tenant-ID": "tenant-f"},
    )
    assert resp.status_code == 200
    assert resp.json()["item"]["id"] == "event-123"
    assert stub.last_event_detail_args == {"tenant_id": "tenant-f", "event_id": "event-123"}


def test_graph_place_detail_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.place_detail_response = {
        "id": "place-xyz",
        "event_ids": ["event-1"],
        "segments": [{"id": "seg-a"}],
        "timeslice_ids": ["ts-1"],
    }
    resp = client.get(
        "/graph/v0/places/place-xyz",
        headers={"X-Tenant-ID": "tenant-g"},
    )
    assert resp.status_code == 200
    assert resp.json()["item"]["id"] == "place-xyz"
    assert stub.last_place_detail_args == {"tenant_id": "tenant-g", "place_id": "place-xyz"}


def test_graph_timeslices_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.timeslices_response = [{"id": "ts-1", "segment_ids": ["seg-a"]}]
    resp = client.get(
        "/graph/v0/timeslices?kind=media&covers_segment=seg-a&covers_event=ev-1&limit=10",
        headers={"X-Tenant-ID": "tenant-h"},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == stub.timeslices_response
    assert stub.last_timeslices_args == {
        "tenant_id": "tenant-h",
        "kind": "media",
        "covers_segment": "seg-a",
        "covers_event": "ev-1",
        "limit": 10,
    }


def test_admin_build_event_relations(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    resp = client.post(
        "/graph/v0/admin/build_event_relations",
        headers={"X-Tenant-ID": "tenant-i"},
        json={"source_id": "src", "place_id": "pl", "limit": 10, "create_causes": False},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == {"next_event": 3, "causes": 1}
    assert stub.last_build_event_rel_args == {
        "tenant_id": "tenant-i",
        "source_id": "src",
        "place_id": "pl",
        "limit": 10,
        "create_causes": False,
    }


def test_admin_build_timeslices(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    resp = client.post(
        "/graph/v0/admin/build_timeslices",
        headers={"X-Tenant-ID": "tenant-j"},
        json={"window_seconds": 1800, "source_id": "src", "modality": "video"},
    )
    assert resp.status_code == 200
    assert stub.last_build_timeslices_args["tenant_id"] == "tenant-j"
    assert stub.last_build_timeslices_args["window_seconds"] == 1800.0
    assert stub.last_build_timeslices_args["source_id"] == "src"
    assert stub.last_build_timeslices_args["modality"] == "video"


def test_admin_build_cooccurs(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    resp = client.post(
        "/graph/v0/admin/build_cooccurs",
        headers={"X-Tenant-ID": "tenant-k"},
        json={"min_weight": 2},
    )
    assert resp.status_code == 200
    assert stub.last_build_cooccurs_args == {"tenant_id": "tenant-k", "min_weight": 2.0}


def test_admin_build_cooccurs_event(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    resp = client.post(
        "/graph/v0/admin/build_cooccurs",
        headers={"X-Tenant-ID": "tenant-k"},
        json={"min_weight": 3, "mode": "event"},
    )
    assert resp.status_code == 200
    assert stub.last_build_cooccurs_event_args == {"tenant_id": "tenant-k", "min_weight": 3.0}


def test_admin_build_first_meetings(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    resp = client.post(
        "/graph/v0/admin/build_first_meetings",
        headers={"X-Tenant-ID": "tenant-f"},
        json={"limit": 123},
    )
    assert resp.status_code == 200
    assert stub.last_build_first_meetings_args == {"tenant_id": "tenant-f", "limit": 123}


def test_graph_explain_first_meeting_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.first_meeting_response = {
        "found": True,
        "event_id": "event-42",
        "t_abs_start": "2025-01-01T00:00:00+00:00",
        "place_id": "place-1",
        "summary": "Met in lobby",
        "evidence_ids": ["ev-1"],
    }
    resp = client.get(
        "/graph/v0/explain/first_meeting?me_id=me&other_id=alice",
        headers={"X-Tenant-ID": "tenant-x"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["item"]["event_id"] == "event-42"
    assert stub.last_first_meeting_args == {
        "tenant_id": "tenant-x",
        "me_id": "me",
        "other_id": "alice",
    }


def test_graph_explain_event_evidence_endpoint(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.event_evidence_response = {
        "event": {"id": "ev-1"},
        "entities": [{"id": "ent-1"}],
        "places": [{"id": "place-1"}],
        "timeslices": [{"id": "ts-1"}],
        "evidences": [{"id": "e1"}],
        "utterances": [{"id": "utt-1"}],
        "utterance_speakers": [{"utterance_id": "utt-1", "entity_id": "ent-1"}],
        "knowledge": [{"id": "k1"}],
    }
    resp = client.get(
        "/graph/v0/explain/event/ev-1",
        headers={"X-Tenant-ID": "tenant-y"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["item"]["event"]["id"] == "ev-1"
    assert body["item"]["knowledge"][0]["id"] == "k1"
    assert stub.last_event_explain_args == {"tenant_id": "tenant-y", "event_id": "ev-1"}


def test_graph_explain_event_evidence_404_when_missing(monkeypatch):
    _, stub, client = _setup(monkeypatch)
    stub.event_evidence_response = {
        "event": None,
        "entities": [],
        "places": [],
        "timeslices": [],
        "evidences": [],
        "utterances": [],
        "knowledge": [],
    }
    resp = client.get(
        "/graph/v0/explain/event/ev-missing",
        headers={"X-Tenant-ID": "tenant-z"},
    )
    assert resp.status_code == 404
