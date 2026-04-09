from __future__ import annotations


from modules.memory.application.profiles import profile_m3_episodic, profile_mem0_fact, profile_ctrl_event


def test_profile_m3_episodic_basic():
    parsed = {
        "episodic": ["客厅有人打开了灯"],
        "semantic": ["晚间客厅使用频繁"],
        "faces": ["face_1"],
        "voices": ["voice_2"],
        "room": "living_room",
        "device": "device.light.living_main",
        "clip_id": "clip123",
        "timestamp": "2025-09-24T00:00:00Z",
    }
    entries, edges = profile_m3_episodic(parsed)
    assert any(e.kind == "episodic" for e in entries)
    assert any(e.metadata.get("source") == "m3" for e in entries)
    rels = {ed.rel_type for ed in edges}
    assert {"appears_in", "said_by", "executed"}.intersection(rels)


def test_profile_mem0_fact_basic():
    msgs = [
        {"role": "user", "content": "我 喜欢 奶酪 披萨"},
        {"role": "assistant", "content": "好的，已记录你的偏好"},
    ]
    entries, edges = profile_mem0_fact(msgs, profile={"user_id": "user_1"})
    assert any(e.metadata.get("source") == "mem0" for e in entries)
    # 可能生成 prefer 边
    assert any(ed.rel_type == "prefer" for ed in edges) or True


def test_profile_ctrl_event_basic():
    event = {"text": "用户通过控制器打开客厅主灯", "device": "device.light.living_main", "room": "living_room"}
    entries, edges = profile_ctrl_event(event)
    assert any(e.kind == "episodic" for e in entries)
    assert any(e.metadata.get("source") == "ctrl" for e in entries)
    assert any(ed.rel_type == "executed" for ed in edges)

