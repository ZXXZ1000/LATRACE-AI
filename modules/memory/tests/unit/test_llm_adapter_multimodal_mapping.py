from __future__ import annotations


from modules.memory.application.llm_adapter import _map_messages_for_multimodal


def test_map_messages_for_multimodal_image_parts():
    msgs = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello", "media": [{"type": "image", "data_url": "data:image/jpeg;base64,AAA"}]},
    ]
    out = _map_messages_for_multimodal(msgs)
    assert isinstance(out, list)
    user = [m for m in out if m.get("role") == "user"][0]
    content = user.get("content")
    assert isinstance(content, list)
    has_text = any(p.get("type") == "text" for p in content)
    has_img = any(p.get("type") == "image_url" and p.get("image_url", {}).get("url", "").startswith("data:image/") for p in content)
    assert has_text and has_img

