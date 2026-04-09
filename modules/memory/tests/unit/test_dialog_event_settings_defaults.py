from __future__ import annotations

from modules.memory.application.config import get_dialog_event_settings


def test_dialog_event_settings_default_no_event_cap() -> None:
    settings = get_dialog_event_settings({})
    assert settings.get("max_events_per_session") == 0
