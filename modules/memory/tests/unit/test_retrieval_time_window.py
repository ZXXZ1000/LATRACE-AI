from modules.memory.retrieval import _parse_time_window


def test_parse_time_window_date_literal_dash():
    start, end, reason = _parse_time_window("need events on 2024-12-31")
    assert reason == "date_literal"
    assert start.startswith("2024-12-31")
    assert end.startswith("2024-12-31")


def test_parse_time_window_date_literal_slash():
    start, end, reason = _parse_time_window("check 2024/01/15 activity")
    assert reason == "date_literal"
    assert start.startswith("2024-01-15")
    assert end.startswith("2024-01-15")
