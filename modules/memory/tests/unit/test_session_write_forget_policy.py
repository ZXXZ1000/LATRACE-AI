from modules.memory.session_write import _merge_forget_policy


def test_merge_forget_policy_ignores_none():
    assert _merge_forget_policy([None, None]) is None


def test_merge_forget_policy_prefers_until_changed():
    assert _merge_forget_policy([None, "until_changed", "temporary"]) == "until_changed"


def test_merge_forget_policy_temporary_only_when_present():
    assert _merge_forget_policy([None, "temporary", ""]) == "temporary"
