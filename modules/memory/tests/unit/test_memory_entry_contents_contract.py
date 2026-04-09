#!/usr/bin/env python3
"""
测试MemoryEntry的contents字段契约化。
验证contents始终是List[str]类型，并提供安全访问方法。
"""

from __future__ import annotations

from modules.memory.contracts.memory_models import MemoryEntry


def test_contents_always_list_of_strings():
    """测试contents字段始终是List[str]类型。"""
    # 测试1: 传入None
    entry1 = MemoryEntry(kind="episodic", modality="text", contents=None)
    assert entry1.contents == []
    assert isinstance(entry1.contents, list)

    # 测试2: 传入字符串，自动转换为列表
    entry2 = MemoryEntry(kind="episodic", modality="text", contents="single string")
    assert entry2.contents == ["single string"]
    assert isinstance(entry2.contents, list)
    assert all(isinstance(c, str) for c in entry2.contents)

    # 测试3: 传入列表，自动转换所有元素为字符串，None被过滤掉
    entry3 = MemoryEntry(
        kind="semantic",
        modality="text",
        contents=["text1", 123, None, "text2"]  # 混合类型
    )
    assert entry3.contents == ["text1", "123", "text2"]  # None被过滤掉，不会包含空字符串
    assert isinstance(entry3.contents, list)
    assert all(isinstance(c, str) for c in entry3.contents)

    # 测试4: 传入空列表
    entry4 = MemoryEntry(kind="semantic", modality="image", contents=[])
    assert entry4.contents == []
    assert isinstance(entry4.contents, list)


def test_get_primary_content_safe():
    """测试安全获取主要内容的API。"""
    # 有内容的条目
    entry1 = MemoryEntry(
        kind="episodic",
        modality="text",
        contents=["first content", "second content"]
    )
    assert entry1.get_primary_content() == "first content"
    assert entry1.get_primary_content("default") == "first content"

    # 空内容的条目
    entry2 = MemoryEntry(kind="episodic", modality="text", contents=[])
    assert entry2.get_primary_content() == ""
    assert entry2.get_primary_content("default") == "default"

    # None内容的条目
    entry3 = MemoryEntry(kind="episodic", modality="text", contents=None)
    assert entry3.get_primary_content() == ""
    assert entry3.get_primary_content("default") == "default"


def test_add_content_safe():
    """测试安全添加内容的API。"""
    entry = MemoryEntry(kind="semantic", modality="text", contents=["original"])

    # 添加字符串
    entry.add_content("new content")
    assert entry.contents == ["original", "new content"]

    # 重复添加（应该去重）
    entry.add_content("new content")
    assert entry.contents == ["original", "new content"]

    # 添加列表
    entry.add_content(["content1", "content2", "original"])  # 包含重复
    assert entry.contents == ["original", "new content", "content1", "content2"]

    # 添加None（应该被忽略）
    entry.add_content(None)
    assert entry.contents == ["original", "new content", "content1", "content2"]


def test_friendly_repr():
    """测试友好的字符串表示。"""
    entry = MemoryEntry(
        id="test-1",
        kind="episodic",
        modality="text",
        contents=["This is a long text content that should be truncated in the repr"]
    )
    repr_str = repr(entry)

    # 验证repr包含关键信息
    assert "test-1" in repr_str
    assert "episodic" in repr_str
    assert "text" in repr_str
    assert "contents_len=1" in repr_str
    # 检查内容被截断（实际实现会截断到50字符）
    assert "This is a long text content that should be truncat..." in repr_str


def test_backwards_compatibility():
    """测试向后兼容性：直接访问contents[0]仍然有效。"""
    entry = MemoryEntry(
        kind="semantic",
        modality="text",
        contents=["primary text", "secondary text"]
    )

    # 直接访问contents[0]（向后兼容）
    assert entry.contents[0] == "primary text"

    # 但推荐使用新的安全API
    assert entry.get_primary_content() == "primary text"

    # 空contents访问应使用安全API
    empty_entry = MemoryEntry(kind="semantic", modality="image", contents=[])
    assert empty_entry.get_primary_content() == ""  # 安全

    # 直接访问会抛出IndexError，但这是预期的行为
    try:
        _ = empty_entry.contents[0]
        assert False, "Should raise IndexError"
    except IndexError:
        pass  # 预期的行为


if __name__ == "__main__":
    test_contents_always_list_of_strings()
    test_get_primary_content_safe()
    test_add_content_safe()
    test_friendly_repr()
    test_backwards_compatibility()
    print("✅ All contents contract tests passed!")
