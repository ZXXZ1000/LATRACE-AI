#!/usr/bin/env python3
"""
测试搜索权重验证功能。
"""

from __future__ import annotations

from modules.memory.application.runtime_config import set_rerank_weights, validate_rerank_weights, clear_rerank_weights_override


def test_valid_weights():
    """测试有效权重设置。"""
    clear_rerank_weights_override()

    # 有效权重：核心权重和=1.0
    weights = {
        "alpha_vector": 0.4,
        "beta_bm25": 0.3,
        "gamma_graph": 0.2,
        "delta_recency": 0.1
    }

    # 应该成功设置
    set_rerank_weights(weights)

    # 验证设置成功
    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()
    assert saved["alpha_vector"] == 0.4
    assert saved["beta_bm25"] == 0.3
    assert saved["gamma_graph"] == 0.2
    assert saved["delta_recency"] == 0.1


def test_weights_sum_not_one():
    """测试权重和不为1.0时抛出异常。"""
    clear_rerank_weights_override()

    # 无效权重：核心权重和=0.5 != 1.0
    weights = {
        "alpha_vector": 0.2,
        "beta_bm25": 0.3,
        "gamma_graph": 0.0,
        "delta_recency": 0.0
    }

    try:
        set_rerank_weights(weights)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "必须为1.0" in str(e)
        assert "0.500" in str(e)


def test_negative_weight():
    """测试负数权重被静默忽略（不抛出异常）。"""
    clear_rerank_weights_override()

    # 负数权重会被静默忽略
    weights = {
        "alpha_vector": 0.4,
        "beta_bm25": 0.3,
        "gamma_graph": 0.2,
        "delta_recency": 0.1,
        "user_boost": -0.1  # 负数权重会被忽略
    }

    # 不应该抛出异常，而是静默忽略无效权重
    set_rerank_weights(weights)

    # 验证只有有效权重被设置
    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()
    assert saved["alpha_vector"] == 0.4
    assert saved["beta_bm25"] == 0.3
    assert saved["gamma_graph"] == 0.2
    assert saved["delta_recency"] == 0.1
    assert "user_boost" not in saved  # 负数权重被忽略


def test_empty_weights():
    """测试空权重字典。"""
    clear_rerank_weights_override()

    # 空权重应该成功（什么都不设置）
    set_rerank_weights({})
    set_rerank_weights(None)

    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()
    # 空权重不应该改变现有设置
    assert len(saved) == 0


def test_partial_weights():
    """测试部分权重设置。"""
    clear_rerank_weights_override()

    # 只设置部分权重
    weights = {
        "alpha_vector": 0.6,
        "beta_bm25": 0.4,
        # gamma_graph和delta_recency未设置
    }

    set_rerank_weights(weights)

    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()
    assert saved["alpha_vector"] == 0.6
    assert saved["beta_bm25"] == 0.4
    # 其他权重不应该被设置
    assert "gamma_graph" not in saved
    assert "delta_recency" not in saved


def test_boost_weights():
    """测试boost权重（不影响核心和）。"""
    clear_rerank_weights_override()

    # 设置核心权重和boost权重
    weights = {
        "alpha_vector": 0.5,
        "beta_bm25": 0.5,
        "user_boost": 0.1,  # boost权重，可以大于1
        "domain_boost": 0.2
    }

    set_rerank_weights(weights)

    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()
    assert saved["alpha_vector"] == 0.5
    assert saved["beta_bm25"] == 0.5
    assert saved["user_boost"] == 0.1
    assert saved["domain_boost"] == 0.2


def test_validate_only():
    """测试validate_rerank_weights函数（不设置）。"""
    clear_rerank_weights_override()

    # 有效权重
    weights = {
        "alpha_vector": 0.4,
        "beta_bm25": 0.3,
        "gamma_graph": 0.2,
        "delta_recency": 0.1
    }

    # 验证应该成功
    validated = validate_rerank_weights(weights)
    assert validated == weights

    # 验证后应该没有设置权重（因为只是验证）
    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()
    assert len(saved) == 0


def test_floating_point_precision():
    """测试浮点数精度处理。"""
    clear_rerank_weights_override()

    # 由于浮点数精度问题，可能总和略小于或大于1.0
    # 但在允许误差范围内（0.001）
    weights = {
        "alpha_vector": 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1,  # 可能不是精确的1.0
        "beta_bm25": 0.0,
        "gamma_graph": 0.0,
        "delta_recency": 0.0
    }

    # 应该成功（允许小误差）
    set_rerank_weights(weights)


def test_invalid_key_ignored():
    """测试无效的权重键被忽略。"""
    clear_rerank_weights_override()

    # 包含无效键
    weights = {
        "alpha_vector": 0.5,
        "beta_bm25": 0.5,
        "invalid_key": 1.0,  # 无效键，应该被忽略
        "another_invalid": -1.0
    }

    set_rerank_weights(weights)

    from modules.memory.application.runtime_config import get_rerank_weights_override
    saved = get_rerank_weights_override()

    # 只有有效键被设置
    assert saved["alpha_vector"] == 0.5
    assert saved["beta_bm25"] == 0.5
    assert "invalid_key" not in saved
    assert "another_invalid" not in saved


if __name__ == "__main__":
    test_valid_weights()
    test_weights_sum_not_one()
    test_negative_weight()
    test_empty_weights()
    test_partial_weights()
    test_boost_weights()
    test_validate_only()
    test_floating_point_precision()
    test_invalid_key_ignored()
    print("✅ All rerank weights validation tests passed!")
