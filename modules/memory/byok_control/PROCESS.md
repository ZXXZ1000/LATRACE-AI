# PROCESS — BYOK Control Plane

本文件记录 BYOK 控制面模块的演进，遵循“实现 → 测试 → 记录”的循环。

---

## 2025-12-28 — Phase2 起步：BYOK Registry 骨架

- 动机（Why）
  - 控制面需要明确的 Provider/Credential/Binding 数据结构与最小可测试的 CRUD 入口。

- 实现（What）
  - 新增模块 `modules/memory/byok_control`：数据模型、SQLite 存储、Fernet 加密器、Registry 服务。
  - 最小 CRUD 覆盖 ProviderProfile / ProviderCredential / LlmBinding。

- 验证（Test）
  - `.venv/bin/python -m pytest modules/memory/byok_control/tests/unit/test_registry.py -q`

---

## 2025-12-29 — Phase2 扩展：更新/轮换/审计钩子

- 动机（Why）
  - 控制面必须支持 profile 更新、credential 轮换/吊销、binding 启停，并提供审计事件出口。

- 实现（What）
  - `modules/memory/byok_control/services/registry.py`
    - 新增 `update_profile`、`rotate_credential`、`set_binding_status`；
    - 引入 `AuditSink`，所有 CRUD 触发结构化审计事件。
  - `modules/memory/byok_control/tests/unit/test_registry.py`
    - 覆盖轮换、状态更新、审计事件触发。

- 验证（Test）
  - `.venv/bin/python -m pytest modules/memory/byok_control/tests/unit/test_registry.py -q`
