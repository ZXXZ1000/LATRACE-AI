# ADK Layer 1 TODO（系统工程总表）

目标：构建 `Layer 1: ADK Tools` 的共享基础件与语义工具实现层，为后续 `Layer 2 Semantic Router API` 提供稳定、可测、可观测的工具运行时。

## 阶段 A：基础件（当前）

- [x] A1 建立模块骨架与共享契约（`ToolResult` / `ToolDebugTrace` / error mapping）
- [x] A2 实现 `_resolve_if_needed`（实体解析共享组件）
- [x] A3 实现 `state_property_vocab`（状态属性词表缓存/映射共享组件）
- [x] A4 组合级最小验证（resolve + property vocab 前置链路）

## 阶段 B：状态类工具（风险优先）

- [x] B1 `entity_status`
- [x] B2 `status_changes`
- [x] B3 `state_time_since`

## 阶段 C：核心记忆类工具（高频价值）

- [x] C1 `entity_profile`
- [x] C2 `time_since`（memory/v1）
- [x] C3 `relations`
- [x] C4 `quotes`
- [x] C5 `topic_timeline`
- [x] C6 `explain`（原子工具）

## 阶段 D：发现类工具（低风险）

- [x] D1 `list_entities`
- [x] D2 `list_topics`

## 阶段 E：Layer 2 语义路由 API（单工具版 v1）

- [x] E1 `/memory/agentic/query` 路由协议与实现
- [x] E2 单工具路由 + 结构化返回（不生成自然语言答案）
- [ ] E3 路由准确率/延迟验收

## 阶段 F：开箱封装与工具定义单一信源

- [x] F1 `HttpMemoryInfraAdapter`（12 个端点映射 + 错误载荷统一）
- [x] F2 `MemoryAdkRuntime` + `create_memory_runtime`（同步工厂 + user_tokens 覆盖）
- [x] F3 `tool_definitions.py`（单一信源 + OpenAI/MCP 导出）
- [x] F4 ADK 文档与根目录文档对齐更新

## 周期执行规则（固定）

每个任务周期必须完成：

1. 代码实现
2. 测试验证（新增/更新测试）
3. 文档记录（`modules/memory/adk/PROCESS.md`；若对外接口/SDK变化，同步根目录文档）
