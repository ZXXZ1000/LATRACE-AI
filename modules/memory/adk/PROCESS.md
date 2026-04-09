# ADK PROCESS

> 记录 `modules/memory/adk` 模块的实现周期、测试结果与目标符合性评估。  
> 执行循环：代码实现 -> 测试验证 -> 文档记录。

## 001. 阶段 A1：共享骨架初始化（已完成）

- 目标：
  - 建立 `Layer 1 ADK` 模块基础结构。
  - 落地共享契约：`ToolResult`（LLM 可见层 4 字段）与 `ToolDebugTrace`（调试层）。
  - 落地错误归一化 helper（HTTP / Exception -> ADK 统一错误语义）。

- 实现内容：
  - 新增 `modules/memory/adk/__init__.py`
  - 新增 `modules/memory/adk/models.py`
    - `ToolResult`
    - `ToolDebugTrace`
  - 新增 `modules/memory/adk/errors.py`
    - `AdkErrorInfo`
    - `normalize_http_error()`
    - `normalize_exception()`
  - 新增 `modules/memory/adk/TODO.md`（顶层任务总表）

- 测试验证（本周期）：
  - `test_adk_models.py`
  - `test_adk_errors.py`
  - `pytest modules/memory/tests/unit/test_adk_models.py modules/memory/tests/unit/test_adk_errors.py -q`
    - 结果：`7 passed`
  - `pytest modules/memory/tests -q`
    - 结果：`502 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK 内部基础件）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估：
  - 为 A2/A3 提供统一返回结构、调试结构、错误分类基础
  - 不引入 Infra API 行为变化
  - 结论：符合。A1 作为纯 ADK 内部基础件落地后，新增单测与 memory 模块全回归均通过。

## 002. 阶段 A2：`_resolve_if_needed` 实体解析共享组件（已完成）

- 目标：
  - 实现实体型工具共享的 `_resolve_if_needed()` 辅助步骤（默认不作为面向 Agent 的终态工具）。
  - 严格执行 ADK 歧义策略：`candidates` 非空即停止并返回 `needs_disambiguation=true`。
  - 为后续 `entity_status / status_changes / entity_profile / quotes / relations / time_since` 提供统一实体解析入口。

- 实现内容：
  - 新增 `modules/memory/adk/resolve.py`
    - `ResolveIfNeededOutcome`
    - `_resolve_if_needed(...)`
  - 设计要点：
    - `entity_id` 直通（跳过 resolver）
    - 缺少 `entity/entity_id` 前置拦截（返回 `invalid_input`）
    - `resolve-entity` 候选非空时终止，不继续后续查询
    - `resolve_limit` 在 ADK 层 clamp 到 `1..5`
    - 通过 `ToolResult + ToolDebugTrace` 返回统一 LLM 层/调试层结构
    - 异常通过 `normalize_exception()` 归一化（含 retryable 标记）
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `_resolve_if_needed`
    - `ResolveIfNeededOutcome`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_resolve.py`
  - 覆盖场景：
    - `entity_id` 直通跳过 resolver
    - 缺少实体参数（invalid_input）
    - resolve 成功
    - resolve 歧义（`candidates` -> `needs_disambiguation`）
    - resolve 未命中（`matched=false`）
    - resolver 异常（timeout -> retryable）
    - `resolve_limit` clamp 到 5
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_resolve.py -q` -> `7 passed`
    - `pytest modules/memory/tests -q` -> `509 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK 内部共享组件）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 A2 目标: 满足。`_resolve_if_needed` 已形成可复用、可测试、可观测的共享入口。
  - 一致性目标: 满足。与 ADK 主文档关于 `resolve-entity` 的定位、歧义策略和错误归一化约束一致。

## 003. 阶段 A3：`state_property_vocab` 状态属性词表缓存/映射组件（已完成）

- 目标：
  - 落地状态类工具共享组件：词表加载缓存（`/memory/v1/state/properties`）+ 属性映射（canonical/alias/normalized）。
  - 将属性映射歧义显式建模为 `needs_disambiguation`，避免状态类工具静默使用错误 property。
  - 为 `entity_status / status_changes / state_time_since` 提供统一属性映射入口。

- 实现内容：
  - 新增 `modules/memory/adk/state_property_vocab.py`
    - `StatePropertyDef`
    - `StatePropertyVocab`
    - `PropertyResolutionResult`
    - `StatePropertyVocabManager`
    - `StatePropertyVocabLoadError`
    - `map_state_property(...)`
  - 组件能力：
    - 进程内缓存（按 `tenant_id`）
    - `force_refresh`
    - `vocab_version` 变化标记（`vocab_refreshed`）
    - ADK 本地 alias map（可后续外置）
    - canonical 精确 / alias 精确 / normalized 匹配
    - 规范化歧义候选返回（`needs_disambiguation`）
    - 加载失败统一包装为 `StatePropertyVocabLoadError`（含 `AdkErrorInfo`）
  - 更新 `modules/memory/adk/__init__.py` 导出相关类型与 helper

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_state_property_vocab.py`
  - 覆盖场景：
    - 词表加载成功 + cache hit
    - `force_refresh` + `vocab_version` 变化
    - canonical/alias/normalized 匹配
    - 规范化歧义候选
    - 加载异常（timeout）
    - in-band HTTP 错误响应映射（`503 temporarily unavailable`）
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_state_property_vocab.py -q` -> `6 passed`
    - `pytest modules/memory/tests -q` -> `515 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK 内部共享组件）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 A3 目标: 满足。状态属性词表加载与映射能力已可复用，可直接支撑状态类三件套实现。
  - 一致性目标: 满足。与 ADK 主文档 `state property vocab` 规范（缓存/版本/映射/歧义/观测）保持一致。

## 004. 阶段 A4：组合级最小验证（`resolve + property vocab` 前置链路）（已完成）

- 目标：
  - 将 A2（实体解析）与 A3（状态属性词表映射）组合为状态类工具可直接复用的前置链路。
  - 在进入 `entity_status / status_changes / state_time_since` 前，先把最易出错的组合分支（缺参、词表失败、属性歧义）统一收敛为可测试行为。

- 实现内容：
  - 新增 `modules/memory/adk/state_preflight.py`
    - `StateQueryPreflightOutcome`
    - `prepare_state_query_preflight(...)`
  - 关键行为：
    - 统一前置链路：`_resolve_if_needed -> state_property_vocab -> map_state_property`
    - 支持 `entity_id` 与 `property_canonical` 高级直通（跳过 resolver / 词表映射）
    - 缺少状态属性时 ADK 前置拦截（`invalid_input`）
    - 词表加载失败统一映射为终止 `ToolResult`（复用 A1/A3 错误归一化）
    - 属性歧义返回 `needs_disambiguation=true` + `property_candidates`
    - 成功路径返回 `entity_id + property_canonical + resolution_meta`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `StateQueryPreflightOutcome`
    - `prepare_state_query_preflight`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_state_preflight.py`
  - 覆盖场景：
    - `entity_id + property_canonical` 直通（跳过 resolver 与 vocab）
    - 缺少状态属性（invalid_input）
    - 词表加载失败（timeout -> retryable）
    - 属性歧义（`property_candidates` + `needs_disambiguation`）
    - 成功路径（entity resolve + alias property 映射）
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_state_preflight.py -q` -> `5 passed`
    - `pytest modules/memory/tests -q` -> `520 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK 内部共享前置组件）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 A4 目标: 满足。状态类工具前置链路已可直接复用，Stage A 基础件完成。
  - 风险控制: 满足。属性缺参/词表故障/属性歧义在进入具体工具前已统一收敛，降低后续工具实现分支复杂度。

## 005. 阶段 B1：`entity_status` 工具实现（已完成）

- 目标：
  - 落地状态类三件套中的第一个工具 `entity_status`，统一封装 `/memory/state/current` 与 `/memory/state/at_time`。
  - 验证 Stage A 基础件（`prepare_state_query_preflight`）在真实工具中的复用性与边界行为一致性。

- 实现内容：
  - 新增 `modules/memory/adk/state_tools.py`
    - `entity_status(...)`
    - 默认严格 ISO 时间解析（可注入 `when_parser` 扩展自然语言时间解析）
  - 关键设计：
    - `when=None` -> `/memory/state/current`
    - `when!=None` -> 先解析 `t_iso`，再调用 `/memory/state/at_time`
    - 前置链路统一复用 A4：实体解析 + 属性映射
    - `404 state_not_found` 映射为 `matched=false`
    - 错误通过 A1 helper 归一化，并保持 LLM 可见层 4 字段不扩张
    - 调试层记录 `api_route`、`resolution_meta`、`raw_api_response_keys`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `entity_status`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_state_tools_entity_status.py`
  - 覆盖场景：
    - current 路径成功（`when=None`）
    - at_time 路径成功（严格 ISO 时间）
    - `when` 无法解析的前置拦截（不调用状态端点）
    - 实体歧义终止（不进入词表/状态端点）
    - 属性歧义终止（不进入状态端点）
    - `404 state_not_found` 映射为 `matched=false`
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_state_tools_entity_status.py -q` -> `6 passed`
    - `pytest modules/memory/tests -q` -> `526 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 B1 目标: 满足。`entity_status` 已按详细规范落地，验证了 Stage A 基础件在状态类工具中的复用路径。
  - 风险控制: 满足。实体/属性歧义与时间解析失败均在进入 Infra 状态查询前被拦截或显式建模。

## 006. 阶段 B2：`status_changes` 工具实现（已完成）

- 目标：
  - 落地状态类第二个工具 `status_changes`，封装 `/memory/state/what-changed`，验证“状态变化列表 + 时间范围归一化 + order/limit 防御性归一化”的模式。
  - 继续复用 Stage A 基础件，保证状态类工具族行为一致。

- 实现内容：
  - 更新 `modules/memory/adk/state_tools.py`
    - 新增 `status_changes(...)`
    - 新增共享 helper（供 B2/B3 复用）：
      - `_parse_time_range(...)`
      - `_normalize_order(...)`
      - `_clamp_limit(...)`
  - 关键行为：
    - 默认调用 `/memory/state/what-changed`（语义路由名更清晰）
    - `order` 归一化为 `asc/desc`（非法值回退 `desc`）
    - `limit` 在 ADK 层 clamp 到 `1..50`
    - `when/time_range` 前置归一化失败即终止（防止静默放大查询范围）
    - `items=[]` 映射为 `matched=false`（非错误），但保留 `data.items=[]` 结构
    - 歧义场景（实体/属性）严格终止，不继续调用底层状态变化接口
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `status_changes`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_state_tools_status_changes.py`
  - 覆盖场景：
    - 默认路由 + `order/limit` 归一化与 clamp
    - `items=[]` -> `matched=false`（保留 `data.items=[]`）
    - 时间范围解析失败 ADK 前置拦截
    - 实体歧义终止（不调用 vocab / endpoint）
    - 属性歧义终止（不调用 endpoint）
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_state_tools_status_changes.py -q` -> `5 passed`
    - `pytest modules/memory/tests -q` -> `531 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 B2 目标: 满足。`status_changes` 已按详细规范落地，且验证了状态类“时间范围前置拦截”防线可执行。
  - 风险控制: 满足。通过 ADK 前置校验避免了底层时间范围静默容错导致的查询范围放大风险。

## 007. 阶段 B3：`state_time_since` 工具实现（已完成）

- 目标：
  - 落地状态类第三个工具 `state_time_since`，封装 `/memory/state/time-since`，完成状态类三件套实现闭环。
  - 复用 B2 新增的时间范围归一化 helper，验证状态类工具共享抽象的稳定性。

- 实现内容：
  - 更新 `modules/memory/adk/state_tools.py`
    - 新增 `state_time_since(...)`
    - 复用 `prepare_state_query_preflight(...)`、`_parse_time_range(...)`
  - 关键行为：
    - 成功时透传 `seconds_ago`（不改字段名）
    - `404 state_not_found` 映射为 `matched=false` + “未找到状态变化记录”
    - 时间范围解析失败 ADK 前置拦截
    - 实体/属性歧义严格终止，不继续调用底层端点
    - `last_changed_at` 缺失时保持 `matched=true`，并给出提示文案，同时在调试层写入 `state_time_missing`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `state_time_since`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_state_tools_state_time_since.py`
  - 覆盖场景：
    - 成功返回并透传 `seconds_ago`
    - `404 state_not_found` 映射为 `matched=false`
    - 时间范围解析失败前置拦截
    - 实体歧义终止（不调用底层）
    - `last_changed_at` 缺失仍 `matched=true` 并提示
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_state_tools_state_time_since.py -q` -> `5 passed`
    - `pytest modules/memory/tests -q` -> `536 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 B3 目标: 满足。状态类三件套已全部落地，形成共享基础件 + 工具实现的完整验证链。
  - 一致性目标: 满足。`state_time_since` 与文档规范在字段透传、404 映射、缺失时间信息提示方面保持一致。

## 008. 阶段 C1：`entity_profile` 工具实现（已完成）

- 目标：
  - 落地核心记忆类首个工具 `entity_profile`，验证 ADK 通用骨架（`_resolve_if_needed` / `ToolResult` / 错误归一化）在 memory/v1 查询工具中的复用质量。
  - 按详细规范实现 `include + limit` 的语义化参数收敛，避免向 Agent 暴露多组底层布尔开关与多重 limit。

- 实现内容：
  - 新增 `modules/memory/adk/memory_tools.py`
    - `entity_profile(...)`
  - 关键设计：
    - `include: list[str]`（默认 `["facts","relations","events"]`），统一收敛到底层 `include_*` 开关
    - `limit` 统一控制 `facts/relations/events/quotes` 的上限（ADK clamp `1..50`）
    - 缺省 `user_tokens` 时按 `tenant_id` 自动派生 `u:{tenant_id}`
    - 复用 `_resolve_if_needed(...)` 实现实体歧义严格终止（不继续查画像）
    - `503 temporarily unavailable` 映射为 `rate_limit + retryable=true`
    - LLM 可见层仅返回业务数据；调试层记录 `resolution_meta`、`raw_api_response_keys`、`api_route`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `entity_profile`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_memory_tools_entity_profile.py`
  - 覆盖场景：
    - 默认 `include/limit` 映射到底层参数（含 clamp）
    - `entity_id` 直通跳过 resolve
    - 实体歧义终止（不调用底层画像端点）
    - 底层 `found=false` 映射为 `matched=false`
    - `503 temporarily unavailable` 映射为 `rate_limit + retryable`
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_memory_tools_entity_profile.py -q` -> `5 passed`
    - ADK 增量回归（相关工具/状态类工具）-> `21 passed`
    - `pytest modules/memory/tests -q` -> `541 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 C1 目标: 满足。`entity_profile` 已达到可复用、可测试、可观测的实现粒度。
  - 一致性目标: 满足。与 ADK 详细规范的 `include + limit` 设计、歧义终止策略、错误映射约束一致。

## 009. 阶段 C2：`time_since`（memory/v1）工具实现（已完成）

- 目标：
  - 落地核心记忆类第二个工具 `time_since`，封装 `/memory/v1/time-since`。
  - 验证记忆类工具在实体解析、时间范围前置校验、AND 语义提示、错误归一化方面与状态类工具的一致性。

- 实现内容：
  - 更新 `modules/memory/adk/memory_tools.py`
    - 新增 `time_since(...)`
    - 增加时间范围归一化 helper（ISO 校验）
  - 关键行为：
    - 支持 `entity/topic` 语义入参与 `entity_id/topic_id/topic_path` 高级直通
    - 缺少查询条件前置拦截（`invalid_input`）
    - 复用 `_resolve_if_needed(...)`，实体歧义严格终止
    - `time_range` 进行 ADK 前置 ISO 校验
    - `last_mentioned=null` 映射为 `matched=false`
    - 同时传 `entity + topic` 时在 LLM 可见层补 AND 语义说明消息
    - 调试层记录 `filter_semantics`（`AND/entity_only/topic_only`）
    - `504` 映射为 `timeout + retryable=true`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `time_since`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_memory_tools_time_since.py`
  - 覆盖场景：
    - entity-only 成功返回
    - entity+topic AND 语义提示
    - 缺少查询条件前置拦截
    - 实体歧义终止（不调用底层）
    - `last_mentioned=null` -> `matched=false`
    - `504 timeout` 映射为 retryable
    - 无效 `time_range` 前置拦截
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_memory_tools_time_since.py -q` -> `7 passed`
    - `pytest modules/memory/tests -q` -> `548 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 C2 目标: 满足。`time_since` 已按详细规范落地，并完成 AND 语义提醒与错误映射。
  - 风险控制: 满足。对缺少条件、实体歧义、无效时间范围均在 ADK 层显式处理，避免底层行为差异泄漏给 Agent。

## 010. 阶段 C3：`relations` 工具实现（已完成）

- 目标：
  - 落地核心记忆类第三个工具 `relations`，封装 `/memory/v1/relations`。
  - 将底层 `found`（实体已解析）与 ADK `matched`（是否有关系结果）显式拆分，避免 Agent 将“无关系”误解为“实体不存在”。

- 实现内容：
  - 更新 `modules/memory/adk/memory_tools.py`
    - 新增 `relations(...)`
    - 新增 `relation_type` 归一化 helper（仅允许 `co_occurs_with`）
  - 关键行为：
    - `relation_type` 非支持值 ADK 前置拦截（不依赖底层“空结果成功”）
    - 复用 `_resolve_if_needed(...)`，实体歧义严格终止
    - `time_range` 前置 ISO 校验
    - 底层 `found=true + relations=[]` -> ADK `matched=false`，但调试层 `entity_resolved=true`
    - `504 relations_timeout` 映射为 `timeout + retryable`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `relations`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_memory_tools_relations.py`
  - 覆盖场景：
    - 成功返回关系
    - `found=true + relations=[]` 语义拆分映射
    - 非支持 `relation_type` 前置拦截
    - 实体歧义终止
    - `relations_timeout` 与 `time_range` 非法输入映射
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_memory_tools_relations.py -q` -> `5 passed`
    - `pytest modules/memory/tests -q` -> `553 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 C3 目标: 满足。`relations` 已按详细规范落地，并落实了 `found`/`matched` 语义拆分的关键设计点。
  - 一致性目标: 满足。与 ADK 详细规范在 `relation_type` 前置拦截、超时映射、无关系非实体缺失等约束一致。

## 011. 阶段 C4：`quotes` 工具实现（已完成）

- 目标：
  - 落地核心记忆类第四个工具 `quotes`，封装 `/memory/v1/quotes`。
  - 按 ADK 详细规范实现三条路径（`entity-only / topic-only / entity+topic`）、实体歧义严格终止、时间范围前置校验与高成本错误映射。

- 实现内容：
  - 更新 `modules/memory/adk/memory_tools.py`
    - 新增 `quotes(...)`
    - 新增 `_quotes_source_mode(...)`（调试层来源模式识别）
  - 关键行为：
    - 至少要求实体或话题参数；缺失时 ADK 前置拦截（`invalid_input`）
    - 复用 `_resolve_if_needed(...)`，实体歧义严格终止（不沿用底层“候选+继续查询”的混合行为）
    - `time_range` 前置 ISO 校验，避免底层静默扩大查询范围
    - `limit` 在 ADK 层 clamp `1..10`（默认 `5`）
    - `503 temporarily unavailable` 映射为 `rate_limit + retryable`
    - `504 quotes_timeout` 映射为 `timeout + retryable`
    - `quotes=[]` 映射为 `matched=false`（非错误）
    - 调试层记录 `source_mode / fallback_used / api_route`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `quotes`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_memory_tools_quotes.py`
  - 覆盖场景：
    - `entity-only` 成功（默认 `user_tokens`、`limit` clamp、生效 `graph_filter` 推断）
    - `topic-only` 路径（带 retrieval trace，标记 `retrieval_rag + fallback_used=true`）
    - `entity+topic` 组合路径成功
    - 实体歧义终止（不调用底层）
    - 缺少参数与无效 `time_range` 前置拦截
    - `503` 与 `504 quotes_timeout` 映射为不同错误类型
    - 空 quotes 映射为 `matched=false`
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_memory_tools_quotes.py -q` -> `7 passed`
    - `pytest modules/memory/tests -q` -> `560 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 C4 目标: 满足。`quotes` 已按详细规范落地，完成歧义严格终止与错误归一化要求。
  - 阶段推进条件: 满足。可继续进入 C5 `topic_timeline`。

## 012. 阶段 C5：`topic_timeline` 工具实现（已完成）

- 目标：
  - 落地核心记忆类第五个工具 `topic_timeline`，封装 `/memory/v1/topic-timeline`。
  - 按 ADK 详细规范实现 `include` 列表封装、时间范围前置校验、空时间线语义映射与高成本错误映射。

- 实现内容：
  - 更新 `modules/memory/adk/memory_tools.py`
    - 新增 `topic_timeline(...)`
    - 新增 `_normalize_timeline_include(...)`
    - 新增 `_topic_timeline_source_mode(...)`
  - 关键行为：
    - 至少要求 `topic/topic_id/topic_path/keywords` 之一；缺失时前置拦截（`invalid_input`）
    - `include: list[str]` 映射到底层 `with_quotes/with_entities`
    - `limit` 在 ADK 层 clamp `1..20`（默认 `10`）
    - `time_range` 前置 ISO 校验，避免静默扩大查询范围
    - 不重复实现 retrieval，仅调用 Infra 的 graph-first + fallback 能力
    - `timeline=[]` 映射为 `matched=false`（非错误）
    - `503 temporarily unavailable` -> `rate_limit + retryable`
    - `504 timeline_timeout` -> `timeout + retryable`
    - 调试层记录 `source_mode / fallback_used / heavy_expand`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `topic_timeline`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_memory_tools_topic_timeline.py`
  - 覆盖场景：
    - 默认 `include=[]` 映射与 `limit` clamp
    - `include=["quotes"]` 的 `heavy_expand=true` 与 `session_id` 透传
    - 缺少输入与无效 `time_range` 前置拦截
    - 空时间线 -> `matched=false`
    - `503` 与 `504 timeline_timeout` 错误映射
    - retrieval trace -> `source_mode="retrieval_rag"` + `fallback_used=true`
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_memory_tools_topic_timeline.py -q` -> `6 passed`
    - `pytest modules/memory/tests -q` -> `566 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 C5 目标: 满足。`topic_timeline` 已按详细规范落地并完成关键边界行为映射。
  - 阶段收口评估: 满足。阶段 C（核心记忆类工具）已全部完成，可进入阶段 D（发现类工具）或整理 ADK 开发者说明文档。

## 013. 阶段 C6：`explain`（原子工具）实现（已完成）

- 目标：
  - 按规范 §6.4 补齐 Layer 1 `explain(event_id)` 原子工具，封装 `/memory/v1/explain`。
  - 保持“只接受 `event_id`、不内置事件搜索”的原子性边界，为 `quotes/topic_timeline/entity_profile -> explain` 的组合调用提供稳定工具。

- 实现内容：
  - 更新 `modules/memory/adk/memory_tools.py`
    - 新增 `explain(...)`
    - 新增 `ExplainFn` 类型别名
  - 关键行为：
    - `event_id` 为空时 ADK 前置拦截（`invalid_input`）
    - 缺省 `user_tokens` 时按 `tenant_id` 自动派生 `u:{tenant_id}`（与 Infra explain 兼容策略一致）
    - 支持透传 `memory_domain`（高级作用域过滤）
    - `found=false`（含空结构）映射为 `matched=false`，但保留结构化 `data`
    - `503 temporarily unavailable` -> `rate_limit + retryable`
    - 调试层固定 `source_mode="graph_filter"`，记录 `api_route`
  - 更新 `modules/memory/adk/__init__.py` 导出：
    - `explain`

- 测试验证:
  - 新增单测：
    - `modules/memory/tests/unit/test_adk_memory_tools_explain.py`
  - 覆盖场景：
    - 成功返回证据链 bundle
    - `found=false` 映射为 `matched=false`
    - 空白 `event_id` 前置拦截
    - `user_tokens/memory_domain` 作用域透传
    - `503`/`500` 错误映射
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_memory_tools_explain.py -q` -> `5 passed`
    - `pytest modules/memory/tests -q` -> `571 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期仅 ADK Layer 1 工具实现）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 阶段 C6 目标: 满足。`explain` 已按原子性约束落地，且与 Infra explain 的作用域过滤行为对齐。
  - 阶段收口评估: 满足。ADK Layer 1 的 Wave 1 + Wave 2 高优先工具现已全部落地（含 explain）。

## 014. 根目录 `ADK开发者说明文档.md` 建立（已完成）

- 目标：
  - 在项目根目录新增一份面向开发者/内部技术人员/产品经理的 ADK 事实手册，作为 Layer 1 实现现状与后续 Layer 2 对接的统一参考。
  - 先基于“规范文档 vs 当前实现”的对齐结果成文，避免文档内容超前或失真。

- 实现内容：
  - 新增根目录文档：
    - `ADK开发者说明文档.md`
  - 内容覆盖：
    - 三层架构边界（Layer 0 / Layer 1 / Layer 2）
    - ADK 统一返回结构（LLM 可见层 + 调试层）
    - 错误归一化语义
    - 共享基础件说明（resolve / property vocab / preflight）
    - Layer 1 工具实现状态总览（已实现 / 未实现）
    - 已实现工具详细说明（状态类 + 记忆类 + explain）
    - 调试与观测约定（`source_mode` / `fallback_used`）
    - 当前测试状态与后续计划

- 测试验证:
  - 文档周期（doc-only），无代码行为变更
  - 不新增测试；沿用上一周期全测结果作为实现事实基线：
    - `pytest modules/memory/tests -q` -> `571 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（本周期新增 ADK 开发者文档，不涉及对外接口契约变更）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 文档目标: 满足。根目录 ADK 说明文档已建立，且明确区分了 Infra 已实现 vs ADK 已实现 vs Layer 2 待实现状态（含 explain 分层状态说明）。

## 015. 根目录 ADK 文档可用性增强（接入示例 + 工具列表定义）（已完成）

- 目标：
  - 修复根目录 `ADK开发者说明文档.md` 偏“能力盘点”、缺少“怎么用”的问题。
  - 让 Agent/Router 开发者看完后可以直接：
    - 绑定 Runtime
    - 定义 Tool Registry
    - 配置默认工具组与动态注入规则

- 实现内容：
  - 更新 `ADK开发者说明文档.md`
  - 新增章节：
    - `12. 如何接入 ADK（最小可用示例）`
    - `13. Agent 工具列表怎么定义（可直接照做）`
    - `14. 给 Layer 2（Semantic Router）的对接约定`
  - 新增内容包括：
    - `MemoryInfraAdapter` + `MemoryAdkRuntime` 最小示例
    - `ToolResult.to_llm_dict()` / `to_debug_dict()` 用法
    - 默认工具组 / 按需注入工具组
    - Tool Registry 建议结构（`name/description/input_schema/executor/enabled`）
    - 已实现工具 JSON Schema 示例（`entity_profile/time_since/quotes/topic_timeline/relations/explain`）
    - Layer 2 对接边界与禁止事项

- 测试验证:
  - 文档周期（doc-only），无代码行为变更
  - 不新增测试；沿用当前实现基线：
    - `pytest modules/memory/tests -q` -> `571 passed, 6 skipped`

- 外部接口/SDK 变更：
  - 无（文档增强，不涉及对外 HTTP API / SDK 契约变更）
  - 根目录 `SDK使用说明.md`、`开发者API 说明文档.md` 本周期无需更新

- 目标符合性评估:
  - 可用性目标: 满足。文档已从“知道有哪些工具”提升到“可以实际定义 Agent 工具列表并接入运行时”的粒度。

## 016. ADK 开箱封装（T1/T2/T3/T5）落地：InfraAdapter + Runtime + ToolDefinitions（已完成）

- 目标：
  - 按修订任务清单完成 ADK 开箱能力，避免 Agent 开发者手写依赖注入层。
  - 建立工具定义单一信源，支持 OpenAI/MCP 工具 schema 代码导出。

- 代码实现：
  - 新增 `modules/memory/adk/infra_adapter.py`
    - `HttpMemoryInfraAdapter`（12 个 HTTP 方法）
    - 统一错误载荷：`{"status_code": int, "body": Any}`
    - 错误体解析策略：JSON 优先，失败回退文本
    - 生命周期：`aclose()` / `async with`
  - 新增 `modules/memory/adk/runtime.py`
    - `MemoryAdkRuntime`（9 个已实现工具方法）
    - 同步工厂 `create_memory_runtime(...)`
    - 支持 `default_user_tokens` + 每次调用 `user_tokens` 覆盖
    - 集成 `StatePropertyVocabManager`
  - 新增 `modules/memory/adk/tool_definitions.py`
    - `MemoryToolDefinition` 数据结构
    - `TOOL_DEFINITIONS`（9 个工具定义）
    - `to_openai_tools()` / `to_mcp_tools()` 导出
  - 更新 `modules/memory/adk/__init__.py` 导出
    - `HttpMemoryInfraAdapter` / `MemoryAdkRuntime` / `create_memory_runtime`
    - `TOOL_DEFINITIONS` / `MemoryToolDefinition` / `to_openai_tools` / `to_mcp_tools`

- 测试验证：
  - 新增测试：
    - `modules/memory/tests/unit/test_adk_infra_adapter.py`
    - `modules/memory/tests/unit/test_adk_runtime.py`
    - `modules/memory/tests/unit/test_adk_tool_definitions.py`
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_*.py -q` -> `88 passed`
    - `pytest modules/memory/tests -q` -> `582 passed, 7 skipped`

- 文档记录：
  - 更新 `ADK开发者说明文档.md`
    - 首屏改为 `create_memory_runtime` 三行接入
    - 保留“高级用法：自定义 adapter”
    - 明确 `tool_definitions.py` 为工具定义单一信源
    - 增加“用户意图 -> 工具”速查表
    - 租户隔离改为条件式表述（依赖服务端鉴权/绑定策略）
  - 更新根目录文档：
    - `SDK使用说明.md`（新增 ADK Runtime 快速接入）
    - `开发者API 说明文档.md`（补充 ADK 文档入口引用）
  - 更新 `modules/memory/adk/TODO.md`（新增并勾选阶段 F）

- 目标符合性评估：
  - T1/T2/T3/T5：满足。ADK 运行时、HTTP 适配层、工具定义导出与测试已闭环。
  - T4：满足。文档已与当前代码行为对齐，并形成“代码为准”的单一信源约束。

## 017. 审查收口修复：GET query list 校验 + quotes schema 上限对齐（已完成）

- 背景：
  - 外部评审提出两点：
    1) 需确认 `state_properties_api` 的 GET + query list（`user_tokens`）是否与服务端契约一致。
    2) `tool_definitions.py` 中 `quotes.limit.maximum=50` 与 ADK 实现 clamp `1..10` 不一致。

- 代码实现：
  - 更新 `modules/memory/adk/tool_definitions.py`
    - `quotes.limit.maximum` 从 `50` 调整为 `10`，与 `memory_tools.quotes()` 一致。
  - 更新 `modules/memory/tests/unit/test_adk_infra_adapter.py`
    - 增加 GET query list 编码断言：`user_tokens=["u:alice","u:bob"]` 应编码为重复 query 参数。
  - 更新 `modules/memory/tests/unit/test_adk_tool_definitions.py`
    - 新增 `test_quotes_limit_schema_aligns_with_runtime_cap`，防止 schema 与实现再漂移。

- 测试验证：
  - `pytest modules/memory/tests/unit/test_adk_infra_adapter.py modules/memory/tests/unit/test_adk_tool_definitions.py -q` -> `8 passed`
  - `pytest modules/memory/tests -q` -> `584 passed, 6 skipped`

- 结论：
  - 问题 1：已通过测试明确 GET + Query(list[str]) 兼容性（与 `/memory/v1/state/properties` 的服务端签名一致）。
  - 问题 2：已修复，工具 schema 与运行时上限完全对齐。

## 018. ADK 文档对齐修正：测试统计与覆盖清单更新（已完成）

- 背景：
  - 对照 `ADK开发者说明文档.md` 做进展对齐时发现测试统计口径滞后。

- 文档修正：
  - 更新 `ADK开发者说明文档.md` §11：
    - 新增 `Runtime / Adapter / ToolDefs` 测试分组
    - `test_adk_*.py` 文件数：`13 -> 17`
    - ADK 单测结果：补充 `89 passed`
    - 全量回归结果：`571 passed, 6 skipped -> 584 passed, 6 skipped`

- 结论：
  - 文档与当前实现、测试基线重新对齐。

## 019. Stage D 完成：`list_entities` / `list_topics` 发现工具落地（已完成）

- 目标：
  - 按 Wave 2 详细规范补齐 Stage D 两个发现工具：`list_entities`、`list_topics`。
  - 保持 Layer 1 边界：只做参数前置校验/分页封装/错误归一，不下沉语义编排。

- 代码实现：
  - 更新 `modules/memory/adk/memory_tools.py`
    - 新增类型别名：`ListEntitiesFn`、`ListTopicsFn`
    - 新增工具：`list_entities(...)`、`list_topics(...)`
    - 新增辅助：cursor 本地校验、`min_events` 非负归一
    - 关键行为：
      - `mentioned_since` 非 ISO 前置拦截（`invalid_input`）
      - `limit` clamp `1..50`
      - 非法 `cursor` 回退首页并在 `message/debug` 标记
      - `auto_page=true` 最多聚合 `max_pages<=3`
      - `status_thresholds` 仅进调试层
  - 更新 `modules/memory/adk/infra_adapter.py`
    - 新增 `list_entities_api()` -> `GET /memory/v1/entities`
    - 新增 `list_topics_api()` -> `GET /memory/v1/topics`
  - 更新 `modules/memory/adk/runtime.py`
    - 新增 `runtime.list_entities()` / `runtime.list_topics()`
  - 更新 `modules/memory/adk/tool_definitions.py`
    - 新增 `list_entities` / `list_topics` 定义（默认 `default_enabled=false`）
  - 更新 `modules/memory/adk/__init__.py`
    - 导出 `list_entities` / `list_topics`

- 测试验证：
  - 新增测试：
    - `modules/memory/tests/unit/test_adk_memory_tools_list_entities.py`
    - `modules/memory/tests/unit/test_adk_memory_tools_list_topics.py`
  - 更新测试：
    - `modules/memory/tests/unit/test_adk_infra_adapter.py`（新增两个 GET 端点断言）
    - `modules/memory/tests/unit/test_adk_runtime.py`（新增运行时发现工具调用）
    - `modules/memory/tests/unit/test_adk_tool_definitions.py`（工具定义集合扩展）
  - 执行结果：
    - `pytest modules/memory/tests/unit/test_adk_*.py -q` -> `100 passed`
    - `pytest modules/memory/tests -q` -> `595 passed, 6 skipped`

- 文档记录：
  - 更新 `ADK开发者说明文档.md`
    - Stage D 从“未实现”改为“已实现（低频发现工具）”
    - 新增意图 -> `list_entities/list_topics` 映射
    - 更新动态工具注入示例（增加 `DISCOVERY_TOOL_NAMES`）
    - 更新测试统计基线
  - 更新 `modules/memory/adk/TODO.md`
    - D1/D2 勾选完成

- 目标符合性评估：
  - Stage D 目标满足：两项发现工具已实现、已测试、已文档化。

## 020. ADK 工具卡片重构：补齐可直接落地的 Tool Call 协议（已完成）

- 背景：
  - 评审反馈 `ADK工具卡片.md` 虽有工具说明，但对“是否可直接作为 system prompt”与“tool call 命令怎么写”不够明确。

- 文档实现：
  - 重写 `ADK工具卡片.md`，结构改为两层：
    - `§1` 直接可复制的 System Prompt 正文
    - `§2~§5` 开发者调用协议与最小可执行代码
  - 补齐关键缺口：
    - 明确 OpenAI function calling 的 `tool_calls` / `role=tool` 回传格式
    - 给出 MCP `tools/call` 的参数示例
    - 增加 11 个工具的最小参数速查表（含 `list_entities` / `list_topics`）
    - 增加 Explain 两步链路和歧义处理范式
    - 给出可直接复用的 `TOOL_EXECUTORS` 代码片段

- 测试验证：
  - 本轮为文档重构，不涉及代码路径变更；未新增自动化测试。

- 目标符合性评估：
  - 该文档已从“概念说明”升级为“可复制执行的集成手册”，可直接指导 Agent 工具接入与 system prompt 配置。

## 021. ADK 工具卡片结构重排：纯净 Prompt 与开发者版分离（已完成）

- 背景：
  - 用户反馈上一版虽然有 Tool Call 协议，但“纯净 Prompt 与开发者集成内容混在一起”，且 Prompt 本体缺少调用协议约束。

- 文档实现：
  - 重构 `ADK工具卡片.md` 为两大部分：
    - Part A: `System Prompt 纯净版`（可直接粘贴）
    - Part B: `开发者集成版`（OpenAI/MCP 调用格式、工具速查、最小集成代码）
  - 在 Part A 内补齐“模型侧工具调用协议”：
    - `function.name + function.arguments(JSON)`
    - 仅按四字段 (`matched/needs_disambiguation/message/data`) 决策
    - 禁止暴露 debug

- 测试验证：
  - 本轮为文档结构优化，不涉及代码逻辑；未新增自动化测试。

- 目标符合性评估：
  - 文档现在既能直接当 System Prompt 用，也能作为工程接入手册使用，分层职责清晰。

## 022. ADK 工具卡片增强：补齐完整工具调用闭环示例（已完成）

- 背景：
  - 用户明确要求在工具卡片中补充“从工具 schema 传入到最终回答”的完整调用闭环，降低 Agent 集成落地门槛。

- 文档实现：
  - 更新 `ADK工具卡片.md`：
    - 新增 `B6. 完整调用闭环（请求1 -> 工具执行 -> 请求2）`
    - 提供可直接复用的两次 `chat.completions.create` 代码模板
    - 明确展示 `function.name`、`function.arguments` 的读取位置与 `role=tool` 回传写法
    - 增加 ASCII 流程图，便于排障
  - 原 `B6/B7` 顺延为 `B7/B8`。

- 测试验证：
  - 本轮为文档增强，不涉及代码行为变更；未新增自动化测试。

- 目标符合性评估：
  - `ADK工具卡片.md` 现已覆盖“可直接复制 Prompt + 可直接运行调用闭环”两类使用场景。

## 023. 工具卡片评审修复：异步闭环与 unknown tool 单一信源（已完成）

- 背景：
  - 评审指出两项可运行性/一致性问题：
    1) B6 示例混用同步 `OpenAI` 与异步 `await fn(...)`。
    2) B5 `unknown tool` 回包手写 JSON，未复用 `ToolResult`。

- 修复内容：
  - 更新 `ADK工具卡片.md`：
    - B6 改为 `AsyncOpenAI`，两次 `chat.completions.create` 均改为 `await`，并补充“需在 async 函数中运行”注释。
    - B5 改为 `ToolResult.no_match(...).to_llm_dict()` 生成 unknown-tool 返回结构。

- 测试验证：
  - 文档示例修复，不涉及运行时代码路径；未新增自动化测试。

- 目标符合性评估：
  - 示例可直接在异步环境执行，且错误返回口径与 `ToolResult` 单一信源一致。

## 024. Stage E（Layer 2）落地：`/memory/agentic/*` 单工具路由 API（已完成 E1/E2）

- 目标：
  - 将“自然语言 -> 单工具调用 -> 结构化结果”下沉为服务端 API，避免业务 Agent 重复实现 tool routing 闭环。
  - 保持 v1 克制：**单请求只执行 1 个工具**，不做多步编排，不生成自然语言答案。

- 代码实现：
  - 更新 `modules/memory/api/server.py`
    - 新增路由：
      - `GET /memory/agentic/tools`
      - `POST /memory/agentic/execute`
      - `POST /memory/agentic/query`
    - 新增请求模型：`AgenticQueryBody`、`AgenticExecuteBody`
    - 新增辅助能力：
      - tool whitelist 归一化与未知工具校验
      - 工具参数 schema 校验（unknown arg / required args）
      - router 调用（OpenAI function calling，单 tool call）
      - runtime 执行与 ToolResult 统一输出
    - 新增鉴权 scope 映射：`/memory/agentic/` -> `memory.read`
    - 新增 `/api/list` 分类：`memory_agentic`

- 测试验证：
  - 新增 `modules/memory/tests/unit/test_memory_agentic_api.py`
    - 覆盖 tools/execute/query 三端点
    - 覆盖无 tool_call、未知工具、router 参数非法、required 参数缺失等边界
  - 回归结果：
    - `pytest modules/memory/tests/unit/test_memory_agentic_api.py -q` -> `7 passed`
    - `pytest modules/memory/tests -q` -> `602 passed, 6 skipped`

- 文档记录：
  - 更新根目录文档：
    - `SDK使用说明.md`（新增 Agentic Query API 接入示例）
    - `开发者API 说明文档.md`（新增 5.24 Agentic 语义路由章节）
  - 更新 ADK 文档：
    - `ADK开发者说明文档.md`（Layer 2 状态从未实现改为已实现，版本升级到 v2.3）
  - 更新 TODO：
    - `modules/memory/adk/TODO.md` 勾选 E1/E2，保留 E3（准确率/延迟验收）

- 目标符合性评估：
  - Stage E1/E2 已闭环：路由协议、单工具执行、结构化返回均落地且通过回归。
  - 后续仅剩 E3（线上路由质量与时延指标验收）。

## 025. Stage E 路由器适配升级：统一多 Provider 配置解析（已完成）

- 背景：
  - `/memory/agentic/query` 初版路由器直接依赖 OpenAI SDK 默认配置，无法稳定覆盖多 provider 场景。
  - 本轮目标：把 Layer 2 路由模型选择对齐到统一 adapter 路径，支持从 `modules/memory/config` 指定 provider/model。

- 实施内容：
  - 新增 `llm_adapter` 统一解析函数：
    - `resolve_openai_compatible_chat_target(kind="agentic_router")`
    - 解析优先级：`memory.llm.agentic_router` -> `memory.llm.text`
    - 支持 provider：`openai/openrouter/qwen(dashscope)/glm/deepseek/moonshot/sglang/openai_compat`
  - Layer 2 调整：
    - `_agentic_route_tool_call` 改为依赖上述解析结果创建 client
    - 增加 env 覆盖：`MEMORY_AGENTIC_ROUTER_PROVIDER/MODEL/BASE_URL`
    - query 返回 `meta.provider`，便于观测实际路由 provider
  - 配置补齐：
    - `modules/memory/config/memory.config.yaml` 增加 `memory.llm.agentic_router`
    - `modules/memory/config/hydra/memory.yaml` 增加 `memory.llm.agentic_router`

- 测试验证：
  - 新增 `modules/memory/tests/unit/test_llm_adapter_chat_target.py`（4 用例）
  - 回归：
    - `pytest modules/memory/tests/unit/test_llm_adapter_chat_target.py modules/memory/tests/unit/test_memory_agentic_api.py -q` -> `11 passed`
    - `pytest modules/memory/tests -q` -> `607 passed, 6 skipped`

- 文档同步：
  - `开发者API 说明文档.md`：补充 `meta.provider` 与 router 选模优先级
  - `SDK使用说明.md`：补充 `agentic_router` 配置示例
  - `ADK开发者说明文档.md`：补充 Layer 2 路由配置入口与环境覆盖

- 目标符合性评估：
  - Layer 2 已具备“统一 adapter 入口 + 配置驱动选模”的可运维能力。

## 026. 评审收敛修复：Layer 2 Prompt / Config / Fallback（已完成）

- 背景：
  - 收到评审后，确认三项问题需立即收敛：
    - router prompt 约束不足（意图映射信息不够）
    - 配置文件重复 `api` key 造成 `topk_defaults` 被覆盖
    - 路由模型 fallback 链跨 provider 串值风险

- 修复内容：
  - `server.py`
    - 强化 `_AGENTIC_ROUTER_SYSTEM_PROMPT`：
      - 明确“不要在工具调用之外输出任何文本”
      - 增加高频工具意图映射（entity_profile/topic_timeline/time_since/quotes/relations）
  - `memory.config.yaml` + `hydra/memory.yaml`
    - 合并重复 `api` 节点，保留 `topk_defaults` 与 `auth/limits/retrieval`
  - `llm_adapter.py`
    - `resolve_openai_compatible_chat_target` 的 model fallback 改为 provider 内收敛，不再跨 provider 读取 `*_MODEL`

- 测试验证：
  - `test_config_hydra_loader.py` 新增：
    - `test_api_topk_defaults_and_auth_coexist_in_loaded_config`
  - `test_llm_adapter_chat_target.py` 新增：
    - `test_resolve_chat_target_does_not_cross_fallback_other_provider_models`
  - 回归：
    - `pytest modules/memory/tests/unit/test_llm_adapter_chat_target.py modules/memory/tests/unit/test_config_hydra_loader.py modules/memory/tests/unit/test_memory_agentic_api.py -q` -> `17 passed`
    - `pytest modules/memory/tests -q` -> `609 passed, 6 skipped`

- 文档同步：
  - `ADK工具卡片.md`
    - 补充 Layer 1/Layer 2 边界说明
    - 新增 `B9`（Layer 2 直连样例）
  - `ADK开发者说明文档.md`
    - 版本更新到 `v2.4`
    - 补充 Layer 2 选模与观测信息

- 目标符合性评估：
  - 评审提及的高优先级工程风险均已修复并经回归验证通过。
