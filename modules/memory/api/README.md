# memory.* API（MCP ToolSpec 摘要）

- memory.search: { query, topk, filters, expand_graph, threshold } → { hits[], neighbors{}, hints, trace }
- memory.write: { entries[], links[], upsert } → { version }
- memory.update: { id, patch, reason } → { version }
- memory.delete: { id, soft, reason } → { version }
- memory.link: { src_id, dst_id, rel_type, weight } → { ok }

详见 `memory_toolspec.json`。

模态过滤（filters.modality）
- 说明：可精确控制检索召回的模态，取值为数组：`["text", "image", "audio"]` 的任意组合。
- 示例：
  - 仅文本：`{"filters": {"modality": ["text"]}}`
  - 文本+图像：`{"filters": {"modality": ["text","image"]}}`
  - 全模态：`{"filters": {"modality": ["text","image","audio"]}}`

默认策略（未显式指定时）
- 可通过配置 `memory.search.ann.default_modalities` 或 `default_all_modalities` 设置默认召回模态；
- 也可运行时热更新：
  - GET `/config/search/ann`：查看覆盖
  - POST `/config/search/ann`：`{"default_modalities":["text","image","audio"]}` 或 `{"default_all_modalities": true}`
