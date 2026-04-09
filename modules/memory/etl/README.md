# m3 VideoGraph → 统一记忆 ETL 说明

本文档描述了 m3-agent 生成的 VideoGraph pkl 的实际结构（基于样例 `bedroom_12.pkl`），以及映射到统一记忆体（MemoryEntry/Edge）的转换规则与使用方法。

## 1. 实际 Schema（样例探查结论）

- 顶层对象：`VideoGraph`
  - 关键属性：
    - `nodes: dict[int -> Node]`
    - `edges: dict[tuple(int,int) -> float]`
    - 其他：`text_nodes`、`text_nodes_by_clip`、`event_sequence_by_clip`、`character_mappings` 等

- Node（对象）常见字段：
  - `id: int`
  - `type: str` ∈ {`episodic`, `semantic`, `voice`, `img`}
  - `embeddings: list`（可选，多模向量/占位；文本/图像/语音节点均可能携带，ETL 会优先使用首个向量作为 `MemoryEntry.vectors`）
  - `metadata: dict`（关键：`contents: list[str]`，`timestamp` 可选）

- Edges（字典）：
  - 键：`(src_id:int, dst_id:int)`，值：`float` 权重
  - 样例统计（Top）：`episodic↔voice`、`semantic↔voice`、`episodic↔img`、`semantic↔img` 成对出现（双向）
  - 无显式关系类型字段，需要“按节点类型对”推断关系

> 样例 bedroom_12.pkl 节点计数：episodic=935, semantic=864, voice=16, img=2。

## 2. 映射到统一记忆体的规则

### 2.1 节点 → MemoryEntry

- `kind`
  - `episodic` → `episodic`
  - 其余（`semantic`/`voice`/`img`）→ `semantic`
- `modality`
  - `episodic`/`semantic` → `text`
  - `voice` → `audio`
  - `img` → `image`
- `contents`
  - 来自 `node.metadata.contents`（如为字符串则转为单元素 list），默认取首条（可按需要扩展多句）
- `metadata`
  - 固定写入：`{"source":"m3", "vg_node_id": 原始节点 id}`
  - 可选写入：`timestamp`（若存在且非空）
- `vectors`
  - 若 `node.embeddings` 存在，ETL 取首个向量作为对应 modality 的预计算向量（如 text/image/audio），优先用于向量入库

### 2.2 边 → Edge（关系）

- 关系推断（规范方向：`voice/img` 作为源 → 指向文本节点）
  - `voice → episodic|semantic` → `said_by`
  - `img → episodic|semantic` → `appears_in`
- 根据 clip 推断文本邻接：
  - 对同一 clip（`timestamp/clip_id`）内的文本节点，建立 `semantic → episodic` 的 `describes` 关系（权重 1.0）。
- 去重与合并
  - 样例里边普遍双向成对存在；ETL 仅保留规范方向并合并权重（累加）

> 若未来数据包含 `semantic ↔ episodic` 或其他关系，可在映射函数中扩展白名单与规则。

## 3. 工具与代码位置

- 映射与导入脚本：`MOYAN_Agent_Infra/modules/memory/etl/pkl_to_db.py`
- 演示脚本：`MOYAN_Agent_Infra/modules/memory/scripts/e2e_cycle13_etl_demo.py`
- 统一模型与服务：
  - `contracts/memory_models.py`（MemoryEntry/Edge）
  - `application/service.py`（MemoryService：写/改/删/连/搜 + 治理补齐 + 审计）

## 4. 使用方法

### 4.1 Dry-run（仅解析与统计）

```
PYTHONPATH=MOYAN_Agent_Infra \
python3 MOYAN_Agent_Infra/modules/memory/etl/pkl_to_db.py \
  --input MOYAN_Agent_Infra/modules/memory/etl/samples/bedroom_12.pkl \
  --dry-run
```

### 4.2 InMem 验证（不依赖外部后端）

```
PYTHONPATH=MOYAN_Agent_Infra \
python3 MOYAN_Agent_Infra/modules/memory/etl/pkl_to_db.py \
  --input MOYAN_Agent_Infra/modules/memory/etl/samples/bedroom_12.pkl \
  --inmem --limit 200
```

### 4.3 真实导入（Qdrant/Neo4j）

确保已正确填写 `modules/memory/config/.env` 与 `modules/memory/config/memory.config.yaml`，并已启动 Qdrant/Neo4j。

```
PYTHONPATH=MOYAN_Agent_Infra \
python3 MOYAN_Agent_Infra/modules/memory/etl/pkl_to_db.py \
  --input MOYAN_Agent_Infra/modules/memory/etl/samples/bedroom_12.pkl \
  --batch-size 1000
```

参数说明：
- `--dry-run`：仅输出映射统计（entries/relations），不写入
- `--limit`：限制导入的节点数量（便于试跑）
- `--batch-size`：批量写入大小，默认 1000
- `--inmem`：使用 InMem 存储（不访问 Qdrant/Neo4j）

说明：若 MemoryEntry.vectors 提供了预计算向量，Qdrant 写入将优先使用该向量；否则回退到 provider 或哈希嵌入。

## 5. 注意与扩展

- contents 的多句：当前默认仅写入首句，可按需要扩展为多向量写入（Qdrant 多向量字段/多条点）
- 关系细化：如未来存在 `semantic ↔ episodic` 等，可在 `map_edges_to_relations()` 增加推断规则与白名单
- 元数据扩充：
  - `clip_id`：样例中 `timestamp` 即 clip 编号，ETL 会同步写入 `metadata.clip_id`
  - `room/device`：若在节点 metadata 中出现，ETL 会为其创建结构化实体节点，并对 episodic 建立 `located_in`（room）。`executed`（device→episodic）可按需要开启
  - `user/character`：如后续提供，可在映射中按规则加入并建立 `prefer/equivalence` 等边
- 多模态真实嵌入：文本已接入，图像/语音可在后续按 provider 配置接入，不影响当前 ETL
