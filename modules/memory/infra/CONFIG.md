# 存储配置示例（Qdrant / Neo4j / 审计SQLite）

本文件提供记忆模块后端存储的示例配置，便于替换内存实现为真实服务。

## Qdrant（向量库）

建议集合：
- `memory_text`（文本）
- `memory_image`（图像）
- `memory_audio`（音频）

示例配置：
```yaml
memory:
  vector_store:
    kind: qdrant
    host: 127.0.0.1
    port: 6333
    api_key: "${QDRANT_API_KEY}"
    prefer_grpc: false
    collections:
      text: memory_text
      image: memory_image
      audio: memory_audio
```

连接说明：
- 无鉴权时可省略 `api_key`。
- `prefer_grpc` 取决于部署环境。

## Neo4j（图数据库）

节点标签建议：`Image|Voice|Episodic|Semantic|Device|Room|User|Character`

示例配置：
```yaml
memory:
  graph_store:
    kind: neo4j
    uri: bolt://127.0.0.1:7687
    user: neo4j
    password: "${NEO4J_PASSWORD}"
    vector_index:
      enabled: true
      metric: cosine
      dims:
        text: 1536
        image: 512
        audio: 256
```

## 审计（SQLite）

示例：
```yaml
memory:
  audit:
    sqlite_path: data/memory_audit.db
```

> 生产建议迁移至 SQLite/PG 持久化，并开启定期备份。

