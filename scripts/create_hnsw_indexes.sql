-- ============================================================================
-- HNSW 向量索引迁移脚本
-- 使用方式: docker exec -i gakki-ai-fitness-postgres-1 psql -U ai_fitness -d fitness_assistant < scripts/create_hnsw_indexes.sql
-- ============================================================================

-- 启用 pgvector 扩展（幂等）
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- exercises 表: HNSW 索引（余弦距离）
-- ============================================================================
-- 参数说明:
--   m = 16: 每层最大连接数。1024 维向量建议 16-32。值越大→召回越高但构建越慢/内存越多
--   ef_construction = 64: 构建时的搜索深度。越大→索引质量越高但构建越慢
--   当前 ~100-300 条数据，16/64 完全够用。扩展到万级后可调高
-- ============================================================================
DROP INDEX IF EXISTS exercises_embedding_hnsw_idx;
CREATE INDEX exercises_embedding_hnsw_idx ON exercises
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- knowledge_chunks 表: HNSW 索引（余弦距离）
-- ============================================================================
DROP INDEX IF EXISTS knowledge_chunks_embedding_hnsw_idx;
CREATE INDEX knowledge_chunks_embedding_hnsw_idx ON knowledge_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- 验证索引
-- ============================================================================
SELECT
  tablename,
  indexname,
  indexdef
FROM pg_indexes
WHERE indexname LIKE '%hnsw%'
ORDER BY tablename, indexname;

-- ============================================================================
-- 检查索引大小
-- ============================================================================
SELECT
  indexname,
  pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE indexname LIKE '%hnsw%';
