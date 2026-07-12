# PDF/Word/MD 文件上传 + 会话级 RAG

## 目标

用户上传体检报告/训练日志（PDF/Word/MD），基于文件内容进行 Q&A。
文件关联到会话（session），关闭页面后不可回溯。文件不进入公共知识库。

## 核心设计决策

- **存储**：PostgreSQL（原始文本+向量），不用 MinIO
- **PDF 解析**：pdfplumber（文本+表格+字号）
- **Word 解析**：python-docx（段落+表格+样式层级）
- **切块**：复用现有 chunk_text（500 字符，80 重叠）
- **检索**：Q&A 时双路检索（knowledge_chunks + document_chunks），RRF 融合

## 数据模型

### user_documents
- id, session_id, filename, file_type(pdf/docx/md), file_size
- raw_content(TEXT 全文), page_count, created_at

### document_chunks
- id, document_id(FK), session_id, chunk_index
- content(TEXT 含标题路径注入), chunk_type(text/table)
- title_path, page_number, embedding(vector 1024)

## 解析管线

PDF → pdfplumber 逐页提取 text + tables → 表格转 Markdown → 字号检测标题
Word → python-docx 逐段提取 text + tables → 样式检测标题层级
MD → 直接读取

## 切块时上下文注入

每个 chunk 注入：文档名 + 章节路径 + 页码

## Q&A 双路检索

knowledge_chunks（公共） + document_chunks WHERE session_id = ? → RRF 融合 → LLM

## 新增依赖

pdfplumber>=0.10.0, python-docx>=1.0.0

## 新增文件

src/parsers/ (pdf_parser, docx_parser, md_parser)
src/storage/document_store.py
app/server.py 新增 POST /api/upload-document

## 已知局限

- 扫描件 PDF 无法提取文字（检测+提示兜底）
- Word 嵌入图片 v1 不处理
- 单文件限制 20MB
