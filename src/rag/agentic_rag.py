from src.rag.vector_search import VectorSearch
from src.rag.keyword_search import KeywordSearch
from src.llm.provider import LLMProvider
from src.llm.prompts.retriever import build_retriever_eval_messages
from src.config import AGENTIC_RAG_MAX_RETRIES


class AgenticRAG:
    def __init__(self):
        self.vector = VectorSearch()
        self.keyword = KeywordSearch()
        self.llm = LLMProvider()

    def search(self, query: str, filters: dict = None, max_retries: int = None) -> list:
        max_retries = max_retries or AGENTIC_RAG_MAX_RETRIES
        current_query = query
        all_results = []

        for attempt in range(max_retries):
            vec_results = self.vector.search(current_query, top_k=5, filters=filters)
            kw_results = self.keyword.search(current_query, top_k=5)
            combined = self._deduplicate(vec_results + kw_results)
            all_results.extend(combined)

            if attempt < max_retries - 1:
                eval_msgs = build_retriever_eval_messages(query, combined[:10])
                eval_result = self.llm.chat_with_json_mode(eval_msgs)
                score = eval_result.get("quality_score", 0)
                if score >= 0.7:
                    break
                current_query = eval_result.get("rewritten_query", current_query)

        return self._deduplicate(all_results)

    def _deduplicate(self, results: list) -> list:
        seen = set()
        unique = []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique
