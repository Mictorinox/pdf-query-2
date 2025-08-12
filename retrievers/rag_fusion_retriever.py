from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma # 假设仍然使用 Chroma 作为向量存储
from langchain.chat_models.base import BaseChatModel # 导入 LLM 基类
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 可以定义一个函数来生成多个查询
def generate_queries(original_query: str, llm: BaseChatModel, num_queries: int = 4) -> List[str]:
    """
    使用 LLM 根据原始查询生成多个相关查询。

    Args:
        llm: 用于生成查询的 LLM 实例。
        original_query: 用户的原始查询字符串。
        num_queries: 希望生成的查询数量。

    Returns:
        包含生成的多个查询字符串的列表。
    """
    # 简单的查询生成 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个查询生成器。根据用户的问题，生成 {num_queries} 个不同的、相关的搜索查询，以帮助找到更多相关信息。每个查询之间用换行符分隔。只返回查询，不要包含其他任何文本。"),
        ("human", "{original_query}"),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        # 调用 LLM 生成查询
        response = chain.invoke({"original_query": original_query})
        # 将响应按行分割，去除空行和首尾空格
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries
    except Exception as e:
        print(f"生成查询失败: {e}")
        # 失败时返回空list
        return []
    pass

# 可以定义一个函数来执行 RRF 融合
def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60, final_k = 4) -> List[Document]:
    """
    对来自多个检索结果列表的文档进行 RRF 融合和重排序。

    Args:
        results: 一个列表的列表，每个子列表是来自一个查询的检索结果 (Document 对象列表)。
        k: RRF 计算中的常数，用于平滑得分。

    Returns:
        融合和重排序后的 Document 对象列表。
    """
    # 具体实现将计算 RRF 得分并重排序
    if not results:
        return []
    fused_scores: Dict[str, float] = {}
    document_map: Dict[str, Document] = {}

    for result_list in results:
        for rank, doc in enumerate(result_list):
            doc_id = doc.page_content # (TODO)使用更稳定的ID
            if doc_id not in document_map:
                document_map[doc_id] = doc
            score = 1 / (rank + k)

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += score

    # 按 RRF 得分降序排序
    sorted_fused_ids = sorted(fused_scores.items(), key=lambda doc_id: doc_id[1], reverse=True)
    
    fused_doc_list = [document_map[doc_id] for doc_id, _ in sorted_fused_ids]
    return fused_doc_list[:final_k]


class RagFusionRetriever(BaseRetriever):
    """
    实现 RAG Fusion 检索策略的自定义检索器。
    """
    vector_store: any # 向量存储实例
    llm: any # 用于生成查询的 LLM 实例
    num_generated_queries: int = 4 # 生成的查询数量
    rrf_k: int = 60 # RRF 常数
    final_k: int = 4 # 最终返回的文档数量

    def __init__(self, vector_store, llm,num_generated_queries: int = 4, rrf_k: int = 60, final_k: int = 4, **kwargs):
        super().__init__(llm=llm, vector_store=vector_store, **kwargs)
        self.llm = llm
        self.vector_store = vector_store
        self.num_generated_queries = num_generated_queries
        self.rrf_k = rrf_k
        self.final_k = final_k


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        实现 RAG Fusion 检索逻辑。

        Args:
            query: 用户的原始查询字符串。
            run_manager: Langchain 提供的回调管理器。

        Returns:
            融合和重排序后得到的顶部文档列表。
        """
        # 1. 生成多个查询
        generated_queries = generate_queries(query,self.llm,self.num_generated_queries)
        all_queries = [query] + generated_queries # 包含原始查询

        # 2. 并行检索
        all_results = []
        for q in all_queries:
            results = self.vector_store.similarity_search(q, k=self.final_k * 2) # 检索比最终数量更多的文档
            all_results.append(results)
        print(f"生成的查询: {all_queries}")


        # 3. 融合和重排序
        final_docs = reciprocal_rank_fusion(all_results, self.rrf_k,self.final_k)
        print(f"最终返回 {len(final_docs)} 个文档")
        return final_docs

    # 如果需要支持异步，可以实现 _aget_relevant_documents 方法
    # async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    #    pass

# 可以在这里添加一个工厂函数，方便在 app.py 中创建实例
# def create_rag_fusion_retriever(vector_store: Chroma, llm: BaseChatModel, **kwargs) -> RagFusionRetriever:
#    return RagFusionRetriever(vector_store=vector_store, llm=llm, **kwargs)