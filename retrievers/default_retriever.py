# retrievers/default_retriever.py
from langchain_community.vectorstores import Chroma
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from typing import List

class DefaultSimilarityRetriever:
    """
    一个简单的默认检索器，基于 Chroma 向量存储的相似性搜索。
    """
    def __init__(self, vector_store: Chroma, search_kwargs: dict = None):
        if not isinstance(vector_store, Chroma):
            raise ValueError("vector_store 必须是 Chroma 实例。")
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs if search_kwargs is not None else {'k': 4} # 默认检索4个文档

    def as_langchain_retriever(self) -> BaseRetriever:
        """
        将此检索器包装为 Langchain兼容的 BaseRetriever。
        """
        return self.vector_store.as_retriever(search_kwargs=self.search_kwargs)

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        直接使用 vector_store 的 similarity_search 方法。
        """
        # 合并默认的 search_kwargs 和调用时传入的 kwargs
        current_search_kwargs = {**self.search_kwargs, **kwargs}
        return self.vector_store.similarity_search(query, **current_search_kwargs)

# --- 为未来高级召回策略预留的接口和基类 ---

class BaseAdvancedRetriever(BaseRetriever):
    """
    高级召回策略的基类接口。
    所有自定义的高级检索器都应继承此类并实现 _get_relevant_documents 方法。
    """
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        子类必须实现此方法以根据查询检索文档。

        参数:
            query: 用户的查询字符串。
            run_manager: Langchain 提供的回调管理器，用于日志记录等。

        返回:
            检索到的相关文档列表。
        """
        raise NotImplementedError("子类必须实现 _get_relevant_documents 方法")

    # 可以根据需要添加其他通用方法或属性
    # 例如，配置加载、元数据处理等


# 示例：未来如何实现一个基于元数据过滤的检索器 (伪代码)
# class MetadataFilteringRetriever(BaseAdvancedRetriever):
#     def __init__(self, vector_store, metadata_field, metadata_value_filter):
#         super().__init__()
#         self.vector_store = vector_store
#         self.metadata_field = metadata_field
#         self.metadata_value_filter = metadata_value_filter # e.g., a function or specific value
#
#     def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
#         # 1. 首先进行向量相似性搜索
#         # initial_docs = self.vector_store.similarity_search(query, k=10) # 获取较多候选
#         # 2. 然后根据元数据进行过滤
#         # filtered_docs = [doc for doc in initial_docs if self.metadata_value_filter(doc.metadata.get(self.metadata_field))]
#         # 或者，如果 Chroma 支持元数据过滤的搜索，直接使用它：
#         # filtered_docs = self.vector_store.similarity_search(
#         #    query,
#         #    k=5,
#         #    filter={self.metadata_field: self.metadata_value_filter} # Chroma filter 语法
#         # )
#         # return filtered_docs
#         pass


if __name__ == '__main__':
    # 这个部分可以用来测试 DefaultSimilarityRetriever
    # 但需要一个实际的 Chroma 实例，通常在集成测试中进行
    print("retrievers/default_retriever.py 包含默认检索器和高级检索器基类。")
    print("可以通过实例化 DefaultSimilarityRetriever 并传入 Chroma 向量库来使用。")
    print("BaseAdvancedRetriever 为自定义检索策略提供了模板。")