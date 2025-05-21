from langchain_community.chat_models import ChatZhipuAI
from langchain_community.chat_models.ollama import ChatOllama # 导入 ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from configs import API_CONFIG # 假设 API_CONFIG 在 configs 包的 __init__.py 中可导入
import os 

def get_llm(provider: str = "glm"):
    """
    根据配置初始化并返回一个 LLM 实例。
    支持 GLM (ZhipuAI) 和 Ollama。
    """
    if provider in API_CONFIG.keys():
        config = API_CONFIG[provider]
        llm = None # 初始化 llm 变量
        if provider == "glm":
            api_key = os.getenv(config.get("api_key_env_var"))
            llm = ChatZhipuAI(
                model=config["llm_model_name"],
                api_key=api_key, 
                temperature=config["temperature"]
            )
        elif provider == "ollama": # 新增对 Ollama 的处理
            llm = ChatOllama(
                base_url=config.get("api_base", "http://localhost:11434"), # 使用配置或默认的 base_url
                model=config["llm_model_name"], # 使用配置的模型名称
                temperature=config.get("temperature", 0.7) # 使用配置或默认的温度
            )
            # 您可以根据需要为 Ollama 添加更多参数，例如 num_ctx, top_k, top_p 等
            # 这些参数可以添加到 API_CONFIG["ollama"] 中，并在此处读取
        # else:
            # 如果未来添加了更多在 API_CONFIG 中配置但在 get_llm 中未实现逻辑的提供商，可以在此抛出错误
            # raise ValueError(f"LLM provider '{provider}' logic not implemented in get_llm, though configured.")
        
        if llm is not None:
            return llm
        else:
            # 如果 provider 在 API_CONFIG 中，但上面所有 if/elif 都没有匹配（理论上不应该发生如果配置正确）
            raise ValueError(f"无法为提供商 '{provider}' 初始化 LLM 实例。请检查 chains/qa_chain.py 中的 get_llm 函数。")
    else:
        raise ValueError(f"未配置的 LLM 提供商: {provider}。请检查 configs/config.py 中的 API_CONFIG。")

def create_qa_chain(llm, retriever, prompt_template_str: str = None):
    """
    创建一个 RetrievalQA 问答链。

    参数:
        llm: 初始化好的语言模型实例。
        retriever: 初始化好的 Langchain BaseRetriever 实例。
        prompt_template_str (str, optional): 自定义的 Prompt 模板字符串。
                                            如果为 None，则使用默认模板。

    返回:
        RetrievalQA: 配置好的问答链。
    """
    if prompt_template_str:
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" 是最常见的链类型，将所有检索到的文本块一次性放入提示中
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True # 可以选择是否返回源文档
        )
    else:
        # 使用 Langchain 的默认 Prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    return qa_chain

# 示例 Prompt 模板 (可选)
DEFAULT_PROMPT_TEMPLATE = """
请根据以下提供的上下文信息来回答问题。
如果你在上下文中找不到答案，请明确说明你不知道，不要试图编造答案。
尽量保持答案简洁。

上下文:
{context}

问题: {question}

回答:
"""

if __name__ == '__main__':
    # 这里可以添加一些简单的测试代码 (需要模拟 retriever 和 llm)
    print("QA Chain module loaded.")
    # 示例：
    # mock_llm = get_llm() # 假设 API Key 已配置
    # print(f"LLM initialized: {type(mock_llm)}")
    #
    # # 模拟一个 retriever (实际应用中会从知识库加载)
    # from langchain.schema import Document
    # from langchain.vectorstores.base import VectorStoreRetriever
    # class MockRetriever(VectorStoreRetriever):
    #     def _get_relevant_documents(self, query, *, run_manager):
    #         return [Document(page_content=f"这是关于'{query}'的模拟文档内容。")]
    #     async def _aget_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
    #         return [Document(page_content=f"这是关于'{query}'的异步模拟文档内容。")]
    #
    # mock_retriever_instance = MockRetriever(vectorstore=None, search_type="similarity", search_kwargs={}) # vectorstore 为 None 仅为示例
    #
    # qa_chain_instance = create_qa_chain(mock_llm, mock_retriever_instance, DEFAULT_PROMPT_TEMPLATE)
    # print(f"QA Chain created: {type(qa_chain_instance)}")
    #
    # # 模拟提问
    # # response = qa_chain_instance({"query": "什么是模拟文档？"})
    # # print(f"Response: {response}")