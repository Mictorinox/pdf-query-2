from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    将加载的文档内容分割成小块。

    参数:
        documents (list[Document]): 从文件加载的 Langchain Document 对象列表。
        chunk_size (int): 每个文本块的最大字符数。
        chunk_overlap (int): 文本块之间的重叠字符数。

    返回:
        list[Document]: 切分后的 Langchain Document 对象列表。
    """
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # 添加起始索引元数据，方便溯源
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def get_embedding_function(model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
    """
    获取 HuggingFace 嵌入模型函数。

    参数:
        model_name (str): HuggingFace sentence-transformers 模型的名称或本地路径。
        device (str): 运行模型的设备 ('cpu', 'cuda', etc.)。

    返回:
        HuggingFaceEmbeddings: Langchain 嵌入函数实例。
    """
    # 注意：如果使用 'mps' (Apple Silicon GPU)，请确保 PyTorch 版本支持
    # 对于 'mps'，可能需要 `PYTORCH_ENABLE_MPS_FALLBACK=1` 环境变量
    model_kwargs = {'device': device}
    # 如果模型在本地，HuggingFaceEmbeddings 会自动处理
    # 如果模型需要从 Hugging Face Hub 下载，它也会处理
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    return embedding_function

if __name__ == '__main__':
    # --- 测试文本切分 ---
    print("--- 测试文本切分 ---")
    sample_text = "这是第一段长文本，它将被用于测试文本切分功能。我们希望看到它被正确地分割成多个小块。" \
                  "第二句话紧随其后，提供更多的内容。Langchain的RecursiveCharacterTextSplitter是一个常用的工具。" \
                  "它会尝试在语义边界（如换行符、句子结束符）进行切分。如果文本块仍然过长，它会继续递归切分。" \
                  "这是为了确保每个块的大小都在限制范围内，并且尽可能保持语义完整性。"
    
    # 模拟从文件加载的 Document 对象
    mock_doc = Document(page_content=sample_text, metadata={"source": "mock_source.txt", "page": 1})
    
    split_docs_output = split_documents([mock_doc], chunk_size=100, chunk_overlap=20)
    print(f"原始文本长度: {len(sample_text)}")
    print(f"切分后得到 {len(split_docs_output)} 个文档块:")
    for i, doc_chunk in enumerate(split_docs_output):
        print(f"  块 {i+1}: '{doc_chunk.page_content}' (长度: {len(doc_chunk.page_content)})")
        print(f"  块 {i+1} 元数据: {doc_chunk.metadata}")
    print("-" * 20)

    # --- 测试获取嵌入函数 ---
    print("\n--- 测试获取嵌入函数 ---")
    try:
        # 从 configs 加载模型路径，如果适用
        # from configs import EMBEDDING_MODEL # 假设可以这样导入
        # test_model_path = EMBEDDING_MODEL.get("local_path", 'all-MiniLM-L6-v2')
        # embeddings = get_embedding_function(model_name=test_model_path)
        
        # 为了独立测试，我们直接使用默认值或一个通用模型
        embeddings = get_embedding_function()
        print(f"成功获取嵌入函数: {type(embeddings)}")
        
        # 测试嵌入一个简单的句子
        test_sentence = "这是一个测试句子。"
        embedded_vector = embeddings.embed_query(test_sentence)
        print(f"测试句子: '{test_sentence}'")
        print(f"嵌入向量 (前5个维度): {embedded_vector[:5]}")
        print(f"嵌入向量维度: {len(embedded_vector)}")
        
        # 测试嵌入多个文档
        test_docs_content = ["文档一的内容。", "这是文档二。"]
        embedded_docs_vectors = embeddings.embed_documents(test_docs_content)
        print(f"\n测试文档内容: {test_docs_content}")
        print(f"嵌入文档向量数量: {len(embedded_docs_vectors)}")
        if embedded_docs_vectors:
            print(f"第一个文档的嵌入向量 (前5个维度): {embedded_docs_vectors[0][:5]}")
            print(f"第一个文档的嵌入向量维度: {len(embedded_docs_vectors[0])}")
            
    except Exception as e:
        print(f"获取或使用嵌入函数时出错: {e}")
        print("请确保已安装 sentence-transformers 和 PyTorch。")
        print("如果遇到模型下载问题，请检查网络连接或尝试手动下载模型。")

    print("-" * 20)