import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from typing import List, Optional

# 假设 KNOWLEDGE_BASE_DIR 是存储所有知识库的根目录
# 你可以根据需要修改这个路径
# 使用 os.path.join 来确保路径的跨平台兼容性
DEFAULT_KB_ROOT_DIR = "vector_stores" 

def get_kb_path(kb_name: str, kb_root_dir: str = DEFAULT_KB_ROOT_DIR) -> str:
    """获取指定知识库的持久化存储路径"""
    if not kb_name:
        raise ValueError("知识库名称不能为空")
    # 清理 kb_name，防止路径遍历等问题，例如只允许字母数字下划线和短横线
    safe_kb_name = "".join(c for c in kb_name if c.isalnum() or c in ('_', '-')).strip()
    if not safe_kb_name:
        raise ValueError(f"无效的知识库名称: {kb_name}")
    return os.path.join(kb_root_dir, safe_kb_name)

def create_kb(
    docs: List[Document],
    embedding_function: SentenceTransformerEmbeddings,
    kb_name: str,
    kb_root_dir: str = DEFAULT_KB_ROOT_DIR,
    overwrite: bool = False
) -> Optional[Chroma]:
    """
    使用 Chroma 创建或更新向量数据库，并将文本块的向量存储进去。

    参数:
        docs (List[Document]): 包含文本块及其元数据的 Langchain Document 对象列表。
        embedding_function (SentenceTransformerEmbeddings): 用于生成文本向量的嵌入函数。
        kb_name (str): 知识库的名称，将用于创建持久化存储的子目录。
        kb_root_dir (str): 存储所有知识库的根目录。
        overwrite (bool): 如果知识库已存在，是否覆盖。默认为 False。

    返回:
        Chroma: 创建或加载的 Chroma 向量存储对象，如果创建失败则返回 None。
    """
    if not docs:
        print("没有文档可供创建知识库。")
        return None

    persist_directory = get_kb_path(kb_name, kb_root_dir)

    if os.path.exists(persist_directory):
        if overwrite:
            print(f"知识库 '{kb_name}' 已存在于 '{persist_directory}'，将执行覆盖操作。")
            try:
                shutil.rmtree(persist_directory) # 删除旧的知识库目录
                os.makedirs(persist_directory, exist_ok=True) # 重新创建目录
            except OSError as e:
                print(f"错误：无法删除或重新创建目录 '{persist_directory}': {e}")
                return None
        else:
            print(f"知识库 '{kb_name}' 已存在于 '{persist_directory}' 且未启用覆盖。")
            # 可以选择加载现有知识库或提示用户
            # 此处简单返回 None，或可以修改为加载现有
            try:
                print(f"尝试加载现有知识库 '{kb_name}'...")
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_function
                )
                print(f"成功加载现有知识库 '{kb_name}'。")
                # 如果需要向现有知识库添加文档，可以使用 vector_store.add_documents(docs)
                # 但当前函数设计为创建，如果存在且不覆盖，则不进行操作或加载
                return vector_store # 或者根据需求决定是否添加新文档
            except Exception as e:
                print(f"加载现有知识库 '{kb_name}' 失败: {e}")
                return None


    print(f"正在为知识库 '{kb_name}' 创建向量存储于: {persist_directory}")
    try:
        os.makedirs(persist_directory, exist_ok=True) # 确保目录存在
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vector_store.persist() # 确保数据被写入磁盘
        print(f"知识库 '{kb_name}' 创建成功并已持久化。包含 {len(docs)} 个文档块。")
        return vector_store
    except Exception as e:
        print(f"创建知识库 '{kb_name}' 失败: {e}")
        # 如果创建失败，清理可能已创建的目录
        if os.path.exists(persist_directory) and not os.listdir(persist_directory):
            try:
                shutil.rmtree(persist_directory)
            except OSError as oe:
                print(f"清理失败的知识库目录 '{persist_directory}' 时出错: {oe}")
        return None

def list_kbs(kb_root_dir: str = DEFAULT_KB_ROOT_DIR) -> List[str]:
    """列出所有已创建的知识库名称"""
    if not os.path.exists(kb_root_dir):
        return []
    return [name for name in os.listdir(kb_root_dir) if os.path.isdir(os.path.join(kb_root_dir, name))]

def load_kb(
    kb_name: str,
    embedding_function: SentenceTransformerEmbeddings,
    kb_root_dir: str = DEFAULT_KB_ROOT_DIR
) -> Optional[Chroma]:
    """
    加载指定的持久化 Chroma 向量知识库。

    参数:
        kb_name (str): 要加载的知识库的名称。
        embedding_function (SentenceTransformerEmbeddings): 用于加载知识库的嵌入函数。
        kb_root_dir (str): 存储所有知识库的根目录。

    返回:
        Chroma: 加载的 Chroma 向量存储对象，如果知识库不存在或加载失败则返回 None。
    """
    persist_directory = get_kb_path(kb_name, kb_root_dir)
    if not os.path.exists(persist_directory):
        print(f"知识库 '{kb_name}' 在路径 '{persist_directory}' 不存在。")
        return None
    
    try:
        print(f"正在从 '{persist_directory}' 加载知识库 '{kb_name}'...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        print(f"知识库 '{kb_name}' 加载成功。")
        return vector_store
    except Exception as e:
        # 常见的错误可能是 collection 不存在，或者 embedding_function 不匹配等
        print(f"加载知识库 '{kb_name}' 失败: {e}")
        return None

def add_documents_to_kb(
    kb_name: str,
    docs: List[Document],
    embedding_function: SentenceTransformerEmbeddings,
    kb_root_dir: str = DEFAULT_KB_ROOT_DIR
) -> Optional[Chroma]:
    """
    向现有的知识库添加新的文档。

    参数:
        kb_name (str): 目标知识库的名称。
        docs (List[Document]): 要添加的新文档列表。
        embedding_function (SentenceTransformerEmbeddings): 嵌入函数。
        kb_root_dir (str): 知识库根目录。

    返回:
        Chroma: 更新后的 Chroma 向量存储对象，如果失败则返回 None。
    """
    if not docs:
        print("没有要添加的文档。")
        return None

    vector_store = load_kb(kb_name, embedding_function, kb_root_dir)
    if not vector_store:
        print(f"无法加载知识库 '{kb_name}' 以添加文档。请先创建该知识库。")
        return None

    try:
        print(f"正在向知识库 '{kb_name}' 添加 {len(docs)} 个新文档块...")
        # Chroma 的 add_documents 方法会处理 ID 冲突（如果文档有自定义 ID）
        # 如果没有自定义 ID，它会为新文档生成 ID
        vector_store.add_documents(documents=docs)
        vector_store.persist() # 确保更改被保存
        print(f"成功向知识库 '{kb_name}' 添加文档。")
        return vector_store
    except Exception as e:
        print(f"向知识库 '{kb_name}' 添加文档失败: {e}")
        return None


if __name__ == '__main__':
    from utils.vector_utils import get_embedding_function, split_documents # 假设在同级或PYTHONPATH中
    
    # --- 初始化 ---
    TEST_KB_ROOT = "test_vector_stores"
    # 清理之前的测试目录
    if os.path.exists(TEST_KB_ROOT):
        shutil.rmtree(TEST_KB_ROOT)
    os.makedirs(TEST_KB_ROOT, exist_ok=True)

    try:
        # 获取嵌入函数
        print("--- 获取嵌入函数 ---")
        test_embeddings = get_embedding_function()
        if not test_embeddings:
            raise Exception("无法获取嵌入函数，测试中止。")
        print(f"嵌入函数类型: {type(test_embeddings)}")

        # 准备测试文档
        print("\n--- 准备测试文档 ---")
        doc1_content = "这是第一个测试文档，关于苹果公司。Apple Inc. is an American multinational technology company."
        doc2_content = "第二个文档是关于香蕉的。Bananas are elongated, edible fruit – botanically a berry."
        doc_list = [
            Document(page_content=doc1_content, metadata={"source": "doc1.txt"}),
            Document(page_content=doc2_content, metadata={"source": "doc2.txt"}),
        ]
        
        # 切分文档
        split_test_docs = split_documents(doc_list, chunk_size=50, chunk_overlap=10)
        print(f"切分后得到 {len(split_test_docs)} 个文档块。")
        for i, chunk in enumerate(split_test_docs):
            print(f"  块 {i+1}: '{chunk.page_content}' (源: {chunk.metadata.get('source')})")


        # --- 测试创建知识库 ---
        print("\n--- 测试创建知识库 'my_test_kb' ---")
        kb_name_1 = "my_test_kb"
        vector_db_1 = create_kb(split_test_docs, test_embeddings, kb_name_1, kb_root_dir=TEST_KB_ROOT)
        if vector_db_1:
            print(f"知识库 '{kb_name_1}' 创建成功。类型: {type(vector_db_1)}")
            # 简单查询测试
            query1 = "apple company"
            results1 = vector_db_1.similarity_search(query1, k=1)
            print(f"查询 '{query1}':")
            for res_doc in results1:
                print(f"  - {res_doc.page_content} (元数据: {res_doc.metadata})")
        else:
            print(f"知识库 '{kb_name_1}' 创建失败。")

        # --- 测试列出知识库 ---
        print("\n--- 测试列出知识库 ---")
        kbs = list_kbs(kb_root_dir=TEST_KB_ROOT)
        print(f"当前知识库列表: {kbs}")
        assert kb_name_1 in kbs

        # --- 测试加载知识库 ---
        print(f"\n--- 测试加载知识库 '{kb_name_1}' ---")
        loaded_kb_1 = load_kb(kb_name_1, test_embeddings, kb_root_dir=TEST_KB_ROOT)
        if loaded_kb_1:
            print(f"知识库 '{kb_name_1}' 加载成功。类型: {type(loaded_kb_1)}")
            query2 = "fruit berry"
            results2 = loaded_kb_1.similarity_search(query2, k=1)
            print(f"查询 '{query2}':")
            for res_doc in results2:
                print(f"  - {res_doc.page_content} (元数据: {res_doc.metadata})")
        else:
            print(f"知识库 '{kb_name_1}' 加载失败。")
        
        # --- 测试创建时覆盖 ---
        print(f"\n--- 测试创建知识库 '{kb_name_1}' (覆盖模式) ---")
        doc_overwrite_content = "这是一个用于覆盖测试的新文档。Overwrite test."
        overwrite_docs = split_documents([Document(page_content=doc_overwrite_content, metadata={"source": "overwrite.txt"})], chunk_size=30, chunk_overlap=5)
        vector_db_overwrite = create_kb(overwrite_docs, test_embeddings, kb_name_1, kb_root_dir=TEST_KB_ROOT, overwrite=True)
        if vector_db_overwrite:
            print(f"知识库 '{kb_name_1}' (覆盖后) 创建/更新成功。")
            query_after_overwrite = "Overwrite"
            results_overwrite = vector_db_overwrite.similarity_search(query_after_overwrite, k=1)
            print(f"查询 '{query_after_overwrite}':")
            for res_doc in results_overwrite:
                print(f"  - {res_doc.page_content} (元数据: {res_doc.metadata})")
            # 验证旧内容是否不存在或相关性降低
            results_old_content = vector_db_overwrite.similarity_search("apple company", k=1)
            print(f"查询旧内容 'apple company':")
            if results_old_content and "Apple Inc" not in results_old_content[0].page_content:
                 print("  - 旧内容已成功被覆盖。")
            else:
                 print(f"  - 旧内容可能仍然存在: {results_old_content[0].page_content if results_old_content else '无结果'}")

        else:
            print(f"知识库 '{kb_name_1}' (覆盖模式) 创建失败。")

        # --- 测试向现有知识库添加文档 ---
        print(f"\n--- 测试向知识库 '{kb_name_1}' 添加新文档 ---")
        # 确保 kb_name_1 是覆盖后的版本
        # 重新加载以确保我们操作的是磁盘上的最新版本
        current_kb_for_add = load_kb(kb_name_1, test_embeddings, kb_root_dir=TEST_KB_ROOT)
        if not current_kb_for_add:
            raise Exception(f"无法加载知识库 {kb_name_1} 以进行添加文档测试。")

        doc_additional_content = "这是额外添加的文档内容。Additional content about oranges."
        additional_docs_split = split_documents([Document(page_content=doc_additional_content, metadata={"source": "additional.txt"})], chunk_size=30, chunk_overlap=5)
        
        updated_kb = add_documents_to_kb(kb_name_1, additional_docs_split, test_embeddings, kb_root_dir=TEST_KB_ROOT)
        if updated_kb:
            print(f"成功向知识库 '{kb_name_1}' 添加文档。")
            query_additional = "oranges"
            results_additional = updated_kb.similarity_search(query_additional, k=1)
            print(f"查询 '{query_additional}':")
            for res_doc in results_additional:
                print(f"  - {res_doc.page_content} (元数据: {res_doc.metadata})")
            
            # 验证之前的内容（覆盖后的内容）是否仍然存在
            query_overwrite_again = "Overwrite"
            results_overwrite_again = updated_kb.similarity_search(query_overwrite_again, k=1)
            print(f"再次查询 '{query_overwrite_again}':")
            found_overwrite = False
            for res_doc in results_overwrite_again:
                print(f"  - {res_doc.page_content} (元数据: {res_doc.metadata})")
                if "Overwrite test" in res_doc.page_content:
                    found_overwrite = True
            assert found_overwrite, "覆盖测试的内容在添加新文档后应该仍然存在"

        else:
            print(f"向知识库 '{kb_name_1}' 添加文档失败。")


        # --- 测试加载不存在的知识库 ---
        print("\n--- 测试加载不存在的知识库 'non_existent_kb' ---")
        loaded_non_existent = load_kb("non_existent_kb", test_embeddings, kb_root_dir=TEST_KB_ROOT)
        if loaded_non_existent is None:
            print("成功处理：尝试加载不存在的知识库返回 None。")
        else:
            print("错误：加载不存在的知识库应返回 None。")

    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    finally:
        # 清理测试目录
        print(f"\n--- 清理测试目录 {TEST_KB_ROOT} ---")
        if os.path.exists(TEST_KB_ROOT):
            shutil.rmtree(TEST_KB_ROOT)
        print("测试完成。")