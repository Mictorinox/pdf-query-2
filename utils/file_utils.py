import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

def load_document(file_path: str) -> list[Document]:
    """
    加载单个文档文件（TXT 或 PDF）。

    参数:
        file_path (str): 文档文件的路径。

    返回:
        list[Document]: Langchain Document 对象列表。

    异常:
        ValueError: 如果文件类型不受支持。
        Exception: 如果加载过程中发生其他错误。
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}。目前仅支持 .txt 和 .pdf 文件。")
        
        documents = loader.load()
        if not documents:
            raise ValueError(f"无法从文件 {file_path} 加载任何内容。文件可能为空或格式不正确。")
        return documents
    except ValueError as ve:
        print(f"文件加载错误: {ve}")
        raise
    except Exception as e:
        print(f"加载文件 '{file_path}' 时发生未知错误: {e}")
        raise

if __name__ == '__main__':
    # 创建一个临时的测试文件
    test_txt_file = "test_book.txt"
    test_pdf_file = "test_book.pdf" # 假设你有一个名为 test_book.pdf 的文件用于测试

    with open(test_txt_file, "w", encoding="utf-8") as f:
        f.write("这是第一行测试文本。\n")
        f.write("这是第二行，包含一些关于 Langchain 和 Streamlit 的内容。\n")
        f.write("向量数据库 Chroma 是一个不错的选择。\n")

    print(f"--- 测试加载 TXT 文件: {test_txt_file} ---")
    try:
        txt_docs = load_document(test_txt_file)
        if txt_docs:
            print(f"成功加载 {len(txt_docs)} 个 TXT 文档片段。")
            for i, doc in enumerate(txt_docs):
                print(f"片段 {i+1} 内容预览: {doc.page_content[:100]}...")
                print(f"片段 {i+1} 元数据: {doc.metadata}")
        else:
            print("未能加载 TXT 文档。")
    except Exception as e:
        print(f"加载 TXT 文件失败: {e}")
    finally:
        if os.path.exists(test_txt_file):
            os.remove(test_txt_file)

    print(f"\n--- 测试加载 PDF 文件 (请确保存在 {test_pdf_file} 文件) ---")
    # 注意：PDF 测试需要你手动放置一个名为 test_book.pdf 的文件在同级目录下
    if os.path.exists(test_pdf_file):
        try:
            pdf_docs = load_document(test_pdf_file)
            if pdf_docs:
                print(f"成功加载 {len(pdf_docs)} 个 PDF 文档页面/片段。")
                for i, doc in enumerate(pdf_docs):
                    print(f"页面/片段 {i+1} 内容预览: {doc.page_content[:100]}...")
                    print(f"页面/片段 {i+1} 元数据: {doc.metadata}")
            else:
                print("未能加载 PDF 文档。")
        except Exception as e:
            print(f"加载 PDF 文件失败: {e}")
    else:
        print(f"测试 PDF 文件 {test_pdf_file} 不存在，跳过 PDF 加载测试。")

    print("\n--- 测试不支持的文件类型 ---")
    try:
        load_document("unsupported.docx")
    except ValueError as e:
        print(f"捕获到预期的错误: {e}")