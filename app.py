# 运行：streamlit run app.py --server.fileWatcherType none --server.address 0.0.0.0 --server.port 8501
# 不然会报错 RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

import streamlit as st
import os
# import tempfile # No longer using tempfile.NamedTemporaryFile directly for persistent temp files
from datetime import datetime
from pathlib import Path # Import Path

from utils.file_utils import load_document
from utils.vector_utils import split_documents, get_embedding_function
from knowledge_base.kb_manager import create_kb, list_kbs, get_kb_path, add_documents_to_kb, load_kb # 
from chains.qa_chain import get_llm, create_qa_chain, DEFAULT_PROMPT_TEMPLATE 
from retrievers.default_retriever import DefaultSimilarityRetriever 
# Import new config variables
from configs import (
    CHROMA_DB_PATH,
    TEMP_UPLOADS_DIR,
    EMBEDDING_MODEL,
    DOCUMENT_PROCESSING,
    SUPPORTED_FILE_TYPES,
    UI_CONFIG
    # API_CONFIG will be used later
)

# --- 应用配置 ---
st.set_page_config(
    page_title=UI_CONFIG.get("page_title", "拆书问答应用"),
    page_icon=UI_CONFIG.get("page_icon", "📚"),
    layout=UI_CONFIG.get("layout", "wide")
)

# --- 初始化 session_state ---
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "messages" not in st.session_state: # 用于聊天记录
    st.session_state.messages = []
if "current_kb_name" not in st.session_state:
    st.session_state.current_kb_name = None
# REMOVE embedding_function initialization from here
# REMOVE llm initialization from here


# --- 确保必要的目录存在 ---
if not CHROMA_DB_PATH.exists():
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
if not TEMP_UPLOADS_DIR.exists():
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# --- 辅助函数 ---
def get_available_kbs():
    """获取可用的知识库列表"""
    return list_kbs(kb_root_dir=str(CHROMA_DB_PATH)) # Pass configured path

def generate_kb_name_from_file(uploaded_file):
    """根据上传的文件名生成一个知识库名称"""
    if uploaded_file:
        filename_stem = Path(uploaded_file.name).stem # Use pathlib for robust name extraction
        # 简单清理文件名，替换空格和特殊字符
        safe_name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in filename_stem)
        return f"{safe_name}_kb"
    return None

# --- UI 界面 ---
st.title(UI_CONFIG.get("page_title", "📚 拆书问答应用")) # Use UI_CONFIG for title
st.caption("上传书籍，创建知识库，然后开始提问吧！")

# --- 侧边栏：知识库管理 ---
with st.sidebar:
    st.header("知识库管理")

    # 1. 创建新知识库
    st.subheader("上传新书创建知识库")
    uploaded_file = st.file_uploader(
        "选择文件:",
        type=SUPPORTED_FILE_TYPES, # Use configured supported types
        key="file_uploader"
    )
    
    custom_kb_name_input = st.text_input(
        "为新知识库命名 (可选, 留空则基于文件名自动生成):", 
        key="custom_kb_name"
    )

    if uploaded_file is not None:
        if st.button("创建/添加到知识库", key="create_kb_button"):
            # 将上传的文件保存到配置的临时目录
            # Generate a unique filename to avoid collisions in TEMP_UPLOADS_DIR
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uploaded_file.name}"
            temp_file_save_path = TEMP_UPLOADS_DIR / unique_filename
            
            try:
                with open(temp_file_save_path, "wb") as tmp_f:
                    tmp_f.write(uploaded_file.getvalue())
                st.session_state.uploaded_file_path = str(temp_file_save_path)
            except Exception as e:
                st.error(f"保存上传文件失败: {e}")
                st.stop()
            
            if st.session_state.uploaded_file_path:
                with st.spinner(f"正在处理 '{uploaded_file.name}'..."):
                    try:
                        # 1. 加载文档
                        st.info(f"步骤 1/4: 加载文档 '{uploaded_file.name}'...")
                        raw_docs = load_document(st.session_state.uploaded_file_path)
                        if not raw_docs:
                            st.error("未能从文件中加载任何内容。请检查文件是否为空或格式正确。")
                            if Path(st.session_state.uploaded_file_path).exists():
                                Path(st.session_state.uploaded_file_path).unlink()
                            st.session_state.uploaded_file_path = None
                            st.stop()

                        # 2. 切分文档 - Use DOCUMENT_PROCESSING config
                        st.info("步骤 2/4: 切分文档...")
                        split_docs = split_documents(
                            raw_docs,
                            chunk_size=DOCUMENT_PROCESSING.get("chunk_size", 1000),
                            chunk_overlap=DOCUMENT_PROCESSING.get("chunk_overlap", 200)
                        )
                        if not split_docs:
                            st.error("文档切分失败，没有生成任何文本块。")
                            if Path(st.session_state.uploaded_file_path).exists():
                                Path(st.session_state.uploaded_file_path).unlink()
                            st.session_state.uploaded_file_path = None
                            st.stop()
                        st.write(f"文档被切分为 {len(split_docs)} 个片段。")

                        # 3. 确定知识库名称
                        kb_name_to_use = custom_kb_name_input.strip() or generate_kb_name_from_file(uploaded_file)
                        if not kb_name_to_use:
                            st.error("无法确定知识库名称。")
                            if Path(st.session_state.uploaded_file_path).exists():
                                Path(st.session_state.uploaded_file_path).unlink()
                            st.session_state.uploaded_file_path = None
                            st.stop()
                        
                        st.info(f"步骤 3/4: 准备为知识库 '{kb_name_to_use}' 添加内容...")

                        # 4. 创建或更新知识库
                        # 检查知识库是否已存在 - Use CHROMA_DB_PATH
                        target_kb_full_path = Path(get_kb_path(kb_name_to_use, kb_root_dir=str(CHROMA_DB_PATH)))
                        embedding_func = st.session_state.embedding_function

                        if target_kb_full_path.exists():
                            st.info(f"知识库 '{kb_name_to_use}' 已存在，将向其添加新文档...")
                            vector_store = add_documents_to_kb(
                                kb_name=kb_name_to_use,
                                docs=split_docs,
                                embedding_function=embedding_func,
                                kb_root_dir=str(CHROMA_DB_PATH) # Pass configured path
                            )
                            if vector_store:
                                st.success(f"成功将 '{uploaded_file.name}' 的内容添加到现有知识库 '{kb_name_to_use}'！")
                            else:
                                st.error(f"向知识库 '{kb_name_to_use}' 添加文档失败。")
                        else:
                            st.info(f"正在创建新知识库 '{kb_name_to_use}'...")
                            vector_store = create_kb(
                                docs=split_docs,
                                embedding_function=embedding_func,
                                kb_name=kb_name_to_use,
                                kb_root_dir=str(CHROMA_DB_PATH), # Pass configured path
                                overwrite=False 
                            )
                            if vector_store:
                                st.success(f"知识库 '{kb_name_to_use}' 创建成功！包含 {len(split_docs)} 个向量化文档块。")
                            else:
                                st.error(f"创建知识库 '{kb_name_to_use}' 失败。")
                        
                        st.session_state.current_kb_name = kb_name_to_use
                        st.rerun()

                    except Exception as e:
                        st.error(f"处理文件并创建/更新知识库时发生错误: {e}")
                    finally:
                        # 清理临时文件
                        if st.session_state.uploaded_file_path and Path(st.session_state.uploaded_file_path).exists():
                            Path(st.session_state.uploaded_file_path).unlink() # Use Path.unlink()
                            st.session_state.uploaded_file_path = None
                        # Streamlit handles uploader reset automatically on rerun or new upload

    st.divider()

    # 2. 选择现有知识库
    st.subheader("选择知识库进行问答")
    available_kbs = get_available_kbs()
    
    if not available_kbs:
        st.info("暂无可用知识库。请先上传文件创建知识库。")
        st.session_state.current_kb_name = None
    else:
        if st.session_state.current_kb_name not in available_kbs:
            st.session_state.current_kb_name = None # 重置为 None 如果当前选择的KB不存在了

        current_kb_index = 0
        if st.session_state.current_kb_name and st.session_state.current_kb_name in available_kbs:
            current_kb_index = available_kbs.index(st.session_state.current_kb_name)
        
        selected_kb = st.selectbox(
            "选择一个知识库:",
            options=available_kbs,
            index=current_kb_index,
            key="selected_kb_dropdown",
            on_change=lambda: setattr(st.session_state, 'current_kb_name', st.session_state.selected_kb_dropdown) # 更新 current_kb_name
        )

        if selected_kb and selected_kb != st.session_state.current_kb_name: # 处理手动选择的情况
             st.session_state.current_kb_name = selected_kb
             st.session_state.messages = [] # 切换知识库时清空聊天记录
             st.rerun()


        if st.session_state.current_kb_name:
            st.success(f"当前选定知识库: **{st.session_state.current_kb_name}**")
        else:
            st.info("请选择一个知识库以开始问答。")

# --- 初始化核心组件（移到UI渲染后） ---
# 初始化嵌入函数
if "embedding_function" not in st.session_state:
    st.info(f"正在加载嵌入模型 ({EMBEDDING_MODEL.get('model_name', EMBEDDING_MODEL.get('local_path'))})... 这可能需要一些时间。")
    try:
        st.session_state.embedding_function = get_embedding_function(
            model_name=EMBEDDING_MODEL["local_path"]
        )
        st.success(f"嵌入模型 ({EMBEDDING_MODEL.get('model_name', EMBEDDING_MODEL.get('local_path'))}) 加载完成!")
    except Exception as e:
        st.error(f"嵌入模型加载失败: {e}")
        st.session_state.embedding_function = None

# 初始化 LLM
if "llm" not in st.session_state:
    st.info("正在初始化语言模型...")
    try:
        st.session_state.llm = get_llm(provider="glm")
        st.success("语言模型初始化完成!")
    except Exception as e:
        st.error(f"初始化语言模型失败: {e}")
        st.session_state.llm = None

# --- 主界面：聊天和问答 ---
st.header("开始问答")

if not st.session_state.current_kb_name:
    st.warning("请先在侧边栏选择或创建一个知识库。")
elif not st.session_state.llm:
    st.error("语言模型未能成功初始化，无法进行问答。请检查配置和API密钥。")
else:
    # 显示聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input(f"针对 '{st.session_state.current_kb_name}' 提问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            try:
                with st.spinner("思考中..."):
                    # 1. 加载选定的知识库向量存储
                    embedding_func = st.session_state.embedding_function
                    vector_store = load_kb(
                        kb_name=st.session_state.current_kb_name,
                        embedding_function=embedding_func,
                        kb_root_dir=str(CHROMA_DB_PATH)
                    )
                    if not vector_store:
                        raise ValueError(f"无法加载知识库 '{st.session_state.current_kb_name}'。")

                    # 2. 初始化检索器
                    retriever = DefaultSimilarityRetriever(
                        vector_store=vector_store,
                        search_kwargs={'k': DOCUMENT_PROCESSING.get("retrieval_k", 4)} # 可配置检索数量
                    ).as_langchain_retriever()


                    # 3. 创建问答链
                    qa_chain = create_qa_chain(
                        llm=st.session_state.llm,
                        retriever=retriever,
                        prompt_template_str=DEFAULT_PROMPT_TEMPLATE # 使用 chains 模块中定义的模板
                    )

                    # 4. 获取答案
                    response = qa_chain({"query": prompt})
                    answer = response.get("result", "抱歉，我无法回答这个问题。")
                    source_documents = response.get("source_documents", [])

                    full_response_content += answer
                    if source_documents:
                        full_response_content += "\n\n--- 参考文档 ---"
                        for i, doc in enumerate(source_documents):
                            # 为了简洁，只显示部分内容和来源
                            source_info = doc.metadata.get('source', '未知来源')
                            page_info = doc.metadata.get('page', '')
                            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                            full_response_content += f"\n\n**片段 {i+1} (来自: {source_info}{f', 第 {page_info+1} 页' if isinstance(page_info, int) else ''}):**\n{preview}"
                    
                    message_placeholder.markdown(full_response_content)

            except Exception as e:
                full_response_content = f"处理您的问题时发生错误: {e}"
                st.error(full_response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

# --- 简单的页脚 ---
st.markdown("---")
st.markdown("<end>")