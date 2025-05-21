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
    UI_CONFIG,
    API_CONFIG
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
if "embedding_function" not in st.session_state: # 确保嵌入模型状态存在
    st.session_state.embedding_function = None
if "selected_llm_provider" not in st.session_state: # 新增：选择的LLM提供商
    # 尝试从 API_CONFIG 获取第一个作为默认值，否则为 None
    st.session_state.selected_llm_provider = list(API_CONFIG.keys())[0] if API_CONFIG else None
if "llm" not in st.session_state: # 新增：LLM实例
    st.session_state.llm = None


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

def _cleanup_uploaded_file(): 
    """辅助函数,清理临时上传文件"""
    if st.session_state.get("uploaded_file_path"):
        file_path_to_clean = Path(st.session_state.uploaded_file_path)
        if file_path_to_clean.exists():
            try:
                file_path_to_clean.unlink()
            except Exception as e:
                st.warning(f"清理临时文件 {file_path_to_clean} 失败: {e}")
        st.session_state.uploaded_file_path = None

# --- 模型初始化函数 ---
def initialize_embedding_model():
    """初始化或获取嵌入模型实例"""
    if st.session_state.embedding_function is None:
        try:
            # 使用主界面来显示状态
            with st.status("正在加载嵌入模型...", expanded=False) as status_bar:
                st.session_state.embedding_function = get_embedding_function(
                    model_name=EMBEDDING_MODEL.get("local_path") or EMBEDDING_MODEL.get("model_name")
                )
                status_bar.update(label="嵌入模型加载成功！", state="complete")
        except Exception as e:
            st.sidebar.error(f"加载嵌入模型失败: {e}")
            st.session_state.embedding_function = None
    return st.session_state.embedding_function

def initialize_llm():
    """根据选择的提供商初始化或获取LLM实例"""
    if st.session_state.llm is None and st.session_state.selected_llm_provider:
        try:
            with st.sidebar.status(f"正在初始化语言模型 ({st.session_state.selected_llm_provider})...", expanded=False) as status_bar:
                st.session_state.llm = get_llm(provider=st.session_state.selected_llm_provider)
                status_bar.update(label=f"语言模型 ({st.session_state.selected_llm_provider}) 初始化成功！", state="complete")
        except Exception as e:
            st.sidebar.error(f"初始化语言模型 ({st.session_state.selected_llm_provider}) 失败: {e}")
            st.session_state.llm = None
    elif not st.session_state.selected_llm_provider and "llm" in st.session_state : # 如果没有选择provider，llm应为None
        st.session_state.llm = None
    return st.session_state.llm

# --- 应用启动时初始化模型 ---
# initialize_embedding_model() # 嵌入模型初始化放在界面初始化之后
# initialize_llm() # LLM 初始化将由选择器触发或在首次需要时进行

# --- UI 界面 ---
st.title(UI_CONFIG.get("page_title", "📚 拆书问答应用")) # Use UI_CONFIG for title
st.caption("上传书籍，创建知识库，然后开始提问吧！")

# --- 侧边栏：知识库管理 ---
with st.sidebar:
    st.header("⚙️ 模型配置") # 新增模型配置区域

    available_llm_providers = list(API_CONFIG.keys())
    
    # 确保 selected_llm_provider 有效
    if not st.session_state.selected_llm_provider and available_llm_providers:
        st.session_state.selected_llm_provider = available_llm_providers[0]
    elif st.session_state.selected_llm_provider not in available_llm_providers and available_llm_providers:
         st.session_state.selected_llm_provider = available_llm_providers[0] # 重置为第一个有效的
         st.session_state.llm = None # 清空旧的LLM实例
    elif not available_llm_providers:
        st.session_state.selected_llm_provider = None
        st.session_state.llm = None


    if available_llm_providers:
        current_provider_index = 0
        if st.session_state.selected_llm_provider in available_llm_providers:
            current_provider_index = available_llm_providers.index(st.session_state.selected_llm_provider)

        selected_provider = st.selectbox(
            "选择语言模型 (LLM):",
            options=available_llm_providers,
            index=current_provider_index,
            key="llm_provider_selector"
        )
        if selected_provider != st.session_state.selected_llm_provider:
            st.session_state.selected_llm_provider = selected_provider
            st.session_state.llm = None # 重置LLM，将在下次使用时重新初始化
            st.session_state.messages = [] # （可选）切换模型时清空聊天记录
            st.rerun() # 重新运行以应用更改
    else:
        st.warning("没有配置可用的语言模型。请检查 `configs/config.py`。")

    # 在这里调用 initialize_llm，确保选择器更改后能立即尝试初始化
    # 或者在聊天逻辑中，如果 llm is None 再调用
    initialize_llm()


    st.divider()
    st.header("📚 知识库管理")

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
            embedding_func = initialize_embedding_model() # 确保嵌入模型已加载
            if not embedding_func:
                st.error("嵌入模型未能加载，无法处理知识库。")
                st.stop()
            
            # 将上传的文件保存到配置的临时目录
            temp_file_save_path = TEMP_UPLOADS_DIR / uploaded_file.name
            
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
                            _cleanup_uploaded_file() # 使用辅助函数清理
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
                            _cleanup_uploaded_file() # 使用辅助函数清理
                            st.stop()
                        st.write(f"文档被切分为 {len(split_docs)} 个片段。")

                        # 3. 确定知识库名称
                        kb_name_to_use = custom_kb_name_input.strip() or generate_kb_name_from_file(uploaded_file)
                        if not kb_name_to_use:
                            st.error("无法确定知识库名称。")
                            _cleanup_uploaded_file() # 使用辅助函数清理
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
                                embedding_function=embedding_func, # 使用已初始化的 embedding_func
                                kb_root_dir=str(CHROMA_DB_PATH) 
                            )
                            if vector_store:
                                st.success(f"成功将 '{uploaded_file.name}' 的内容添加到现有知识库 '{kb_name_to_use}'！")
                            else:
                                st.error(f"向知识库 '{kb_name_to_use}' 添加文档失败。")
                        else:
                            st.info(f"正在创建新知识库 '{kb_name_to_use}'...")
                            vector_store = create_kb(
                                docs=split_docs,
                                embedding_function=embedding_func, # 使用已初始化的 embedding_func
                                kb_name=kb_name_to_use,
                                kb_root_dir=str(CHROMA_DB_PATH), 
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
                        _cleanup_uploaded_file() # 使用辅助函数清理
                        # Streamlit handles uploader reset automatically on rerun or new upload

    st.divider()

    # 2. 选择现有知识库
    st.subheader("选择知识库进行提问")
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
            st.info("请选择一个知识库以开始提问。")
# --- 初始化核心组件（移到UI渲染后） ---
initialize_embedding_model()
# --- 主界面：聊天和问答 ---
st.header("开始提问")

if not st.session_state.current_kb_name:
    st.warning("请先在侧边栏选择或创建一个知识库。")
elif not st.session_state.llm:
    st.error("语言模型未能成功初始化，无法进行提问。请检查配置和API密钥。")
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