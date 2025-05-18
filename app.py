# 运行：streamlit run app.py --server.fileWatcherType none
# 不然会报错 RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

import streamlit as st
import os
# import tempfile # No longer using tempfile.NamedTemporaryFile directly for persistent temp files
from datetime import datetime
from pathlib import Path # Import Path

from utils.file_utils import load_document
from utils.vector_utils import split_documents, get_embedding_function
from knowledge_base.kb_manager import create_kb, list_kbs, get_kb_path, add_documents_to_kb # DEFAULT_KB_ROOT_DIR is no longer used from here
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
if "embedding_function" not in st.session_state:
    # 初始化嵌入函数，避免重复加载
    # Using EMBEDDING_MODEL["local_path"] as the model_name for SentenceTransformerEmbeddings
    # Assuming EMBEDDING_MODEL["local_path"] is a valid path or model identifier
    # The 'device' will use the default 'cpu' from get_embedding_function if not specified here
    with st.spinner(f"正在加载嵌入模型 ({EMBEDDING_MODEL.get('model_name', EMBEDDING_MODEL.get('local_path'))})... 这可能需要一些时间。"):
        st.session_state.embedding_function = get_embedding_function(
            model_name=EMBEDDING_MODEL["local_path"] # Or EMBEDDING_MODEL["model_name"] if that's preferred
        )

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
            st.session_state.current_kb_name = None

        current_kb_index = 0
        if st.session_state.current_kb_name and st.session_state.current_kb_name in available_kbs:
            current_kb_index = available_kbs.index(st.session_state.current_kb_name)
        
        selected_kb = st.selectbox(
            "选择一个知识库:",
            options=available_kbs,
            index=current_kb_index,
            key="kb_selector",
            on_change=lambda: setattr(st.session_state, 'current_kb_name', st.session_state.kb_selector)
        )
        if selected_kb and selected_kb != st.session_state.current_kb_name :
            st.session_state.current_kb_name = selected_kb
            st.session_state.messages = [] 
            st.rerun()

    if st.session_state.current_kb_name:
        st.success(f"当前操作的知识库: **{st.session_state.current_kb_name}**")
    else:
        st.warning("未选择知识库。问答功能将作为通用聊天机器人。")

# --- 主聊天界面 ---
st.header("💬 开始提问")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        if st.session_state.current_kb_name:
            full_response = f"（模拟回答）您选择了知识库 '{st.session_state.current_kb_name}'。关于 '{prompt}' 的答案正在生成中..."
            response_placeholder.markdown(full_response + "▌") 
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            response_placeholder.markdown(full_response)
        else:
            full_response = f"（模拟通用聊天）您没有选择知识库。关于 '{prompt}' 的回复正在生成中..."
            response_placeholder.markdown(full_response + "▌")
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            response_placeholder.markdown(full_response)

# --- 简单的页脚 ---
st.markdown("---")
st.markdown("<end>")