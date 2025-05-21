import os
from pathlib import Path

# API 提供商配置
API_CONFIG = {
    "glm": {
        "api_key_env_var": "GLM_API_Key",
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "llm_model_name": "glm-4-flash",
        "llm_provider_type": "zhipuai",
        "temperature": 0.1
    },
    "ollama": { # 新增 Ollama 配置
        "api_base": "http://localhost:11434", # Ollama 默认 API 地址
        "llm_model_name": "qwen3:4b",           # 模型列表在 http://localhost:11434/api/tags
        "llm_provider_type": "ollama",
        "temperature": 0.7                    # Ollama 模型的温度参数
    }
    # 未来可以添加更多提供商的配置
}

# 项目路径配置
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # 处理 __file__ 不存在的情况
    PROJECT_ROOT = Path.cwd()

# 确保必要的目录存
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db_store"
TEMP_UPLOADS_DIR = PROJECT_ROOT / "temp_uploads"

# 模型配置
EMBEDDING_MODEL = {
    "local_path": "D:\\0-source\\models\\paraphrase-multilingual-MiniLM-L12-v2",
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2"
}

# 文档处理配置
DOCUMENT_PROCESSING = {
    "chunk_size": 1024,
    "chunk_overlap": 20
}

# 支持的文件类型
SUPPORTED_FILE_TYPES = ["pdf", "txt"]

# UI 配置
UI_CONFIG = {
    "page_title": "PDF智能问答助手",
    "page_icon": "📚",
    "layout": "wide"
}