# PDF 智能问答助手

## 项目概述

本项目是一个轻量化的本地知识库构建工具，允许用户上传 PDF 或 TXT 文档构建本地知识库，并针对知识库中的内容进行提问。应用前端使用 Streamlit 构建，后端利用 Langchain 进行文档处理、向量化、检索和问答链的构建。

## 项目结构

```
.
├── .gitignore
├── app.py                      # Streamlit 主应用入口
├── chains/                     # Langchain 链相关模块
│   └── qa_chain.py             # 问答链的创建和 LLM 初始化
├── configs/                    # 配置模块
│   ├── init .py
│   └── config.py               # 核心配置文件 (API, 路径, 模型等)
├── knowledge_base/             # 知识库管理模块
│   └── kb_manager.py           # 创建、加载、管理 ChromaDB 知识库
├── requirements.txt            # Python 依赖列表
├── retrievers/                 # 检索器模块
│   └── default_retriever.py    # 默认的相似度检索器
├── temp_uploads/               # (运行时创建) 临时存储上传的文件
├── test/                       # 测试代码目录
│   └── test_api_config.py      # API 配置和 LLM 调用测试示例
├── utils/                      # 辅助工具模块
│   ├── file_utils.py           # 文件加载工具
│   └── vector_utils.py         # 文本分割和嵌入模型获取工具
└── chroma_db_store/            # (运行时创建) ChromaDB 向量数据库存储目录
```

## 快速开始

### 环境准备

```bash
git clone https://github.com/Mictorinox/pdf-query
cd pdf-query
pip install -r requirements.txt
```

### 配置 LLM

参见配置文件`configs/config.py`，以智谱 AI 为例，需要在环境变量中设置`GLM_API_Key`变量，值为智谱 ai 官网获得的 api key。

如果使用 ollama 部署本地模型，需要注意把配置文件`API_CONFIG ollama`的`"llm_provider_type"`字段修改为本地部署的模型名。

### 配置嵌入模型

配置文件`configs/config.py`的`EMBEDDING_MODEL`字段可以指定使用的本地嵌入模型。需要是 HuggingFace sentence_transformers 库支持的嵌入模型。另外，此处的本地路径可以删除，这种情况下会在首次运行时自动下载模型（会很慢，不推荐）。

### 启动应用

运行指令
```sh
streamlit run app.py --server.fileWatcherType none --server.address localhost --server.port 8501
```
此处需要指定 `--server.fileWatcherType none` 参数，避免报错

### 程序界面

启动后可以使用浏览器`localhost:8501`打开程序界面