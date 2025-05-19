import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# 将项目根目录添加到 sys.path，以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chains.qa_chain import get_llm
from configs import API_CONFIG
from langchain_community.chat_models import ChatZhipuAI # 用于类型检查或作为 mock 对象的规范
from langchain_core.messages import AIMessage # 用于检查响应类型

class TestGetLLM(unittest.TestCase):

    @patch('chains.qa_chain.ChatZhipuAI') # Mock ChatZhipuAI 类在 qa_chain 模块中的引用
    def test_get_llm_glm_initialization_mocked(self, MockChatZhipuAI):
        """
        测试 get_llm 函数是否能为 'glm' 提供商正确调用 ChatZhipuAI 初始化 (使用 Mock)。
        假设 API 密钥环境变量已设置。
        """
        # 从配置中获取 "glm" 提供商的 API 密钥环境变量名称
        if "glm" not in API_CONFIG or "api_key_env_var" not in API_CONFIG["glm"]:
            self.fail("API_CONFIG 中未正确配置 'glm' 或 'api_key_env_var'")
            return 

        glm_api_key_env_var = API_CONFIG["glm"]["api_key_env_var"]
        mock_api_key_value = "mock_glm_api_key_for_test"

        # 配置 mock ChatZhipuAI 实例
        mock_llm_instance = MagicMock(spec=ChatZhipuAI)
        MockChatZhipuAI.return_value = mock_llm_instance

        # 使用 patch.dict 临时设置环境变量，仅在此测试方法的作用域内有效
        with patch.dict(os.environ, {glm_api_key_env_var: mock_api_key_value}):
            llm = get_llm(provider="glm")

            self.assertIsNotNone(llm, "LLM 实例不应为 None")
            self.assertEqual(llm, mock_llm_instance, "返回的 LLM 实例应为 mock 对象")

            MockChatZhipuAI.assert_called_once_with(
                model=API_CONFIG["glm"]["llm_model_name"],
                api_key=mock_api_key_value, 
                temperature=API_CONFIG["glm"]["temperature"]
            )

    # 条件跳过：如果环境变量未设置，则跳过此测试
    # 首先检查配置文件中指定的 GLM_API_Key，然后检查 ZHIPUAI_API_KEY
    @unittest.skipIf(
        not os.getenv(API_CONFIG.get("glm", {}).get("api_key_env_var", "GLM_API_Key_placeholder")) and \
        not os.getenv("ZHIPUAI_API_KEY"),
        f"环境变量 {API_CONFIG.get('glm', {}).get('api_key_env_var', 'GLM_API_Key_placeholder')} 或 ZHIPUAI_API_KEY 未设置。跳过实际 API 调用测试。"
    )
    def test_get_llm_glm_live_api_call(self):
        """
        测试 get_llm 函数是否能为 'glm' 提供商进行实际的 API 调用并获取响应。
        需要正确设置 GLM_API_Key (或 ZHIPUAI_API_KEY) 环境变量并联网。
        """
        print("\n--- 开始实际 API 调用测试 ---")
        llm_instance = None
        try:
            # 调用 get_llm 获取真实的 LLM 实例
            # 注意：这里没有 @patch，所以会创建真实的 ChatZhipuAI 实例
            llm_instance = get_llm(provider="glm")
            
            self.assertIsNotNone(llm_instance, "LLM 实例 (实际调用) 不应为 None")
            self.assertIsInstance(llm_instance, ChatZhipuAI, "LLM 实例 (实际调用) 应为 ChatZhipuAI 类型")

            # 准备一个简单的提问
            prompt_text = "你好，请介绍一下你自己。"
            print(f"发送给 LLM 的提问: '{prompt_text}'")

            # 进行 API 调用
            # ChatZhipuAI().invoke("...") 返回一个 AIMessage 对象
            response_message = llm_instance.invoke(prompt_text)
            
            print("LLM 响应对象类型:", type(response_message))
            print("LLM 响应对象:", response_message)

            self.assertIsInstance(response_message, AIMessage, "LLM 响应应为 AIMessage 类型")
            
            response_content = response_message.content
            print(f"LLM 响应内容: '{response_content}'")

            self.assertIsNotNone(response_content, "LLM 响应内容不应为 None")
            self.assertTrue(len(response_content.strip()) > 0, "LLM 响应内容不应为空")
            print("--- 实际 API 调用测试成功 ---")

        except Exception as e:
            print(f"实际 API 调用测试失败，错误: {e}")
            if llm_instance is None:
                self.fail(f"LLM 实例未能成功初始化以进行实际 API 调用测试: {e}")
            else:
                self.fail(f"实际 API 调用测试中发生错误: {e}")

if __name__ == '__main__':
    unittest.main()