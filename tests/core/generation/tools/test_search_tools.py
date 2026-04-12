import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
sys.path.insert(0, str(project_root))

from core.workflow.video_summary.tools.search_tools import execute_tavily_search

class TestSearchTools(unittest.TestCase):
    
    @patch.dict(os.environ, clear=True)
    def test_execute_tavily_search_no_key(self):
        """边界情况 1：无 API Key 时应平滑降级，而不是抛出异常"""
        result = execute_tavily_search("test query")
        self.assertIn("Tool Execution Failed", result)
        self.assertIn("TAVILY_API_KEY is not configured", result)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "fake_tavily_key"})
    @patch('urllib.request.urlopen')
    def test_execute_tavily_search_success(self, mock_urlopen):
        """一般情况：成功请求 Tavily 并解析结果"""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {"content": "This is fine is a meme..."},
                {"content": "It features a dog..."}
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        result = execute_tavily_search("this is fine dog")
        self.assertIn("Web Search Results:", result)
        self.assertIn("This is fine is a meme", result)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "fake_tavily_key"})
    @patch('urllib.request.urlopen')
    def test_execute_tavily_search_network_error(self, mock_urlopen):
        """边界情况 2：网络超时或外部 API 报错时，应被捕获并返回平滑降级的字符串"""
        mock_urlopen.side_effect = Exception("Timeout")
        
        result = execute_tavily_search("query")
        self.assertIn("Tool Execution Failed", result)
        self.assertIn("Network error", result)
        self.assertIn("Timeout", result)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"})
    @patch('urllib.request.urlopen')
    def test_execute_tavily_search_empty_results(self, mock_urlopen):
        """边界情况 3：请求成功，但搜索结果为空列表，不应抛出 IndexError"""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"results": []}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = execute_tavily_search("some extremely rare keyword that yields nothing")
        self.assertIsInstance(result, str)
        self.assertIn("No relevant information found on the internet", result)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"})
    @patch('urllib.request.urlopen')
    def test_execute_tavily_search_malformed_json(self, mock_urlopen):
        """边界情况 4：外部 API 返回意外 JSON 结构（缺少 results 键），应安全降级而非 KeyError"""
        mock_response = MagicMock()
        # 模拟配额耗尽或 API 变更时返回的非标准结构
        mock_response.read.return_value = json.dumps({"error": "Rate limit exceeded"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = execute_tavily_search("query")
        # 源码使用 .get("results", []) 安全提取，故不抛出 KeyError，而是返回空结果提示
        self.assertIsInstance(result, str)
        self.assertIn("No relevant information found on the internet", result)

if __name__ == '__main__':
    unittest.main()