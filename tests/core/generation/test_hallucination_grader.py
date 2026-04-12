import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, str(project_root))

from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.nodes.hallucination_grader import hallucination_grader_node, MAX_REVISIONS

class TestHallucinationGraderNode(unittest.TestCase):
    
    def setUp(self):
        """准备符合最新架构 State 的测试数据"""
        self.valid_state: VideoSummaryState = {
            "transcript": "",
            "keyframes": [],
            "user_prompt": "",
            "draft_summary": "草稿：这里说有一只飞天猪。",
            "text_insights": "听觉：没有提到飞天猪。",
            "visual_insights": "视觉：画面里只有一棵树。",
            "revision_count": 0,
            "hallucination_score": "",
            "usefulness_score": "",
            "feedback_instructions": ""
        }

    def test_short_circuit_empty_draft(self):
        """边界情况 1：草稿为空时，短路放行"""
        state = self.valid_state.copy()
        state["draft_summary"] = ""
        result = hallucination_grader_node(state)
        self.assertEqual(result["hallucination_score"], "no")
        self.assertEqual(result["feedback_instructions"], "")

    def test_short_circuit_max_revisions(self):
        """边界情况 2：达到最大重写次数，短路放行（防死循环）"""
        state = self.valid_state.copy()
        # [优化建议 1 落地]：消除魔法数字，直接使用从业务代码导入的常量 MAX_REVISIONS
        state["revision_count"] = MAX_REVISIONS
        result = hallucination_grader_node(state)
        self.assertEqual(result["hallucination_score"], "no")

    @patch.dict(os.environ, clear=True)
    def test_missing_api_key(self):
        """边界情况 3：缺少 API Key 报错"""
        with self.assertRaisesRegex(ValueError, ".*OPENAI_API_KEY.*"):
            hallucination_grader_node(self.valid_state)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key_123", "OPENAI_BASE_URL": "https://fake.url"})
    @patch('core.workflow.video_summary.nodes.hallucination_grader.OpenAI')
    # [优化建议 3 确认]：由于 OpenAI 客户端是在节点函数内实例化(按需实例化)的，这里的 Mock 完全安全且生效。
    def test_grader_no_hallucination(self, mock_openai_class):
        """一般情况 1：无幻觉，返回 score='no'"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        
        # 模拟大模型输出标准的 JSON
        mock_response.choices[0].message.content = json.dumps({
            "score": "no",
            "faulty_timestamp": "",
            "reason": ""
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        result = hallucination_grader_node(self.valid_state)
        
        mock_client.chat.completions.create.assert_called_once()
        
        # 验证是否开启了 JSON Mode
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs.get("response_format"), {"type": "json_object"})
        
        self.assertEqual(result["hallucination_score"], "no")
        self.assertEqual(result["feedback_instructions"], "")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key_123", "OPENAI_BASE_URL": "https://fake.url"})
    @patch('core.workflow.video_summary.nodes.hallucination_grader.OpenAI')
    def test_grader_yes_hallucination(self, mock_openai_class):
        """一般情况 2：检测到幻觉，返回 score='yes' 及具体的反馈指令"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        
        mock_response.choices[0].message.content = json.dumps({
            "score": "yes",
            "faulty_timestamp": "第一段",
            "reason": "源数据未提及飞天猪，属于捏造。"
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        result = hallucination_grader_node(self.valid_state)
        
        self.assertEqual(result["hallucination_score"], "yes")
        self.assertIn("发生位置 第一段", result["feedback_instructions"])
        self.assertIn("源数据未提及飞天猪", result["feedback_instructions"])

    @patch('builtins.print')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key_123"})
    @patch('core.workflow.video_summary.nodes.hallucination_grader.OpenAI')
    def test_grader_api_or_json_error(self, mock_openai_class, mock_print):
        """边界情况 4：API报错或 JSON 解析失败，降级为无幻觉 (no)，并严格验证降级日志记录"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # 模拟输出非 JSON 格式引发 JSONDecodeError
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "这是一段普通的文字，不是 JSON"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = hallucination_grader_node(self.valid_state)
        self.assertEqual(result["hallucination_score"], "no", "非 JSON 响应必须降级放行")
        
        # [优化建议 2 落地]：断言日志是否正确捕获并输出了 JSON 解析错误
        print_args = [call_args[0][0] for call_args in mock_print.call_args_list]
        json_error_logged = any("Error or Invalid JSON" in arg for arg in print_args)
        self.assertTrue(json_error_logged, "系统应当详细记录 JSON 解析失败的降级日志，以便排查")
        
        mock_print.reset_mock()
        
        # 模拟网络异常
        mock_client.chat.completions.create.side_effect = Exception("API Server Timeout")
        result2 = hallucination_grader_node(self.valid_state)
        self.assertEqual(result2["hallucination_score"], "no", "网络异常必须降级放行")

        # 断言网络超时等硬报错也被记录
        print_args_timeout = [call_args[0][0] for call_args in mock_print.call_args_list]
        timeout_error_logged = any("API Server Timeout" in arg for arg in print_args_timeout)
        self.assertTrue(timeout_error_logged, "系统应当详细记录 API 超时导致降级的日志，以便排查")

if __name__ == '__main__':
    unittest.main()