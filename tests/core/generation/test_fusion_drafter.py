import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path
import base64
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, str(project_root))

from core.workflow.video_summary.state import VideoSummaryState
from core.workflow.video_summary.nodes.text_analyzer import text_analyzer_node
from core.workflow.video_summary.nodes.vision_analyzer import vision_analyzer_node
from core.workflow.video_summary.nodes.fusion_drafter import fusion_drafter_node

env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

class TestFusionDrafterNode(unittest.TestCase):
    
    def setUp(self):
        """准备单元测试用的虚假 State"""
        self.valid_state: VideoSummaryState = {
            "transcript": "",
            "keyframes": [],
            "text_insights": "核心观点：苹果发布了新款 iPhone。金句：这是迄今为止最伟大的产品。",
            "visual_insights": "[00:15] PPT上显示了 A17 芯片的性能提升 20%。\n[00:30] 演讲者展示了钛金属外壳。",
            "user_prompt": "侧重芯片性能",
            "draft_summary": "",
            "revision_count": 0,
            "feedback_instructions": "",
            "hallucination_score": "",
            "usefulness_score": ""
        }

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key_123", "OPENAI_BASE_URL": "https://fake.url"})
    @patch('core.workflow.video_summary.nodes.fusion_drafter.OpenAI')
    def test_fusion_drafter_mocked_normal(self, mock_openai_class):
        """单元测试：一般情况的融合组装，验证参数是否正确透传"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "苹果发布了新款iPhone，搭载性能提升20%的A17芯片。"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = fusion_drafter_node(self.valid_state)
        
        # 验证大模型调用
        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(result["revision_count"], 1, "重写次数应该加 1")
        self.assertEqual(result["draft_summary"], "苹果发布了新款iPhone，搭载性能提升20%的A17芯片。")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key_123", "OPENAI_BASE_URL": "https://fake.url"})
    @patch('core.workflow.video_summary.nodes.fusion_drafter.OpenAI')
    def test_fusion_drafter_mocked_with_feedback(self, mock_openai_class):
        """单元测试：存在 feedback_instructions 时的回流重写逻辑，验证 System Prompt 的动态修改"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "修正版：添加了之前遗漏的关于钛金属的描述。"
        mock_client.chat.completions.create.return_value = mock_response
        
        state_with_feedback = self.valid_state.copy()
        state_with_feedback["revision_count"] = 1 # 假设已经是第1次重写失败，这是第2次尝试
        state_with_feedback["feedback_instructions"] = "你忘记提到钛金属外壳了！请务必加上。"
        
        result = fusion_drafter_node(state_with_feedback)
        
        # 验证提示词中是否成功注入了 feedback 和重写次数
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        system_msg = [msg["content"] for msg in messages if msg["role"] == "system"][0]
        
        self.assertIn("这是第 2 次重写草稿", system_msg)
        self.assertIn("你忘记提到钛金属外壳了！请务必加上", system_msg)
        self.assertEqual(result["revision_count"], 2, "重写次数应该加 1")

    def test_fusion_drafter_integration(self):
        """
        集成测试：真实的 API 级联调用验证。
        从 test_output 中读取提取层产生的真实转录文本和抽帧图片，
        按顺序执行 text_analyzer -> vision_analyzer -> fusion_drafter 的完整数据链。
        并将最终的草稿报告持久化保存到 test_output 的新文件夹中。
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY not found. Skipping integration test.")
            
        output_dir = project_root / "test_output" / "test_local_source_integration"
        transcript_path = output_dir / "transcript.txt"
        frames_dir = output_dir / "frames"
        
        if not transcript_path.exists() or not frames_dir.exists():
            self.skipTest("Integration test data not found in test_output. Please run test_local_source_integration.py first to generate local test data.")
            
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
            
        keyframes = []
        # 只取前 2 张图，为了节省自动化测试的运行时间和 API Token 成本
        for img_path in sorted(frames_dir.glob("*.jpg"))[:2]:
            time_str = "00:00"
            parts = img_path.stem.split("_")
            if len(parts) >= 3:
                time_str = parts[2].replace("-", ":")
                
            with open(img_path, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                keyframes.append({"time": time_str, "image": b64_data})
                
        # 初始化符合架构的 State
        state: VideoSummaryState = {
            "transcript": transcript,
            "keyframes": keyframes,
            "text_insights": "",
            "visual_insights": "",
            "draft_summary": "",
            "user_prompt": "这是一次自动化的集成测试，请尽量简短地将画面和文字结合进行总结。",
            "revision_count": 0,
            "feedback_instructions": "",
            "hallucination_score": "",
            "usefulness_score": ""
        }
        
        # 1. 生成 Text Insights
        print("\n--- [Integration] Running Text Analyzer Node ---")
        text_result = text_analyzer_node(state)
        self.assertIn("text_insights", text_result)
        state.update(text_result)
        
        # 2. 生成 Visual Insights
        print("\n--- [Integration] Running Vision Analyzer Node ---")
        vision_result = vision_analyzer_node(state)
        self.assertIn("visual_insights", vision_result)
        state.update(vision_result)
        
        # 3. 融合生成 Draft
        print("\n--- [Integration] Running Fusion Drafter Node ---")
        fusion_result = fusion_drafter_node(state)
        
        draft = fusion_result.get("draft_summary", "")
        print(f"\n====== [Generated Draft Summary] ======\n{draft}\n=======================================\n")
        
        # --- 4. 将产物持久化保存到独立目录 ---
        save_dir = project_root / "test_output" / "test_fusion_drafter_integration"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        draft_path = save_dir / "draft_summary.md"
        with open(draft_path, "w", encoding="utf-8") as f:
            # 同时将文字提炼和视觉提炼也附加上去，方便对照审查
            f.write("# 听觉侧洞察 (Text Insights)\n")
            f.write(state.get("text_insights", "") + "\n\n")
            f.write("# 视觉侧洞察 (Visual Insights)\n")
            f.write(state.get("visual_insights", "") + "\n\n")
            f.write("# 最终合成草稿 (Generated Draft)\n")
            f.write(draft)
            
        print(f"\n[Success] Execution artifacts saved to: {draft_path}")
        
        self.assertTrue(len(draft) > 20, "Draft summary should be successfully synthesized and not be empty.")
        self.assertEqual(fusion_result.get("revision_count"), 1)

if __name__ == '__main__':
    unittest.main()