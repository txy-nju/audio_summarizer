import unittest
import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(project_root))

from services.workflow_service import VideoSummaryService
from utils.file_utils import clear_temp_folder

# 尝试可靠地加载项目根目录的 .env 文件
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

# [优化建议 1 落地] 消除硬编码绝对路径，使用相对路径指向轻量级测试样本
# 请确保在 tests/data/ 目录下放置了一个名为 sample_e2e.mp4 的极短测试视频 (如 3-5秒, < 1MB)
TEST_VIDEO_PATH = project_root / "tests" / "data" / "sample_e2e.mp4"

class TestEndToEndPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """设置全链路 E2E 测试环境"""
        # [优化建议 2 落地] 增加 E2E 测试的执行开关，防止 CI/CD 或本地全量测试时意外消耗大量 Token 和时间
        run_e2e = os.getenv("RUN_E2E", "false").lower() == "true"
        if not run_e2e:
            raise unittest.SkipTest("Skipping expensive E2E test. Set RUN_E2E=true in environment to execute.")

        cls.api_key = os.getenv("OPENAI_API_KEY")
        cls.base_url = os.getenv("OPENAI_BASE_URL")
        
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY not found in environment variables. Skipping E2E test.")
        
        if not TEST_VIDEO_PATH.exists():
             raise unittest.SkipTest(f"Test video file not found at {TEST_VIDEO_PATH}. Please put a small mp4 file there.")

        # 创建独立的 E2E 测试产物输出目录
        cls.output_dir = project_root / "test_output" / "e2e_pipeline"
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """
        [优化建议 3 落地] 中间产物的彻底清理
        确保无论测试成功与否，系统临时目录（抽取音频、切帧）都被干净抹除，防止在 CI 机器上填满磁盘。
        """
        print("\n🧹 Cleaning up temporary artifacts after E2E test...")
        clear_temp_folder()

    def test_full_pipeline_execution(self):
        """
        [全局 E2E 集成测试]
        测试宏观链路：前端文件流上传 -> 提取层(音频分离/防抖抽帧/Whisper转录) -> AI工作流(双轨分析/组装/审查重写) -> 最终交付
        这是一次真实网络请求、真实文件 I/O 与真实图状态机流转的大联调。
        """
        print(f"\n🚀 Starting End-to-End Pipeline Test with video: {TEST_VIDEO_PATH.name}")
        
        # 1. 实例化核心门面服务 (Facade Service)
        service = VideoSummaryService(api_key=self.api_key, base_url=self.base_url)
        
        # 2. 模拟前端 UI 传入二进制文件流
        with open(TEST_VIDEO_PATH, "rb") as video_file:
            user_prompt = "这是一次全链路 E2E 自动化测试，请严格提取画面中的图文内容，并验证你的双重反思拦截器是否生效。请极其简短地总结（总字数严格控制在300字以内）。"
            
            print("⏳ Processing... (Phase 1: Video Extraction & Whisper Transcription)")
            print("⏳ Processing... (Phase 2: Multimodal LangGraph Execution)")
            
            # 3. 第一阶段：生成待审批聚合稿
            review_package = service.analyze_uploaded_video(
                uploaded_file=video_file, 
                original_filename=TEST_VIDEO_PATH.name,
                user_prompt=user_prompt
            )

        self.assertIsInstance(review_package, dict, "第一阶段应返回待审批包")
        self.assertEqual(review_package.get("stage"), "pending_human_review")

        # 4. 第二阶段：不额外改写聚合稿，走空 guidance 的默认审批降级路径
        final_summary = service.finalize_summary(
            thread_id=str(review_package.get("thread_id", "")),
            edited_aggregated_chunk_insights=str(review_package.get("editable_aggregated_chunk_insights", "")),
            human_guidance="",
        )
            
        # 5. 严酷断言检查
        self.assertIsInstance(final_summary, str, "最终产物必须是字符串格式的 Markdown")
        self.assertTrue(len(final_summary) > 20, "总结内容过短，疑似流程中断或严重降级")
        
        # [优化建议 4 落地] 断言大模型是否听从了“极其简短”的非确定性指令约束
        self.assertTrue(len(final_summary) < 1500, "总结内容过长，大模型未遵循'简短'约束或发生了幻觉发散")

        self.assertNotIn("[系统自动提示]：综合图文大纲失败", final_summary, "图状态机中途发生了未处理的兜底降级异常")
        self.assertNotIn("文本分析提取失败", final_summary, "文本分析节点崩溃")
        self.assertNotIn("视觉分析提取失败", final_summary, "视觉分析节点崩溃")
        
        # 6. 落盘保存以供肉眼复核 (Artifacts Persistence)
        output_path = self.output_dir / "e2e_final_summary.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# 全链路 E2E 测试结果 (End-to-End Output)\n\n")
            f.write(f"**Target Video:** {TEST_VIDEO_PATH.name}\n")
            f.write(f"**User Prompt:** {user_prompt}\n\n")
            f.write(f"**Thread ID:** {review_package.get('thread_id', '')}\n\n")
            f.write(f"---\n\n")
            f.write(final_summary)
            
        print(f"\n✅ E2E Test Passed! Final summary perfectly synthesized and saved to: {output_path}")

if __name__ == '__main__':
    unittest.main()