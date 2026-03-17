import unittest
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import base64

# 确保绝对可靠地加载根目录的 .env 文件
project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
env_path = project_root / '.env'

if env_path.exists():
    print(f"Loading environment variables from: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)
else:
    print(f"[WARNING] .env file not found at {env_path}. Falling back to default env variables.")
    load_dotenv()

try:
    from core.extraction.sources.local_source import LocalFileVideoSource
except ImportError:
    import sys
    sys.path.insert(0, str(project_root))
    from core.extraction.sources.local_source import LocalFileVideoSource

# 【重要】请修改此路径为您本机实际存在的视频文件路径
TEST_VIDEO_PATH = Path(r"C:\Users\rbxu3\Downloads\偶尔小头控制大头可以理解、长期被控制就有问题.mp4")

class TestLocalSourceIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.api_key = os.getenv("OPENAI_API_KEY")
        cls.base_url = os.getenv("OPENAI_BASE_URL")
        
        print(f"Debug - Base URL loaded: {cls.base_url}")
        
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY not found in environment variables. Skipping integration test.")
        
        if not TEST_VIDEO_PATH.exists():
             print(f"Warning: Test video not found at {TEST_VIDEO_PATH}. Skipping test.")
             raise unittest.SkipTest(f"Test video file not found at {TEST_VIDEO_PATH}")

        test_name = Path(__file__).stem
        cls.output_dir = project_root / "test_output" / test_name
        
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Test outputs will be saved to: {cls.output_dir.resolve()}")

    def test_process_local_video(self):
        """测试 LocalFileVideoSource 处理真实视频并保存结果"""
        print(f"Testing with video: {TEST_VIDEO_PATH}")
        
        with open(TEST_VIDEO_PATH, "rb") as video_file:
            source = LocalFileVideoSource(
                uploaded_file=video_file,
                original_filename=TEST_VIDEO_PATH.name,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            print("Starting processing... this may take a while.")
            transcript, frames = source.process()
            
            # --- 保存结果 ---
            transcript_path = self.output_dir / "transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Transcript saved to {transcript_path}")
            
            frames_dir = self.output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            # 【核心改造】适配包含时间戳的字典列表
            for i, frame_dict in enumerate(frames):
                try:
                    time_str = frame_dict.get("time", "unknown_time")
                    frame_b64 = frame_dict.get("image")

                    if not frame_b64:
                        print(f"Skipping frame {i} due to missing image data.")
                        continue
                        
                    frame_data = base64.b64decode(frame_b64)
                    # 文件名中包含时间戳，便于追溯和测试复核
                    safe_time_str = time_str.replace(':', '-')
                    frame_path = frames_dir / f"frame_{i:03d}_{safe_time_str}.jpg"
                    with open(frame_path, "wb") as f:
                        f.write(frame_data)
                except Exception as e:
                    print(f"Failed to save frame {i} at time {time_str}: {e}")

            print(f"Saved {len(frames)} frames to {frames_dir}")
            
            # --- 断言 ---
            self.assertTrue(len(transcript) > 0, "Transcript should not be empty")
            self.assertTrue(len(frames) > 0, "Should extract at least one frame")
            
            # 【核心改造】增加对帧数据契约结构的严格断言
            first_frame = frames[0]
            self.assertIsInstance(first_frame, dict, "Frame data should be a dictionary")
            self.assertIn("time", first_frame, "Frame dictionary must contain a 'time' key")
            self.assertIn("image", first_frame, "Frame dictionary must contain an 'image' key")
            self.assertIsInstance(first_frame["image"], str, "Image data should be a base64 string")

if __name__ == '__main__':
    unittest.main()