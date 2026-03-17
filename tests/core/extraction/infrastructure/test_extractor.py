import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import shutil
import sys
import os
import cv2
import numpy as np
import base64
import re

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from core.extraction.infrastructure.extractor import MediaExtractor

class TestMediaExtractor(unittest.TestCase):

    def setUp(self):
        """创建测试所需的临时目录和文件"""
        self.test_output_audio_dir = Path("./test_temp_audios")
        self.test_input_video_dir = Path("./test_temp_videos_input")
        
        for d in [self.test_output_audio_dir, self.test_input_video_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        self.extractor = MediaExtractor(audio_dir=self.test_output_audio_dir)
        
        self.dummy_video_path = self.test_input_video_dir / "dummy_video.mp4"
        self.dummy_video_path.touch()

    def tearDown(self):
        """清理测试产生的临时目录"""
        for d in [self.test_output_audio_dir, self.test_input_video_dir]:
            if d.exists():
                shutil.rmtree(d)

    @patch('core.extraction.infrastructure.extractor.VideoFileClip')
    def test_extract_audio(self, mock_video_file_clip):
        """测试 extract_audio 方法"""
        mock_clip_instance = MagicMock()
        mock_video_file_clip.return_value.__enter__.return_value = mock_clip_instance
        mock_audio = MagicMock()
        mock_clip_instance.audio = mock_audio

        result_path = self.extractor.extract_audio(self.dummy_video_path)

        mock_video_file_clip.assert_called_once_with(str(self.dummy_video_path))
        expected_audio_path = self.test_output_audio_dir / "dummy_video.mp3"
        mock_audio.write_audiofile.assert_called_once_with(
            str(expected_audio_path), 
            codec='mp3', 
            logger=None
        )
        self.assertEqual(result_path, expected_audio_path)

    @patch('core.extraction.infrastructure.extractor.cv2.imencode')
    @patch('core.extraction.infrastructure.extractor.cv2.VideoCapture')
    def test_extract_frames(self, mock_video_capture, mock_imencode):
        """测试 extract_frames 方法，验证返回包含时间戳的字典列表"""
        # --- 准备模拟 ---
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        
        fps = 30
        mock_cap_instance.get.return_value = fps

        total_frames = 61
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        side_effect = [(True, fake_image) for _ in range(total_frames)]
        side_effect.append((False, None))
        mock_cap_instance.read.side_effect = side_effect

        fake_encoded_buffer = (True, np.array([1, 2, 3]))
        mock_imencode.return_value = fake_encoded_buffer
        
        # --- 执行测试 ---
        interval_seconds = 1
        frames = self.extractor.extract_frames(self.dummy_video_path, interval=interval_seconds)

        # --- 断言 ---
        mock_video_capture.assert_called_once_with(str(self.dummy_video_path))
        mock_cap_instance.get.assert_called_once_with(cv2.CAP_PROP_FPS)
        self.assertEqual(mock_cap_instance.read.call_count, total_frames + 1)

        # 验证抽帧次数
        expected_extracted_count = 3 # 0s, 1s, 2s
        self.assertEqual(mock_imencode.call_count, expected_extracted_count)
        
        # 验证返回的帧列表长度
        self.assertEqual(len(frames), expected_extracted_count)
        
        # 【核心改造】验证返回结果的结构和内容
        expected_b64_string = base64.b64encode(fake_encoded_buffer[1]).decode('utf-8')
        expected_timestamps = ["00:00", "00:01", "00:02"]

        for i, frame_dict in enumerate(frames):
            self.assertIsInstance(frame_dict, dict, "每个元素都应该是字典")
            self.assertIn("time", frame_dict, "字典应包含 'time' 键")
            self.assertIn("image", frame_dict, "字典应包含 'image' 键")
            
            # 验证时间戳
            self.assertEqual(frame_dict["time"], expected_timestamps[i])
            self.assertTrue(re.match(r"^\d{2}:\d{2}$", frame_dict["time"]), "时间戳格式应为 MM:SS")
            
            # 验证图像数据
            self.assertEqual(frame_dict["image"], expected_b64_string)

        mock_cap_instance.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()