from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from pathlib import Path

from config.settings import DEFAULT_FRAME_INTERVAL
from core.extraction.infrastructure.extractor import MediaExtractor
from core.extraction.infrastructure.transcriber import AudioTranscriber

class VideoSource(ABC):
    """
    视频源抽象基类。
    封装了从视频源获取、提取、转录的完整流程。
    """
    def __init__(self, api_key: str, base_url: str = None):
        if not api_key:
            raise ValueError("API key is required for transcription.")
        
        self.extractor = MediaExtractor()
        self.transcriber = AudioTranscriber(api_key, base_url=base_url)

    @abstractmethod
    def acquire_video(self) -> Path:
        """
        获取视频文件并返回其本地路径。
        具体子类需实现此方法（如从 URL 下载或保存上传文件）。
        
        Returns:
            Path: 本地视频文件的绝对路径。
        """
        pass

    def process(self) -> Tuple[str, List[Dict]]:
        """
        模板方法：执行标准的视频处理流程。
        1. 获取视频 (acquire_video)
        2. 提取音频和关键帧 (self.extractor)
        3. 转录音频 (self.transcriber)
            
        Returns:
            Tuple[str, List[Dict]]: (转录文本, 包含时间戳与 base64 图像的关键帧字典列表)
        """
        # 1. 获取视频路径
        video_path = self.acquire_video()
        print(f"Video acquired at: {video_path}")

        # 2. 提取内容
        print("Extracting audio...")
        audio_path = self.extractor.extract_audio(video_path)
        print(f"Audio extracted to: {audio_path}")

        print("Extracting frames...")
        frames = self.extractor.extract_frames(video_path, interval=DEFAULT_FRAME_INTERVAL)
        print(f"Extracted {len(frames)} frames.")

        # 3. 生成转录
        print("Transcribing audio...")
        transcript = self.transcriber.transcribe(audio_path)
        print(f"Transcription complete. Length: {len(transcript)}")

        return transcript, frames