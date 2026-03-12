
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from pathlib import Path
from config.settings import DEFAULT_FRAME_INTERVAL

class VideoSource(ABC):
    """
    视频源抽象基类。
    定义了从不同来源获取视频并处理成标准输出（转录文本+关键帧）的接口。
    """

    @abstractmethod
    def acquire_video(self) -> Path:
        """
        获取视频文件并返回其本地路径。
        具体子类需实现此方法（如从 URL 下载或保存上传文件）。
        
        Returns:
            Path: 本地视频文件的绝对路径。
        """
        pass

    def process(self, extractor: Any, transcriber: Any) -> Tuple[str, List[str]]:
        """
        模板方法：执行标准的视频处理流程。
        1. 获取视频 (acquire_video)
        2. 提取音频和关键帧 (extractor)
        3. 转录音频 (transcriber)
        
        Args:
            extractor: 实现了 extract_audio 和 extract_frames 的对象。
            transcriber: 实现了 transcribe 的对象。
            
        Returns:
            Tuple[str, List[str]]: (转录文本, 关键帧列表)
        """
        # 1. 获取视频路径
        video_path = self.acquire_video()
        print(f"Video acquired at: {video_path}")

        # 2. 提取内容
        print("Extracting audio...")
        audio_path = extractor.extract_audio(video_path)
        print(f"Audio extracted to: {audio_path}")

        print("Extracting frames...")
        frames = extractor.extract_frames(video_path, interval=DEFAULT_FRAME_INTERVAL)
        print(f"Extracted {len(frames)} frames.")

        # 3. 生成转录
        print("Transcribing audio...")
        transcript = transcriber.transcribe(audio_path)
        print(f"Transcription complete. Length: {len(transcript)}")

        return transcript, frames
