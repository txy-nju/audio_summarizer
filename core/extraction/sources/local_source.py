
from pathlib import Path
from typing import IO
from core.extraction.base import VideoSource
from core.extraction.video.local_video_handler import LocalVideoHandler

class LocalFileVideoSource(VideoSource):
    """
    针对本地上传视频源的实现。
    使用 LocalVideoHandler 保存上传的文件。
    """
    def __init__(self, uploaded_file: IO[bytes], original_filename: str):
        self.uploaded_file = uploaded_file
        self.original_filename = original_filename
        self.handler = LocalVideoHandler()

    def acquire_video(self) -> Path:
        """
        保存上传的文件到临时目录。
        """
        print(f"Saving uploaded file {self.original_filename}...")
        return self.handler.save_uploaded_file(self.uploaded_file, self.original_filename)
