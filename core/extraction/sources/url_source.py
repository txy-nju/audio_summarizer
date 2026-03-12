
from pathlib import Path
from core.extraction.base import VideoSource
from core.extraction.video.downloader import VideoDownloader

class UrlVideoSource(VideoSource):
    """
    针对 URL 视频源的实现。
    使用 VideoDownloader 从网络下载视频。
    """
    def __init__(self, url: str):
        self.url = url
        self.downloader = VideoDownloader()

    def acquire_video(self) -> Path:
        """
        从 URL 下载视频。
        """
        print(f"Downloading video from {self.url}...")
        return self.downloader.download(self.url)
