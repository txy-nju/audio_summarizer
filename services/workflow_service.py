import os
from typing import IO, Callable, Optional

# 引入抽象层和具体策略
from core.extraction.base import VideoSource
from core.extraction.sources import UrlVideoSource, LocalFileVideoSource

# 【核心改造】引入新的工作流接口，不再需要旧的 analysis 和 generation
from core.workflow import summarize_video
from core.workflow.session import ensure_thread_id
from utils.file_utils import clear_temp_folder

class VideoSummaryService:
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.last_thread_id = ""
        # 优先使用前端传入的 base_url，如果为空则尝试读取环境变量，最后回退到官方默认地址
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # 【环境变量依赖注入】
        # 将前端传入的 API Key 和 Base URL 挂载到系统环境变量中。
        # 这是为了确保完全解耦的 LangGraph 内部节点（如 text_analyzer_node、vision_analyzer_node）
        # 可以在底层安全且纯粹地调用 os.getenv() 来初始化 OpenAI() 客户端，而无需破坏状态机签名。
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
            
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
            
        # 核心组件的初始化现在由 langgraph 内部管理

    def _process_source(
        self,
        source: VideoSource,
        user_prompt: str = "",
        status_callback: Optional[Callable[[str], None]] = None,
        thread_id: str = ""
    ) -> str:
        """
        统一的内部处理逻辑：
        1. 使用 VideoSource 获取内容 (Transcript + Frames)
        2. 调用新的工作流进行分析和总结
        """
        try:
            # 1. 获取内容 (VideoSource 现在是完全独立的)
            # [前端透传] 将来自 UI 的回调函数注入到底层的抽取生命周期中
            transcript, frames = source.process(status_callback=status_callback)

            # 2. 【核心改造】调用新的、符合架构的 summarize_video 函数
            if status_callback:
                status_callback("🔗 音频底模与关键帧视觉流预处理成功。多模态数据已向 AI 并发流水线集结。")
                
            print("Invoking AI workflow...")
            
            # 如果用户未输入提示，则使用架构默认的综合总结提示
            if not user_prompt or not user_prompt.strip():
                user_prompt = "请结合画面与语音，给出一个全面、客观的高质量视频总结。"
                
            # [前端透传] 将 UI 回调函数挂载进 LangGraph 执行总线上
            resolved_thread_id = ensure_thread_id(thread_id)
            self.last_thread_id = resolved_thread_id

            summary = summarize_video(
                transcript=transcript,
                keyframes=frames,
                user_prompt=user_prompt,
                status_callback=status_callback,
                thread_id=resolved_thread_id
            )
            
            if status_callback:
                status_callback("🎉 LangGraph 复杂的自反思闭环流转结束。一份完美的报告已送达交付点！")
                
            print("Workflow complete.")
            
            return summary
        finally:
            # 在结束后清理，确保不留垃圾文件
            clear_temp_folder()

    def process_video_from_url(
        self,
        url: str,
        user_prompt: str = "",
        status_callback: Optional[Callable[[str], None]] = None,
        thread_id: str = ""
    ) -> str:
        """
        针对 URL 的完整流程。
        """
        # [生命周期 Bugfix]：必须在实例化 Source (其底层处理器会在 init 时建文件夹) 之前，进行清空操作。
        # 否则 clear_temp_folder 会把刚建好的 temp/videos 删掉，导致 FileNotFoundError。
        clear_temp_folder()
        
        # 创建 Source 实例时传入必要的配置
        source = UrlVideoSource(url, api_key=self.api_key, base_url=self.base_url)
        return self._process_source(
            source,
            user_prompt=user_prompt,
            status_callback=status_callback,
            thread_id=thread_id
        )

    def process_uploaded_video(
        self,
        uploaded_file: IO[bytes],
        original_filename: str,
        user_prompt: str = "",
        status_callback: Optional[Callable[[str], None]] = None,
        thread_id: str = ""
    ) -> str:
        """
        针对上传文件的完整流程。
        """
        # [生命周期 Bugfix]：必须在实例化 Source 之前进行环境清理。
        clear_temp_folder()
        
        # 创建 Source 实例时传入必要的配置
        source = LocalFileVideoSource(
            uploaded_file, 
            original_filename, 
            api_key=self.api_key, 
            base_url=self.base_url
        )
        return self._process_source(
            source,
            user_prompt=user_prompt,
            status_callback=status_callback,
            thread_id=thread_id
        )