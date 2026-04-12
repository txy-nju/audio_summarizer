
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import json
from config.settings import TRANSCRIBER_MODEL

class AudioTranscriber:
    def __init__(self, api_key: str, base_url: str = None, model: str = None):
        """
        初始化 AudioTranscriber。

        Args:
            api_key (str): OpenAI API Key。
            base_url (str, optional): OpenAI API 的中转地址。默认为 None。
            model (str, optional): 转文本模型名称。默认从 TRANSCRIBER_MODEL 读取。
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or TRANSCRIBER_MODEL
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def transcribe(self, audio_path: Path) -> str:
        """
        调用 Whisper API 将音频转录为 JSON 格式的文本。

        Args:
            audio_path (Path): 音频文件的路径。

        Returns:
            str: JSON 格式的转录结果（包含详细时间戳段落）。
        """
        print(f"Transcribing audio file: {audio_path}...")
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="verbose_json"  # 请求 verbose_json 以获取带时间戳的结构化数据
            )
        
        print("Transcription successful.")
        # 当 response_format 是 verbose_json 时，返回的是一个 TranscriptionVerbose 对象
        # 我们使用 model_dump_json() 将其转换为纯文本的 JSON 字符串
        return transcript.model_dump_json(indent=2)
