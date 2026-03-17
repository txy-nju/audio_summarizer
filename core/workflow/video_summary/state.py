from typing import TypedDict, List, Dict

class VideoSummaryState(TypedDict):
    # 输入层数据
    transcript: str                 # 视频语音识别文本 (ASR/Whisper 输出)
    keyframes: List[Dict]           # 关键帧列表，包含 base64 数据及时间戳：[{"time": "00:15", "image": "base64_str"}]
    user_prompt: str                # 用户具体的总结侧重点（如：侧重产品演示、侧重理论讲解）
    
    # 中间态数据
    text_insights: str              # 语音文本的提炼结果
    visual_insights: str            # 关键帧画面的动作/图表解析结果
    
    # 输出与循环控制
    draft_summary: str              # 当前生成的融合总结草稿
    critique: str                   # 审阅节点给出的修改意见（如：图文不符、遗漏关键画面）
    revision_count: int             # 重写次数