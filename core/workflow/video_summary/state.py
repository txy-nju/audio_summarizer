from typing import TypedDict, List, Dict, Any, Annotated

def _merge_chunk_results(base: List[Dict], update: List[Dict]) -> List[Dict]:
    """
    深度合并两个并行分支的 chunk_results，确保来自不同节点的更新不会相互覆盖。
    
    策略：
    1. 按 chunk_id 构建索引，使用 base 中的原始顺序
    2. 对 latency_ms (dict) 进行递归合并（保留所有时间点记录）
    3. 其他字段默认进行覆盖更新（新值优先，但不删除现有字段）
    4. 保持 chunk_results 列表的与 chunk_plan 一致的顺序
    
    Args:
        base: 第一个分支返回的 chunk_results（如 chunk_audio_analyzer_node）
        update: 第二个分支返回的 chunk_results（如 chunk_vision_analyzer_node）
    
    Returns:
        已合并的 chunk_results，包含来自两个分支的所有更新
    """
    if not base:
        return update if isinstance(update, list) else []
    if not update:
        return base if isinstance(base, list) else []
    
    def _extract_chunk_id(item: Dict[str, Any]) -> str:
        """从字典项中安全提取 chunk_id，返回空字符串表示无效"""
        chunk_id = item.get("chunk_id")
        # 避免 None 被转为字符串 "None"
        if chunk_id is None or not isinstance(chunk_id, str):
            return ""
        return chunk_id.strip()
    
    # 构建结果映射（chunk_id → merged_item）
    result_map: Dict[str, Dict[str, Any]] = {}
    
    # 阶段 1: 加入 base 中的所有项
    for item in base:
        if isinstance(item, dict):
            chunk_id = _extract_chunk_id(item)
            if chunk_id:
                result_map[chunk_id] = dict(item)  # 浅拷贝
    
    # 阶段 2: 合并 update 中的项
    for item in update:
        if isinstance(item, dict):
            chunk_id = _extract_chunk_id(item)
            if chunk_id:
                if chunk_id in result_map:
                    # 深度合并策略
                    existing = result_map[chunk_id]
                    for key, val in item.items():
                        if key == "latency_ms":
                            # latency_ms 是 dict，需要递归合并而不是覆盖
                            if isinstance(existing.get(key), dict) and isinstance(val, dict):
                                existing[key].update(val)
                            else:
                                existing[key] = val
                        else:
                            # 其他字段：只有当不存在时才添加，存在则保留（基于 base 优先）
                            # 但如果是空字符串或初始值，则更新
                            if key not in existing or not existing[key]:
                                existing[key] = val
                else:
                    # 新的 chunk_id，直接加入
                    result_map[chunk_id] = dict(item)
    
    # 阶段 3: 按原始 base 顺序重新排序，并追加新增项
    ordered: List[Dict[str, Any]] = []
    seen: set = set()
    
    for base_item in base:
        if isinstance(base_item, dict):
            chunk_id = _extract_chunk_id(base_item)
            if chunk_id and chunk_id in result_map:
                ordered.append(result_map[chunk_id])
                seen.add(chunk_id)
    
    # 追加只在 update 中的新项（保持 update 的顺序）
    for update_item in update:
        if isinstance(update_item, dict):
            chunk_id = _extract_chunk_id(update_item)
            if chunk_id and chunk_id not in seen and chunk_id in result_map:
                ordered.append(result_map[chunk_id])
                seen.add(chunk_id)
    
    return ordered


class VideoSummaryState(TypedDict):
    # 输入层数据
    concurrency_mode: str           # 并发模式：threadpool / send_api
    transcript: str                 # 视频语音识别文本 (ASR/Whisper 输出)
    keyframes: List[Dict]           # 关键帧列表，包含 base64 数据及时间戳：[{"time": "00:15", "image": "base64_str"}]
    keyframes_base_path: str        # 关键帧文件引用模式下的根目录（用于 frame_file 解析）
    user_prompt: str                # 用户具体的总结侧重点
    
    # 中间态数据
    text_insights: str              # 语音文本的提炼结果
    visual_insights: str            # 关键帧画面的动作/图表解析结果

    # 5.3 Map-Reduce（迭代 A）中间态
    video_duration_seconds: int     # 推断出的视频总时长（秒）
    chunk_plan: List[Dict]          # 分片计划
    chunk_results: Annotated[List[Dict], _merge_chunk_results]  # 带 reducer 的分片结果，支持并行分支合并
    current_chunk: Dict             # 当前分片上下文（为 Send API 预留）
    current_chunk_base_item: Dict   # 当前分片已有结果（Send API worker 合并基座）
    chunk_audio_insights: Dict      # 分片音频洞察映射（可选中间态）
    chunk_visual_insights: Dict     # 分片视觉洞察映射（可选中间态）
    chunk_retry_count: Dict         # 分片重试计数
    reduce_debug_info: Dict         # Reduce 阶段调试元信息
    
    # 输出与循环控制
    draft_summary: str              # 当前生成的融合总结草稿
    
    # [Self-RAG 架构升级新增字段]
    hallucination_score: str        # 幻觉评分器的裁定结果 (取值: "yes" 表示有幻觉, "no" 表示无幻觉)
    usefulness_score: str           # 有用性评分器的裁定结果 (取值: "yes" 表示有用, "no" 表示无用/偏题)
    feedback_instructions: str      # 精确的反馈修改指导，替代原先单一模糊的 critique

    revision_count: int             # 重写次数