from core.workflow.video_summary.state import VideoSummaryState

def consistency_checker_node(state: VideoSummaryState) -> dict:
    """
    幻觉与一致性审查员。核对草稿中的内容：是否有“文本里没提到且画面里也没有”的幻觉？
    是否遗漏了重要的 PPT 画面解析？如果不合格，输出具体的修改点至 critique。
    
    :param state: VideoSummaryState
    :return: dict 包含更新的 critique 字段
    """
    # 存根：架构占位符
    return {"critique": ""}