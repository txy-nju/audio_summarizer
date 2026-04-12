import uuid


def ensure_thread_id(thread_id: str = "") -> str:
    """
    标准化 thread_id。
    - 若传入有效字符串，返回去首尾空格后的值
    - 若为空，自动生成 UUID
    """
    if thread_id and thread_id.strip():
        return thread_id.strip()
    return str(uuid.uuid4())
