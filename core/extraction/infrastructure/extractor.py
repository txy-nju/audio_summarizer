import cv2
import base64
import math
from pathlib import Path
from typing import Optional
from moviepy.video.io.VideoFileClip import VideoFileClip
from config.settings import TEMP_AUDIO_DIR, MAX_IMAGE_SIZE

class MediaExtractor:
    def __init__(self, audio_dir: Path = TEMP_AUDIO_DIR):
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path: Path) -> Optional[Path]:
        """
        从视频中提取音频并返回音频文件路径。若视频无音轨，则优雅返回 None。
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.audio_dir / f"{video_path.stem}.mp3"
        
        try:
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    print(f"Warning: No audio track found in {video_path}")
                    return None
                video.audio.write_audiofile(str(audio_path), codec='mp3', logger=None)
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise IOError(f"Failed to extract audio from {video_path}: {e}")

    @staticmethod
    def _calc_histogram(image, compare_max_size: int = 320):
        """计算图像的归一化灰度直方图（用于场景比较，默认先降采样以提速）。"""
        h, w = image.shape[:2]
        if max(h, w) > compare_max_size:
            if h > w:
                new_h = compare_max_size
                new_w = int(w * (compare_max_size / h))
            else:
                new_w = compare_max_size
                new_h = int(h * (compare_max_size / w))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def extract_frames(
        self,
        video_path: Path,
        interval: int = 2,
        max_interval: int = 60,
        threshold: float = 0.90,
        probe_fps: int = 5,
    ) -> list[dict]:
        """
        基于场景检测（直方图对比）的智能关键帧提取，有效降低 Token 消耗并保留重要信息。
        
        :param video_path: 视频路径
        :param interval: 两次抽帧的最小时间间隔（秒），即防抖间隔 (兼容旧代码中的 interval)
        :param max_interval: 强制抽帧的最大时间间隔（秒），用于防断层兜底
        :param threshold: 直方图相关性阈值，低于该值则认为场景发生突变
        :param probe_fps: 探测频率（每秒参与场景判定的帧数）。值越小越快，但可能降低突变检测灵敏度
        :return: 符合 VideoSummaryState 契约的字典列表
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
            
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # 兼容 fps 为 0, NaN 或 None 的异常情况
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 30.0 

        frames = []
        frame_count = 0

        # 探测步长：逐帧读取，但仅按固定频率执行场景判定与编码，降低热路径计算开销
        safe_probe_fps = max(1, probe_fps)
        probe_stride = max(1, int(fps / safe_probe_fps))
        
        last_extracted_time = -1000.0  # 确保第0秒强制抽取
        last_hist = None
        
        while True:
            success, image = vidcap.read()
            if not success or image is None:
                # 视频流中断、损坏或自然结束，安全退出
                break

            # 跳过非探测帧，降低直方图比较与编码开销
            if probe_stride > 1 and frame_count % probe_stride != 0:
                frame_count += 1
                continue
                
            current_time = frame_count / fps
            should_extract = False
            
            # 条件 1：首帧强制抽取作为基准 Anchor
            current_hist = None

            if last_hist is None:
                should_extract = True
            else:
                time_since_last = current_time - last_extracted_time
                
                # 条件 2：是否达到最大间隔兜底（防止静止画面太长导致大模型时空断层）
                if time_since_last >= max_interval:
                    should_extract = True
                # 条件 3：距离上次抽帧超过防抖阈值，且画面发生剧变
                elif time_since_last >= interval:
                    current_hist = self._calc_histogram(image)
                    correlation = cv2.compareHist(last_hist, current_hist, cv2.HISTCMP_CORREL)
                    
                    if correlation < threshold:
                        should_extract = True

            if should_extract:
                # 缩放图片以控制 Base64 长度
                h, w, _ = image.shape
                if max(h, w) > MAX_IMAGE_SIZE:
                    if h > w:
                        new_h = MAX_IMAGE_SIZE
                        new_w = int(w * (MAX_IMAGE_SIZE / h))
                    else:
                        new_w = MAX_IMAGE_SIZE
                        new_h = int(h * (MAX_IMAGE_SIZE / w))
                    image = cv2.resize(image, (new_w, new_h))

                # 更新基准与时间
                # 复用已计算的 current_hist，避免重复计算
                if current_hist is None:
                    current_hist = self._calc_histogram(image)
                last_hist = current_hist
                last_extracted_time = current_time

                # 编码为 Base64
                _, buffer = cv2.imencode('.jpg', image)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # 计算时间戳并格式化为 HH:MM:SS 或 MM:SS
                # [防御性修复]: 避免浮点数精度丢失导致时间戳被截断 (如 3664.9999999 被 int() 截断为 3664)
                total_seconds = int(round(current_time))
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 0:
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = f"{minutes:02d}:{seconds:02d}"
                
                frames.append({
                    "time": time_str,
                    "image": jpg_as_text
                })
                
            frame_count += 1

        vidcap.release()
        return frames