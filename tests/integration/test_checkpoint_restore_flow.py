import os
import unittest
from unittest.mock import patch

from core.workflow.api import analyze_video, answer_question_at_timestamp


class _SharedFakeCheckpointer:
    def __init__(self):
        self._storage = {}

    def get(self, config):
        thread_id = config.get("configurable", {}).get("thread_id")
        return self._storage.get(thread_id)

    def put_channel_values(self, thread_id, channel_values):
        self._storage[thread_id] = {"channel_values": channel_values}


class _FakeWorkflowApp:
    def __init__(self, checkpointer, summary_text="mock draft summary"):
        self._checkpointer = checkpointer
        self._summary_text = summary_text

    def stream(self, initial_state, config, stream_mode="updates"):
        thread_id = config.get("configurable", {}).get("thread_id")
        final_state = dict(initial_state)
        final_state.update(
            {
                "draft_summary": self._summary_text,
                "revision_count": 1,
            }
        )
        self._checkpointer.put_channel_values(thread_id, final_state)
        yield {
            "fusion_drafter_node": {
                "draft_summary": self._summary_text,
                "revision_count": 1,
            }
        }


class TestCheckpointRestoreFlow(unittest.TestCase):
    def test_analyze_video_persists_state_for_time_travel(self):
        """阶段4：同一 thread_id 首次总结后，应能被时间旅行接口读取并回答。"""
        checkpointer = _SharedFakeCheckpointer()
        transcript = '{"segments": [{"start": 9, "end": 13, "text": "persisted evidence"}]}'
        keyframes = [{"time": "00:10", "image": ""}]

        with patch("core.workflow.api.create_checkpointer", return_value=checkpointer):
            with patch(
                "core.workflow.api.build_video_summary_graph",
                return_value=_FakeWorkflowApp(checkpointer, summary_text="first summary"),
            ):
                review_package = analyze_video(
                    transcript=transcript,
                    keyframes=keyframes,
                    user_prompt="请关注关键细节",
                    thread_id="thread-persist",
                )

            with patch.dict(os.environ, {}, clear=True):
                answer = answer_question_at_timestamp(
                    thread_id="thread-persist",
                    timestamp="00:10",
                    question="这里说了什么？",
                )

        self.assertEqual(review_package.get("stage"), "pending_human_review")
        self.assertEqual(review_package.get("thread_id"), "thread-persist")
        self.assertIn("persisted evidence", answer)
        self.assertIn("目标时间戳: 00:10", answer)

    def test_same_thread_id_uses_latest_checkpoint_state(self):
        """阶段4：同一 thread_id 二次写入后，时间旅行应读取最新状态而不是旧状态。"""
        checkpointer = _SharedFakeCheckpointer()
        old_transcript = '{"segments": [{"start": 5, "end": 6, "text": "old evidence"}]}'
        new_transcript = '{"segments": [{"start": 15, "end": 18, "text": "new evidence"}]}'

        with patch("core.workflow.api.create_checkpointer", return_value=checkpointer):
            with patch(
                "core.workflow.api.build_video_summary_graph",
                return_value=_FakeWorkflowApp(checkpointer, summary_text="first summary"),
            ):
                analyze_video(
                    transcript=old_transcript,
                    keyframes=[{"time": "00:05", "image": ""}],
                    user_prompt="old prompt",
                    thread_id="thread-latest",
                )

            with patch(
                "core.workflow.api.build_video_summary_graph",
                return_value=_FakeWorkflowApp(checkpointer, summary_text="second summary"),
            ):
                analyze_video(
                    transcript=new_transcript,
                    keyframes=[{"time": "00:16", "image": ""}],
                    user_prompt="new prompt",
                    thread_id="thread-latest",
                )

            with patch.dict(os.environ, {}, clear=True):
                answer = answer_question_at_timestamp(
                    thread_id="thread-latest",
                    timestamp="00:16",
                    question="最新这一段在讲什么？",
                )

        self.assertIn("new evidence", answer)
        self.assertNotIn("old evidence", answer)


if __name__ == "__main__":
    unittest.main()