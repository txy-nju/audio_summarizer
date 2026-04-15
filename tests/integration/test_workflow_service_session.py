import unittest
from unittest.mock import patch

from services.workflow_service import VideoSummaryService


class TestWorkflowServiceSession(unittest.TestCase):
    def test_ask_at_timestamp_reuses_last_thread_id(self):
        """阶段4：服务层时间旅行追问应默认复用最近一次会话 thread_id。"""
        service = VideoSummaryService(api_key="fake-key")
        service.last_thread_id = "thread-from-summary"

        with patch("services.workflow_service.answer_question_at_timestamp", return_value="ok") as mock_answer:
            result = service.ask_at_timestamp(
                timestamp="00:10",
                question="这里讲了什么？",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(mock_answer.call_args.kwargs["thread_id"], "thread-from-summary")

    def test_finalize_summary_reuses_last_thread_id(self):
        """两阶段改造后，finalize_summary 在未显式传参时应复用 last_thread_id。"""
        service = VideoSummaryService(api_key="fake-key")
        service.last_thread_id = "thread-from-phase1"

        with patch(
            "services.workflow_service._finalize_summary_api",
            return_value="final-summary",
        ) as mock_finalize:
            result = service.finalize_summary(
                thread_id="",
                edited_aggregated_chunk_insights="edited",
                human_guidance="guide",
            )

        self.assertEqual(result, "final-summary")
        self.assertEqual(mock_finalize.call_args.kwargs["thread_id"], "thread-from-phase1")
        self.assertEqual(mock_finalize.call_args.kwargs["edited_aggregated_chunk_insights"], "edited")
        self.assertEqual(mock_finalize.call_args.kwargs["human_guidance"], "guide")


if __name__ == "__main__":
    unittest.main()