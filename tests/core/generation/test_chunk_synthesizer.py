import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from core.workflow.video_summary.nodes.chunk_synthesizer import chunk_synthesizer_node
from core.workflow.video_summary.state import VideoSummaryState


class TestChunkSynthesizerNode(unittest.TestCase):
    def test_generates_chunk_summary_for_all_chunks_without_api_key(self):
        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000"}, {"chunk_id": "chunk-001"}],
                "chunk_results": [
                    {"chunk_id": "chunk-000", "audio_insights": "a0", "vision_insights": "v0"},
                    {"chunk_id": "chunk-001", "audio_insights": "a1", "vision_insights": "v1"},
                ],
                "user_prompt": "关注流程",
            },
        )

        with patch.dict("os.environ", {}, clear=True):
            result = chunk_synthesizer_node(state)

        self.assertIn("chunk_results", result)
        self.assertEqual(len(result["chunk_results"]), 2)
        self.assertEqual(
            result["chunk_results"][0]["chunk_summary"],
            "[chunk=chunk-000] 分片融合（降级）\\n- Audio: a0\\n- Vision: v0",
        )
        self.assertEqual(
            result["chunk_results"][1]["chunk_summary"],
            "[chunk=chunk-001] 分片融合（降级）\\n- Audio: a1\\n- Vision: v1",
        )
        self.assertIn("latency_ms", result["chunk_results"][0])
        self.assertIn("synthesizer", result["chunk_results"][0]["latency_ms"])

    def test_handles_empty_chunk_plan(self):
        state = cast(VideoSummaryState, {"chunk_plan": [], "chunk_results": [{"chunk_id": "x"}]})
        result = chunk_synthesizer_node(state)
        self.assertEqual(result["chunk_results"], [{"chunk_id": "x"}])

    @patch("core.workflow.video_summary.nodes.chunk_synthesizer.OpenAI")
    def test_generates_chunk_summary_with_successful_llm_response(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="主干融合摘要"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [{"chunk_id": "chunk-000"}],
                "chunk_results": [
                    {
                        "chunk_id": "chunk-000",
                        "audio_insights": "audio data",
                        "vision_insights": "vision data",
                        "latency_ms": {"audio": 1},
                    }
                ],
                "user_prompt": "关注流程",
            },
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = chunk_synthesizer_node(state)

        self.assertEqual(len(result["chunk_results"]), 1)
        self.assertEqual(result["chunk_results"][0]["chunk_summary"], "主干融合摘要")
        self.assertEqual(result["chunk_results"][0]["latency_ms"]["audio"], 1)
        self.assertIn("synthesizer", result["chunk_results"][0]["latency_ms"])
        mock_openai.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

    @patch("core.workflow.video_summary.nodes.chunk_synthesizer.as_completed")
    @patch("core.workflow.video_summary.nodes.chunk_synthesizer.ThreadPoolExecutor")
    @patch("core.workflow.video_summary.nodes.chunk_synthesizer._process_single_chunk_synthesis")
    def test_parallel_execution_reduces_runtime(self, mock_process, mock_executor_cls, mock_as_completed):
        mock_process.side_effect = lambda chunk_id, user_prompt, base_item: (
            chunk_id,
            {
                "chunk_id": chunk_id,
                "chunk_summary": f"fusion-{chunk_id}",
                "latency_ms": {"synthesizer": 1},
            },
        )

        class _FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        submitted_futures = []
        submitted_calls = []
        executor_instance = MagicMock()

        def _submit(fn, *args, **kwargs):
            submitted_calls.append((fn, args, kwargs))
            future = _FakeFuture(fn(*args, **kwargs))
            submitted_futures.append(future)
            return future

        executor_instance.submit.side_effect = _submit
        mock_executor_cls.return_value.__enter__.return_value = executor_instance
        mock_as_completed.side_effect = lambda futures: futures

        state = cast(
            VideoSummaryState,
            {
                "chunk_plan": [
                    {"chunk_id": "chunk-000"},
                    {"chunk_id": "chunk-001"},
                    {"chunk_id": "chunk-002"},
                    {"chunk_id": "chunk-003"},
                ],
                "chunk_results": [
                    {"chunk_id": "chunk-000", "audio_insights": "a0", "vision_insights": "v0"},
                    {"chunk_id": "chunk-001", "audio_insights": "a1", "vision_insights": "v1"},
                    {"chunk_id": "chunk-002", "audio_insights": "a2", "vision_insights": "v2"},
                    {"chunk_id": "chunk-003", "audio_insights": "a3", "vision_insights": "v3"},
                ],
                "user_prompt": "并行验证",
            },
        )

        with patch("core.workflow.video_summary.nodes.chunk_synthesizer.MAP_MAX_PARALLELISM", 2):
            result = chunk_synthesizer_node(state)

        self.assertEqual(len(result["chunk_results"]), 4)
        mock_executor_cls.assert_called_once_with(max_workers=2)
        self.assertEqual(executor_instance.submit.call_count, 4)
        self.assertEqual(len(submitted_calls), 4)
        self.assertEqual(len(submitted_futures), 4)
        mock_as_completed.assert_called_once()
        self.assertEqual(mock_process.call_count, 4)


if __name__ == "__main__":
    unittest.main()
