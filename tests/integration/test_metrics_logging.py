import unittest
from unittest.mock import patch

from core.workflow.api import analyze_video


class _FakeWorkflowApp:
    def stream(self, initial_state, config, stream_mode="updates"):
        yield {"chunk_planner_node": {"chunk_plan": [{"chunk_id": "chunk-001"}]}}
        yield {
            "chunk_aggregator_node": {"aggregated_chunk_insights": "mock aggregated"},
        }
        yield {
            "human_gate_node": {
                "human_gate_status": "pending",
                "human_edited_aggregated_insights": "mock aggregated",
            }
        }


class TestMetricsLogging(unittest.TestCase):
    def test_metrics_events_are_emitted_when_enabled(self):
        with patch("core.workflow.api.create_checkpointer", return_value=object()):
            with patch("core.workflow.api.build_video_summary_graph", return_value=_FakeWorkflowApp()):
                with patch("core.workflow.api.ENABLE_METRICS_LOGGING", True):
                    with patch("core.workflow.api.METRICS_SAMPLE_RATE", 1.0):
                        with patch("core.workflow.api.log_metric_event") as mock_metric:
                            result = analyze_video(
                                transcript='{"segments": [{"start": 0, "end": 1, "text": "hello"}]}',
                                keyframes=[{"time": "00:00", "image": "x"}],
                                thread_id="thread-metrics",
                            )

        self.assertEqual(result.get("stage"), "pending_human_review")
        self.assertEqual(result.get("aggregated_chunk_insights"), "mock aggregated")
        event_names = [call.args[1] for call in mock_metric.call_args_list]
        self.assertIn("workflow_started", event_names)
        self.assertIn("workflow_node_update", event_names)
        self.assertIn("workflow_finished", event_names)


if __name__ == "__main__":
    unittest.main()

