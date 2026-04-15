import unittest
from unittest.mock import patch

from core.workflow.api import analyze_video, finalize_summary


class _SharedFakeCheckpointer:
    def __init__(self):
        self._storage = {}

    def get(self, config):
        thread_id = config.get("configurable", {}).get("thread_id")
        return self._storage.get(thread_id)

    def put_channel_values(self, thread_id, channel_values):
        self._storage[thread_id] = {"channel_values": channel_values}


class _FakePhase1WorkflowApp:
    def __init__(self, checkpointer):
        self._checkpointer = checkpointer

    def stream(self, initial_state, config, stream_mode="updates"):
        thread_id = config.get("configurable", {}).get("thread_id")
        final_state = dict(initial_state)
        final_state.update(
            {
                "aggregated_chunk_insights": "phase1 aggregated",
                "human_gate_status": "pending",
                "human_edited_aggregated_insights": "phase1 aggregated",
                "human_gate_reason": "human_review_required",
            }
        )
        self._checkpointer.put_channel_values(thread_id, final_state)
        yield {
            "chunk_aggregator_node": {
                "aggregated_chunk_insights": "phase1 aggregated",
            }
        }
        yield {
            "human_gate_node": {
                "human_gate_status": "pending",
                "human_edited_aggregated_insights": "phase1 aggregated",
                "human_gate_reason": "human_review_required",
            }
        }


class _FakePhase2WorkflowApp:
    def __init__(self):
        self.last_initial_state = None

    def stream(self, initial_state, config, stream_mode="updates"):
        self.last_initial_state = dict(initial_state)
        yield {
            "fusion_drafter_node": {
                "draft_summary": "final summary from phase2",
                "revision_count": 1,
            }
        }


class TestHumanReviewFinalizeFlow(unittest.TestCase):
    def test_summarize_returns_pending_review_package(self):
        checkpointer = _SharedFakeCheckpointer()

        with patch("core.workflow.api.create_checkpointer", return_value=checkpointer):
            with patch(
                "core.workflow.api.build_video_summary_graph",
                return_value=_FakePhase1WorkflowApp(checkpointer),
            ):
                result = analyze_video(
                    transcript='{"segments": []}',
                    keyframes=[{"time": "00:00", "image": "x"}],
                    thread_id="thread-hitl",
                )

        self.assertEqual(result.get("stage"), "pending_human_review")
        self.assertEqual(result.get("thread_id"), "thread-hitl")
        self.assertEqual(result.get("aggregated_chunk_insights"), "phase1 aggregated")
        self.assertEqual(result.get("editable_aggregated_chunk_insights"), "phase1 aggregated")

    def test_finalize_uses_human_inputs_and_returns_summary(self):
        checkpointer = _SharedFakeCheckpointer()
        checkpointer.put_channel_values(
            "thread-hitl",
            {
                "transcript": '{"segments": []}',
                "keyframes": [{"time": "00:00", "image": "x"}],
                "aggregated_chunk_insights": "old aggregated",
                "user_prompt": "focus",
            },
        )
        fake_phase2_app = _FakePhase2WorkflowApp()

        with patch("core.workflow.api.create_checkpointer", return_value=checkpointer):
            with patch("core.workflow.api.build_finalization_graph", return_value=fake_phase2_app):
                result = finalize_summary(
                    thread_id="thread-hitl",
                    edited_aggregated_chunk_insights="edited aggregated",
                    human_guidance="please keep it concise",
                )

        self.assertEqual(result, "final summary from phase2")
        self.assertIsNotNone(fake_phase2_app.last_initial_state)
        self.assertEqual(
            fake_phase2_app.last_initial_state.get("human_edited_aggregated_insights"),
            "edited aggregated",
        )
        self.assertEqual(
            fake_phase2_app.last_initial_state.get("human_guidance"),
            "please keep it concise",
        )
        self.assertEqual(fake_phase2_app.last_initial_state.get("human_gate_status"), "approved")


if __name__ == "__main__":
    unittest.main()


