"""Regression checks for revision changes, not a scientific FAB ablation."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services import memory_store as m


class MemoryRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(m, "MEMORY_PATH", Path(self.tmp.name) / "memory.md")
        self.path_patch.start()
        self.now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        self.time_patch = patch.object(m, "_utc_now", return_value=self.now)
        self.time_patch.start()

    def tearDown(self):
        self.time_patch.stop()
        self.path_patch.stop()
        self.tmp.cleanup()

    def test_filtered_full_read_keeps_unmatched_entries(self):
        m.memory_append("polyether workflow")
        m.memory_append("polyimide workflow")
        m.memory_read(mode="full", query="polyether")
        self.assertEqual(len(m.load_memory_document()[1]), 2)
        m.memory_read(mode="full", query="absent")
        self.assertEqual(len(m.load_memory_document()[1]), 2)

    def test_expired_memory_not_injected_without_reflect(self):
        m.memory_append("expired fact")
        with patch.object(m, "_utc_now", return_value=self.now + timedelta(days=100)):
            self.assertNotIn("expired fact", m.build_inject_summary())

    def test_reinforcement_uses_decayed_strength_and_resets_clock(self):
        eid = m.memory_append("polyether")['id']
        later = self.now + timedelta(days=8)
        expected = m._decay_factor_for_interval(8) + 0.08
        with patch.object(m, "_utc_now", return_value=later):
            m.memory_update(eid, "updated polymer evidence")
            entry = m.load_memory_document()[1][0]
            self.assertAlmostEqual(m.effective_strength_now(entry), expected, places=3)
            first = m.apply_time_decay([entry], later)[0]
            second = m.apply_time_decay([first], later)[0]
            self.assertEqual(first.strength, second.strength)

    def test_large_time_step_compresses_directly_to_minimal(self):
        e = m.MemoryEntry('test', self.now, self.now, 1.0, 'full', body='x' * 500,
                          last_decay_applied=self.now)
        result = m.apply_time_decay([e], self.now + timedelta(days=13))
        self.assertEqual(result[0].layer, 'minimal')


if __name__ == '__main__':
    unittest.main()
