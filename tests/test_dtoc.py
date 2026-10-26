"""Unit tests for Harpe Dynamic Tool Output Compression (DTOC)."""

import unittest
from harpe import CompressedToolOutput, DynamicToolCompressor


class TestDynamicToolCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = DynamicToolCompressor(min_compress_chars=200, preview_lines=3)

    def test_short_output_bypasses_compression(self):
        short_text = "Exit code: 0\nAll tests passed."
        placeholder, artifact = self.compressor.maybe_compress("test_runner", short_text)
        self.assertEqual(placeholder, short_text)
        self.assertIsNone(artifact)

    def test_verbose_output_is_compressed_and_reversible(self):
        verbose_lines = [f"Step {i}: processing data item {i*17} with status ok" for i in range(50)]
        verbose_text = "\n".join(verbose_lines)
        self.assertTrue(len(verbose_text) > 500)

        placeholder, artifact = self.compressor.maybe_compress("batch_job", verbose_text)
        self.assertIsNotNone(artifact)
        assert artifact is not None

        # Verify placeholder contains ref, line count, summary
        self.assertIn("TOOL_OUTPUT_COMPRESSED", placeholder)
        self.assertIn(artifact.ref_id, placeholder)
        self.assertIn("lines=50", placeholder)
        self.assertTrue(len(placeholder) < len(verbose_text))

        # Test reversible restoration
        restored = self.compressor.restore(artifact.ref_id)
        self.assertEqual(restored, verbose_text)

    def test_restore_unknown_ref_returns_none(self):
        self.assertIsNone(self.compressor.restore("non_existent_ref"))

    def test_single_line_verbose_output_bounded_by_max_preview_chars(self):
        long_line = '{"records": [' + ", ".join(f'{{"id": {i}, "val": "item_{i}"}}' for i in range(500)) + "]}"
        self.assertTrue(len(long_line) > 5000)

        compressor = DynamicToolCompressor(min_compress_chars=200, preview_lines=8, max_preview_chars=600)
        placeholder, artifact = compressor.maybe_compress("fetch_data", long_line)
        self.assertIsNotNone(artifact)
        assert artifact is not None

        # Summary should be bounded to around 600 chars + ellipsis
        self.assertTrue(len(artifact.summary) <= 610)
        self.assertTrue(len(placeholder) < len(long_line))
        self.assertTrue(len(placeholder) < 800)

        # Restoration should still return the full raw payload
        self.assertEqual(compressor.restore(artifact.ref_id), long_line)

    def test_invalid_compressor_parameters(self):
        with self.assertRaises(ValueError):
            DynamicToolCompressor(min_compress_chars=0)
        with self.assertRaises(ValueError):
            DynamicToolCompressor(preview_lines=0)
        with self.assertRaises(ValueError):
            DynamicToolCompressor(max_preview_chars=0)


if __name__ == "__main__":
    unittest.main()
