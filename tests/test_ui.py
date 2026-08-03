"""
test_ui.py — Module 20: Web Application Unit Tests
"""

import unittest
from ui.ui_configuration import SUPPORTED_MODALITIES, MODALITY_KEY_MAP
from ui.backend_bridge import get_db_statistics, get_benchmark_summary, engine_health, run_prediction, export_report
from ui.theme import MEDICAL_CSS, get_theme
from app import create_app


class TestWebApplication(unittest.TestCase):
    """Test suite for Module 20 Web Application."""

    def test_ui_configuration(self):
        self.assertIn("Brain MRI", SUPPORTED_MODALITIES)
        self.assertIn("Chest X-Ray", SUPPORTED_MODALITIES)
        self.assertEqual(MODALITY_KEY_MAP["Chest X-Ray"], "chest_xray")

    def test_theme_and_css(self):
        self.assertTrue(len(MEDICAL_CSS) > 100)
        theme = get_theme()
        self.assertIsNotNone(theme)

    def test_backend_bridge(self):
        health = engine_health()
        self.assertIn("status", health)

        db_stats = get_db_statistics()
        self.assertIn("total_predictions", db_stats)

        bm = get_benchmark_summary()
        self.assertIn("overall_accuracy", bm)

    def test_prediction_bridge_invalid_image(self):
        success, res, err = run_prediction("non_existent_image.png")
        self.assertFalse(success)
        self.assertIn("No valid image", err)

    def test_report_export(self):
        sample_report = {
            "case_id": "test-case-001",
            "patient_id": "PT-999",
            "modality": "chest_xray",
            "predicted_disease": "Pneumonia",
            "confidence": "95.0%",
        }
        ok, fpath, err = export_report(sample_report, "json")
        self.assertTrue(ok)
        self.assertTrue(len(fpath) > 0)

    def test_app_creation(self):
        app = create_app()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
