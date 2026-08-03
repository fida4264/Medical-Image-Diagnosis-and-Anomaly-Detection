"""
tests/test_api.py — Module 21: Production REST API Unit & Integration Tests
"""

from __future__ import annotations

import io
import os
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import settings


class TestRESTAPI(unittest.TestCase):
    """Test suite covering all Module 21 FastAPI REST API endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        # Disable authentication for functional endpoint tests
        settings.AUTH_REQUIRED = False

        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.img_bytes = cls._create_synthetic_image_bytes()

    @staticmethod
    def _create_synthetic_image_bytes() -> bytes:
        img = Image.new("RGB", (128, 128), color=(30, 40, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_01_health_endpoint(self) -> None:
        """Verify GET /health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("api_version", data)
        self.assertIn("engine_state", data)
        self.assertIn("database_status", data)
        self.assertEqual(data["api_version"], "0.21.0.0")

    def test_02_system_info_endpoint(self) -> None:
        """Verify GET /system/info endpoint."""
        response = self.client.get("/system/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("platform_name", data)
        self.assertEqual(len(data["supported_modalities"]), 7)
        self.assertEqual(len(data["pipeline_stages"]), 8)

    def test_03_benchmark_endpoint(self) -> None:
        """Verify GET /benchmark endpoint."""
        response = self.client.get("/benchmark")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("overall_accuracy", data)
        self.assertIn("overall_auc", data)
        self.assertGreaterEqual(len(data["models"]), 1)

    def test_04_predict_single_endpoint(self) -> None:
        """Verify POST /predict single image upload."""
        files = {"file": ("test_scan.png", self.img_bytes, "image/png")}
        data = {"modality": "chest_xray", "patient_id": "PT-TEST-101"}
        response = self.client.post("/predict", files=files, data=data)
        self.assertEqual(response.status_code, 200)

        res = response.json()
        self.assertIn("request_id", res)
        self.assertEqual(res["patient_id"], "PT-TEST-101")
        self.assertEqual(res["modality"], "chest_xray")
        self.assertIn("predicted_class", res)
        self.assertIn("confidence_score", res)
        self.assertIn("reliability", res)
        self.assertIn("risk_level", res)
        self.assertIn("gradcam", res)
        self.assertIn("anomaly", res)
        self.assertIn("timings", res)

        # Store prediction ID for history & report tests
        TestRESTAPI.test_prediction_id = res["request_id"]

    def test_05_predict_invalid_image(self) -> None:
        """Verify POST /predict with invalid image content fails with 400."""
        files = {"file": ("bad_file.png", b"this is not an image", "image/png")}
        response = self.client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error_code"], "INVALID_IMAGE")
        self.assertIn("request_id", data)

    def test_06_predict_unsupported_modality(self) -> None:
        """Verify POST /predict with unsupported modality fails with 422."""
        files = {"file": ("test_scan.png", self.img_bytes, "image/png")}
        data = {"modality": "invalid_modality_name"}
        response = self.client.post("/predict", files=files, data=data)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error_code"], "UNSUPPORTED_MODALITY")

    def test_07_batch_predict_endpoint(self) -> None:
        """Verify POST /batch-predict multiple images upload."""
        files = [
            ("files", ("scan_1.png", self.img_bytes, "image/png")),
            ("files", ("scan_2.png", self.img_bytes, "image/png")),
        ]
        data = {"modality": "chest_xray"}
        response = self.client.post("/batch-predict", files=files, data=data)
        self.assertEqual(response.status_code, 200)

        res = response.json()
        self.assertEqual(res["total_images"], 2)
        self.assertEqual(res["successful"], 2)
        self.assertEqual(res["failed"], 0)
        self.assertEqual(len(res["results"]), 2)

    def test_08_history_list_endpoint(self) -> None:
        """Verify GET /history endpoint."""
        response = self.client.get("/history?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("records", data)
        self.assertIn("total_records", data)

    def test_09_history_single_endpoint(self) -> None:
        """Verify GET /history/{id} endpoint."""
        pred_id = getattr(TestRESTAPI, "test_prediction_id", "req_demo_001")
        response = self.client.get(f"/history/{pred_id}")
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["prediction_id"], pred_id)
        else:
            self.assertEqual(response.status_code, 200)  # Should always be 200 after test_04

    def test_10_history_not_found(self) -> None:
        """Verify GET /history/{id} with missing ID returns 404."""
        response = self.client.get("/history/non_existent_prediction_id_99999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["error_code"], "PREDICTION_NOT_FOUND")
        self.assertIn("request_id", data)

    def test_11_report_metadata_endpoint(self) -> None:
        """Verify GET /reports/{id} endpoint."""
        pred_id = getattr(TestRESTAPI, "test_prediction_id", "req_demo_001")
        response = self.client.get(f"/reports/{pred_id}")
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["prediction_id"], pred_id)
            self.assertIn("download_urls", data)
        else:
            self.assertEqual(response.status_code, 200)

    def test_12_report_download_json(self) -> None:
        """Verify GET /reports/{id}/download?format=json endpoint."""
        pred_id = getattr(TestRESTAPI, "test_prediction_id", "req_demo_001")
        response = self.client.get(f"/reports/{pred_id}/download?format=json")
        if response.status_code == 200:
            self.assertIn("application/json", response.headers["content-type"])
        else:
            self.assertEqual(response.status_code, 200)

    def test_13_report_download_markdown(self) -> None:
        """Verify GET /reports/{id}/download?format=md endpoint."""
        pred_id = getattr(TestRESTAPI, "test_prediction_id", "req_demo_001")
        response = self.client.get(f"/reports/{pred_id}/download?format=md")
        if response.status_code == 200:
            self.assertIn("text/markdown", response.headers["content-type"])
        else:
            self.assertEqual(response.status_code, 200)

    def test_14_api_key_authentication(self) -> None:
        """Verify API key security with multi-key constant-time verification."""
        orig_req = settings.AUTH_REQUIRED
        orig_keys = settings.API_KEYS_RAW
        settings.AUTH_REQUIRED = True
        settings.API_KEYS_RAW = "secret_test_key_123"

        try:
            # Request without key -> 401
            resp_no_key = self.client.get("/health")
            self.assertEqual(resp_no_key.status_code, 401)
            self.assertEqual(resp_no_key.json()["error_code"], "UNAUTHORIZED")

            # Request with invalid key -> 401
            resp_bad_key = self.client.get("/health", headers={"X-API-Key": "wrong_key"})
            self.assertEqual(resp_bad_key.status_code, 401)

            # Request with valid header key -> 200
            resp_valid_hdr = self.client.get("/health", headers={"X-API-Key": "secret_test_key_123"})
            self.assertEqual(resp_valid_hdr.status_code, 200)

            # Request with valid query param key -> 200
            resp_valid_qry = self.client.get("/health?api_key=secret_test_key_123")
            self.assertEqual(resp_valid_qry.status_code, 200)
        finally:
            settings.AUTH_REQUIRED = orig_req
            settings.API_KEYS_RAW = orig_keys

    def test_15_openapi_schema(self) -> None:
        """Verify GET /openapi.json returns valid OpenAPI specification."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertIn("openapi", schema)
        self.assertIn("paths", schema)
        self.assertIn("/predict", schema["paths"])
        self.assertIn("/batch-predict", schema["paths"])
        self.assertIn("/history", schema["paths"])
        self.assertIn("/reports/{prediction_id}", schema["paths"])
        self.assertIn("/benchmark", schema["paths"])
        self.assertIn("/health", schema["paths"])


if __name__ == "__main__":
    unittest.main()
