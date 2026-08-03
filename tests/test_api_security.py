"""
tests/test_api_security.py — Module 21: Production REST API Security & Hardening Test Suite
"""

from __future__ import annotations

import io
import os
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import settings


class TestAPISecurity(unittest.TestCase):
    """Security, reliability, and hardening test suite for Module 21 REST API."""

    @classmethod
    def setUpClass(cls) -> None:
        # Default settings for general tests
        settings.AUTH_REQUIRED = False
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.img_bytes = cls._create_synthetic_image_bytes()

    @staticmethod
    def _create_synthetic_image_bytes() -> bytes:
        img = Image.new("RGB", (64, 64), color=(50, 60, 70))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_01_security_headers_present(self) -> None:
        """Verify production security headers are set on outgoing HTTP responses."""
        response = self.client.get("/health/liveness")
        self.assertEqual(response.status_code, 200)

        headers = response.headers
        self.assertIn("Strict-Transport-Security", headers)
        self.assertIn("X-Content-Type-Options", headers)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("X-Frame-Options", headers)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("Referrer-Policy", headers)
        self.assertIn("Permissions-Policy", headers)
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("Server", headers)
        self.assertEqual(headers["Server"], "MedicalAI-Engine/0.21.0")

    def test_02_request_id_tracing_propagation(self) -> None:
        """Verify X-Request-ID header generation and propagation."""
        response = self.client.get("/health/liveness")
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-ID", response.headers)
        self.assertTrue(response.headers["X-Request-ID"].startswith("req_"))

        # Verify incoming custom X-Request-ID is preserved
        custom_id = "req_custom_trace_999"
        res2 = self.client.get("/health/liveness", headers={"X-Request-ID": custom_id})
        self.assertEqual(res2.headers["X-Request-ID"], custom_id)

    def test_03_liveness_and_readiness_probes(self) -> None:
        """Verify Kubernetes liveness and readiness probe endpoints."""
        live_res = self.client.get("/health/liveness")
        self.assertEqual(live_res.status_code, 200)
        self.assertEqual(live_res.json()["status"], "UP")

        ready_res = self.client.get("/health/readiness")
        self.assertEqual(ready_res.status_code, 200)
        self.assertEqual(ready_res.json()["status"], "READY")
        self.assertTrue(ready_res.json()["engine_ready"])
        self.assertTrue(ready_res.json()["database_ready"])

    def test_04_file_size_limit_rejection(self) -> None:
        """Verify files exceeding limit are rejected with 413 FILE_TOO_LARGE."""
        orig_max = settings.MAX_UPLOAD_SIZE_BYTES
        settings.MAX_UPLOAD_SIZE_BYTES = 100  # Set small 100-byte limit for test

        try:
            files = {
                "file": ("large_scan.png", self.img_bytes, "image/png"),
            }
            response = self.client.post("/predict", files=files)
            self.assertEqual(response.status_code, 413)
            data = response.json()
            self.assertEqual(data["error_code"], "FILE_TOO_LARGE")
            self.assertIn("request_id", data)
        finally:
            settings.MAX_UPLOAD_SIZE_BYTES = orig_max

    def test_05_invalid_mime_type_rejection(self) -> None:
        """Verify unallowed MIME types are rejected with 400 INVALID_IMAGE."""
        files = {
            "file": ("malicious.exe", b"executable_data_content", "application/x-msdownload"),
        }
        response = self.client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error_code"], "INVALID_IMAGE")

    def test_06_corrupt_file_header_rejection(self) -> None:
        """Verify corrupt byte sequence with image extension is rejected with 400."""
        files = {
            "file": ("corrupt.png", b"corrupt_header_bytes_1234567890", "image/png"),
        }
        response = self.client.post("/predict", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error_code"], "INVALID_IMAGE")

    def test_07_constant_time_multi_key_authentication(self) -> None:
        """Verify multi-key constant-time API key verification."""
        orig_req = settings.AUTH_REQUIRED
        orig_keys = settings.API_KEYS_RAW
        settings.AUTH_REQUIRED = True
        settings.API_KEYS_RAW = "key_alpha_123,key_beta_456"

        try:
            # Unauthorized missing key -> 401
            r1 = self.client.get("/health")
            self.assertEqual(r1.status_code, 401)
            self.assertEqual(r1.json()["error_code"], "UNAUTHORIZED")

            # Unauthorized bad key -> 401
            r2 = self.client.get("/health", headers={"X-API-Key": "invalid_key_xxx"})
            self.assertEqual(r2.status_code, 401)

            # Authorized Key 1 -> 200
            r3 = self.client.get("/health", headers={"X-API-Key": "key_alpha_123"})
            self.assertEqual(r3.status_code, 200)

            # Authorized Key 2 -> 200
            r4 = self.client.get("/health", headers={"X-API-Key": "key_beta_456"})
            self.assertEqual(r4.status_code, 200)
        finally:
            settings.AUTH_REQUIRED = orig_req
            settings.API_KEYS_RAW = orig_keys

    def test_08_rate_limiting_enforcement(self) -> None:
        """Verify in-memory rate limiting triggers 429 Too Many Requests."""
        orig_limit = settings.RATE_LIMIT_PREDICT_PER_MIN
        settings.RATE_LIMIT_PREDICT_PER_MIN = 3  # Set low limit of 3 req/min

        try:
            client = TestClient(create_app())
            files = {"file": ("scan.png", self.img_bytes, "image/png")}

            # First 3 requests -> 200 OK
            for i in range(3):
                res = client.post("/predict", files=files, data={"modality": "chest_xray"})
                self.assertEqual(res.status_code, 200)

            # 4th request -> 429 Rate Limit Exceeded
            res_exceeded = client.post("/predict", files=files, data={"modality": "chest_xray"})
            self.assertEqual(res_exceeded.status_code, 429)
            data = res_exceeded.json()
            self.assertEqual(data["error_code"], "RATE_LIMIT_EXCEEDED")
            self.assertIn("Retry-After", res_exceeded.headers)
            self.assertEqual(res_exceeded.headers["Retry-After"], "60")
        finally:
            settings.RATE_LIMIT_PREDICT_PER_MIN = orig_limit

    def test_09_sql_injection_resilience(self) -> None:
        """Verify history query parameterization resists SQL injection payloads."""
        sql_payload = "chest_xray' OR '1'='1"
        response = self.client.get(f"/history?modality={sql_payload}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("records", data)
        self.assertEqual(data["total_records"], 0)

    def test_10_docs_disabling_switch(self) -> None:
        """Verify Swagger UI docs can be disabled via settings.ENABLE_DOCS."""
        orig_docs = settings.ENABLE_DOCS
        settings.ENABLE_DOCS = False

        try:
            disabled_app = create_app()
            disabled_client = TestClient(disabled_app)

            res_docs = disabled_client.get("/docs")
            self.assertEqual(res_docs.status_code, 404)

            res_openapi = disabled_client.get("/openapi.json")
            self.assertEqual(res_openapi.status_code, 404)
        finally:
            settings.ENABLE_DOCS = orig_docs


if __name__ == "__main__":
    unittest.main()
