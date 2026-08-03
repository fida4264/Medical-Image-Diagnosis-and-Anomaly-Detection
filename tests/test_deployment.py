"""
tests/test_deployment.py — Module 22: Deployment & Packaging Hardening Verification Tests
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestDeploymentPackaging(unittest.TestCase):
    """Test suite verifying Module 22 deployment configurations, scripts, and containers."""

    def test_01_dockerfile_exists_and_valid(self) -> None:
        """Verify production Dockerfile structure and OCI labels."""
        dockerfile_path = ROOT_DIR / "docker" / "Dockerfile"
        self.assertTrue(dockerfile_path.exists(), "docker/Dockerfile must exist")

        content = dockerfile_path.read_text(encoding="utf-8")
        self.assertIn("FROM --platform=${TARGETPLATFORM} python:3.11-slim AS builder", content)
        self.assertIn("FROM --platform=${TARGETPLATFORM} python:3.11-slim AS runtime", content)
        self.assertIn("USER medai", content)
        self.assertIn("EXPOSE 8000 7860", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("ENTRYPOINT", content)

        # OCI Labels verification
        self.assertIn("org.opencontainers.image.title", content)
        self.assertIn("org.opencontainers.image.version", content)
        self.assertIn("org.opencontainers.image.authors", content)
        self.assertIn("org.opencontainers.image.licenses", content)

    def test_02_docker_dev_and_ignore_exist(self) -> None:
        """Verify Dockerfile.dev and .dockerignore exist."""
        dev_dockerfile = ROOT_DIR / "docker" / "Dockerfile.dev"
        dockerignore = ROOT_DIR / "docker" / ".dockerignore"

        self.assertTrue(dev_dockerfile.exists(), "docker/Dockerfile.dev must exist")
        self.assertTrue(dockerignore.exists(), "docker/.dockerignore must exist")

        ignore_content = dockerignore.read_text(encoding="utf-8")
        self.assertIn("__pycache__", ignore_content)
        self.assertIn(".venv", ignore_content)
        self.assertIn(".env", ignore_content)

    def test_03_docker_compose_valid_yaml_and_hardened(self) -> None:
        """Verify docker-compose.yml structure, profiles, security options, and limits."""
        prod_compose = ROOT_DIR / "docker" / "docker-compose.yml"
        dev_compose = ROOT_DIR / "docker" / "docker-compose.dev.yml"

        self.assertTrue(prod_compose.exists())
        self.assertTrue(dev_compose.exists())

        with open(prod_compose, "r", encoding="utf-8") as f:
            prod_data = yaml.safe_load(f)
            self.assertIn("services", prod_data)
            self.assertIn("medical-ai-platform", prod_data["services"])

            service = prod_data["services"]["medical-ai-platform"]
            self.assertIn("profiles", service)
            self.assertTrue(service["read_only"])
            self.assertIn("cap_drop", service)
            self.assertEqual(service["cap_drop"], ["ALL"])
            self.assertIn("security_opt", service)
            self.assertIn("deploy", service)
            self.assertIn("resources", service["deploy"])
            self.assertIn("logging", service)

        with open(dev_compose, "r", encoding="utf-8") as f:
            dev_data = yaml.safe_load(f)
            self.assertIn("services", dev_data)

    def test_04_deployment_scripts_exist(self) -> None:
        """Verify deployment shell and batch scripts exist."""
        scripts = [
            ROOT_DIR / "deployment" / "startup.sh",
            ROOT_DIR / "deployment" / "run.sh",
            ROOT_DIR / "deployment" / "run.bat",
            ROOT_DIR / "deployment" / "healthcheck.sh",
            ROOT_DIR / "deployment" / "wait_for_services.sh",
            ROOT_DIR / "run.bat",
            ROOT_DIR / "run.sh",
        ]
        for script in scripts:
            self.assertTrue(script.exists(), f"Script {script.name} must exist")

    def test_05_makefile_and_config_exist(self) -> None:
        """Verify Makefile, configs/deployment.yaml, and Kubernetes template exist."""
        makefile = ROOT_DIR / "Makefile"
        config_file = ROOT_DIR / "configs" / "deployment.yaml"
        k8s_file = ROOT_DIR / "configs" / "k8s_deployment_template.yaml"

        self.assertTrue(makefile.exists(), "Makefile must exist")
        self.assertTrue(config_file.exists(), "configs/deployment.yaml must exist")
        self.assertTrue(k8s_file.exists(), "configs/k8s_deployment_template.yaml must exist")

        make_content = makefile.read_text(encoding="utf-8")
        for target in ["install:", "run:", "api:", "ui:", "docker-build:", "docker-run:", "docker-stop:", "test:"]:
            self.assertIn(target, make_content, f"Makefile must contain {target}")

        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            self.assertIn("deployment", cfg)
            self.assertEqual(cfg["deployment"]["version"], "0.22.0.0")

    def test_06_documentation_files_exist(self) -> None:
        """Verify Module 22 documentation files exist."""
        docs = [
            ROOT_DIR / "INSTALL.md",
            ROOT_DIR / "QUICK_START.md",
            ROOT_DIR / "docs" / "module22_docker.md",
        ]
        for doc in docs:
            self.assertTrue(doc.exists(), f"Document {doc.name} must exist")


if __name__ == "__main__":
    unittest.main()
