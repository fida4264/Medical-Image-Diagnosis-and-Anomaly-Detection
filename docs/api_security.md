# Medical AI Platform — API Security Architecture Guide

## Overview

This document outlines the security architecture, threat model, and hardening controls implemented for the **Medical Image Diagnosis and Anomaly Detection Platform REST API** (Module 21).

---

## 1. Authentication & Key Management

### Multi-Key Constant-Time Verification
- Authentication is enforced via the `verify_api_key` dependency provider (`api/dependencies.py`).
- Key comparison uses **`secrets.compare_digest`** to guarantee constant-time evaluation across all configured valid keys, preventing timing side-channel attacks.
- Supports key rotation: multiple valid keys can be supplied as a comma-separated list in `API_KEYS`.

### Configuration
```bash
# Set in .env or environment variables
AUTH_REQUIRED=true
API_KEYS=med_key_clinician_99812,med_key_service_33412
```

### Client Request Header
```http
POST /predict HTTP/1.1
Host: api.medical-ai.internal
X-API-Key: med_key_clinician_99812
Content-Type: multipart/form-data
```

---

## 2. Security Headers & Network Protection

The `SecurityHeadersMiddleware` (`api/middleware.py`) applies production HTTP response headers:

| Header | Value | Purpose |
|---|---|---|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | Enforces HTTPS connections for 1 year. |
| `X-Content-Type-Options` | `nosniff` | Prevents browser MIME-type sniffing. |
| `X-Frame-Options` | `DENY` | Protects against Clickjacking attacks. |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Minimizes sensitive referrer leakage. |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` | Disables unneeded browser APIs. |
| `Content-Security-Policy` | `default-src 'self'; frame-ancestors 'none';` | Restricts resource loading and frame embedding. |
| `Server` | `MedicalAI-Engine/0.21.0` | Overwrites standard Uvicorn/Python server signatures. |

---

## 3. Rate Limiting & Denial of Service Protection

The `RateLimiterMiddleware` enforces a sliding 1-minute window rate limit per client IP:

- `POST /predict`, `POST /batch-predict`: **20 requests/minute**
- `GET /history`: **60 requests/minute**
- `GET /health`: **120 requests/minute**
- Default endpoints: **100 requests/minute**

Breaching limits returns an HTTP `429 Too Many Requests` response with a `Retry-After: 60` header.

---

## 4. File Upload Security & Sanitization

Before processing medical images, `_validate_and_sanitize_image_file()` executes a 4-point check:

1. **Extension Whitelist**: Only `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.dcm` allowed.
2. **File Size Limit**: Uploads exceeding 20MB (`MAX_UPLOAD_SIZE_BYTES`) immediately return `413 Payload Too Large`.
3. **MIME Type Validation**: Rejects executable binary MIME types (`application/x-msdownload`, `application/zip`, etc.).
4. **Header & Pixel Decoding**: PIL decodes image headers and pixel data to detect corrupted or polyglot files before passing to AI model adapters.

---

## 5. SQL Injection & Path Traversal Mitigation

- All database operations in `QueryEngine` and `PredictionRepository` use **parameterized SQLite queries (`?`)**.
- User input string concatenation in SQL queries is strictly prohibited.
- Temporary files are created using `tempfile.NamedTemporaryFile` and deleted immediately upon request completion.

---

## 6. Logging & Protected Observability

- Log lines mask API key query parameters (`api_key=***`).
- Medical image pixel arrays, raw files, and Patient Health Information (PHI) are never written to disk logs.
- Client responses mask raw Python stack traces during `ENVIRONMENT=production` mode.
- Request correlation is tracked via the `X-Request-ID` UUID header.
