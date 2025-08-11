# Debugging and Client Log Ingestion

This document describes how to send client-side logs to the backend for centralized logging.

## POST /logs

- Endpoint: `/logs`
- Header: `X-Log-Token: YOUR_TOKEN` (configure `system_config.logging_token` in `conf.yaml`)
- Rate limiting: 10 requests/second per IP
- Body: arbitrary JSON payload, recommended fields:
  - `component`: e.g., `frontend`, `web`, `twitch_bot`
  - `level`: `info|warn|error`
  - `message`: short message
  - `details`: optional structured data
  - `request_id`: optional request correlation id

### Example

```bash
curl -X POST http://localhost:8000/logs \
  -H "Content-Type: application/json" \
  -H "X-Log-Token: YOUR_TOKEN" \
  -d '{
    "component": "frontend",
    "level": "error",
    "message": "Unhandled exception",
    "details": {"stack": "..."},
    "request_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

Notes:
- Secrets in keys like `token*`, `*key*`, `secret*` are masked before being written to logs.
- All server logs are written to `logs/app_{date}.jsonl` as structured JSON. 