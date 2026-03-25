# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability in InferScope, please report it responsibly.

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please email: **security@inferscope.dev** (or open a private security advisory on GitHub).

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Security Model

InferScope operates in two modes with different trust boundaries:

### Read-only mode (default)
- All MCP tools and CLI commands are **read-only** by default
- No config changes are applied without explicit `--apply` flag
- No shell commands are executed
- No file system writes (except to `~/.inferscope/` cache directory)

### Network access
- Engine adapters (`/metrics`, `/v1/models`) make outbound HTTP requests to user-specified endpoints
- Endpoints are validated and restricted to HTTP/HTTPS schemes
- Private IP ranges are blocked by default to prevent SSRF
- The MCP server binds to `127.0.0.1` (localhost only) by default

### Input validation
- All user inputs (model names, GPU names, endpoints) are validated against known registries or sanitized
- URL endpoints are validated for scheme and host before any HTTP request
- No user input is passed to shell commands or `eval()`

## Known Limitations

- The HTTP transport (`--transport streamable-http`) exposes an HTTP server. Use a reverse proxy with authentication in production.
- GPU telemetry endpoints (DCGM port 9400, AMD DME port 5000) are assumed to be on a trusted network.
- The empirical profile store (future) may contain deployment fingerprints — treat it as sensitive.
