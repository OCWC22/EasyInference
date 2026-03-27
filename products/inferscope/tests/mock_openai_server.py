"""Lightweight mock OpenAI-compatible server for end-to-end benchmark testing.

Run with: python tests/mock_openai_server.py [--port 8199]
Responds to /v1/models and /v1/chat/completions with streaming SSE.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import sys

WORDS = [
    "The", "code", "has", "several", "issues", "for", "production", "use",
    "including", "connection", "pooling", "error", "handling", "and",
    "backpressure", "mechanisms", "that", "should", "be", "addressed",
    "before", "deploying", "to", "a", "high-traffic", "environment",
    "where", "reliability", "is", "critical", "for", "serving",
    "inference", "requests", "at", "scale", "with", "low", "latency",
]


async def handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        raw = b""
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            raw += line
            if line == b"\r\n" or line == b"\n" or not line:
                break

        request_line = raw.split(b"\r\n")[0].decode(errors="replace")
        method, path, *_ = request_line.split(" ", 2)

        # Read body if Content-Length present
        headers_text = raw.decode(errors="replace")
        content_length = 0
        for hdr_line in headers_text.split("\r\n"):
            if hdr_line.lower().startswith("content-length:"):
                content_length = int(hdr_line.split(":", 1)[1].strip())
        body = b""
        if content_length > 0:
            body = await asyncio.wait_for(reader.readexactly(content_length), timeout=10.0)

        if path == "/v1/models" and method == "GET":
            resp_body = json.dumps({
                "object": "list",
                "data": [{"id": "mock-model", "object": "model", "owned_by": "mock"}],
            })
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp_body)}\r\n\r\n{resp_body}".encode()
            )
            await writer.drain()

        elif path == "/v1/chat/completions" and method == "POST":
            payload = json.loads(body) if body else {}
            max_tokens = min(payload.get("max_tokens", 64), 128)
            model = payload.get("model", "mock-model")
            stream = payload.get("stream", False)

            if not stream:
                # Non-streaming response
                text = " ".join(random.choices(WORDS, k=max_tokens))
                prompt_tokens = sum(len(m.get("content", "")) // 4 for m in payload.get("messages", []))
                resp_body = json.dumps({
                    "id": f"chatcmpl-mock-{random.randint(1000,9999)}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens, "total_tokens": prompt_tokens + max_tokens},
                })
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(resp_body)}\r\n\r\n{resp_body}".encode()
                )
                await writer.drain()
            else:
                # Streaming SSE response
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n"
                )
                await writer.drain()

                prompt_tokens = sum(len(m.get("content", "")) // 4 for m in payload.get("messages", []))
                tokens_generated = 0

                # Simulate TTFT delay (5-20ms)
                await asyncio.sleep(random.uniform(0.005, 0.020))

                for i in range(max_tokens):
                    word = random.choice(WORDS)
                    chunk = {
                        "id": f"chatcmpl-mock-{random.randint(1000,9999)}",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                    }
                    data = f"data: {json.dumps(chunk)}\n\n"
                    chunk_hex = f"{len(data):x}\r\n{data}\r\n"
                    writer.write(chunk_hex.encode())
                    tokens_generated += 1
                    # Simulate inter-token latency (1-5ms)
                    await asyncio.sleep(random.uniform(0.001, 0.005))

                # Final chunk with usage
                final = {
                    "id": f"chatcmpl-mock-{random.randint(1000,9999)}",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens_generated, "total_tokens": prompt_tokens + tokens_generated},
                }
                data = f"data: {json.dumps(final)}\n\n"
                chunk_hex = f"{len(data):x}\r\n{data}\r\n"
                writer.write(chunk_hex.encode())

                done_data = "data: [DONE]\n\n"
                done_hex = f"{len(done_data):x}\r\n{done_data}\r\n"
                writer.write(done_hex.encode())

                # Terminate chunked encoding
                writer.write(b"0\r\n\r\n")
                await writer.drain()

        elif path == "/health" and method == "GET":
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot found")
            await writer.drain()

    except Exception as e:
        print(f"  [error] {e}", flush=True)
        try:
            writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 5\r\n\r\nerror")
            await writer.drain()
        except Exception:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def main(port: int = 8199) -> None:
    server = await asyncio.start_server(handle_request, "127.0.0.1", port)
    print(f"Mock OpenAI server listening on http://127.0.0.1:{port}", flush=True)
    print(f"  GET  /v1/models           -> model list", flush=True)
    print(f"  POST /v1/chat/completions -> streaming SSE", flush=True)
    print(f"  GET  /health              -> ok", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    port = 8199
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    asyncio.run(main(port))
