"""Convert real Claude Code + Codex session transcripts to SessionTraceV0 format.

Usage:
    python -m workloads.convert_real_traces --source claude --limit 201 --output workloads/synthetic_v0/traces/real/
    python -m workloads.convert_real_traces --source codex --limit 201 --output workloads/synthetic_v0/traces/real/
    python -m workloads.convert_real_traces --source both --limit 201 --output workloads/synthetic_v0/traces/real/
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from uuid import uuid4


# ---------------------------------------------------------------------------
# Claude Code converter
# ---------------------------------------------------------------------------

def convert_claude_session(jsonl_path: Path) -> dict | None:
    """Convert a Claude Code .jsonl session to SessionTraceV0 dict."""
    lines = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if len(lines) < 4:
        return None

    turns = []
    cumulative_tokens = 0
    turn_id = 0
    session_id = jsonl_path.stem
    goal = ""
    contains_tools = False
    contains_rag = False
    contains_branching = False
    first_timestamp = ""
    last_timestamp = ""
    total_input = 0
    total_output = 0

    for msg in lines:
        msg_type = msg.get("type", "")

        if msg_type == "user":
            turn_id += 1
            content = msg.get("message", {}).get("content", "")
            if isinstance(content, list):
                text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            else:
                text = str(content)

            if turn_id == 1:
                goal = text[:200]

            tokens_est = len(text.split()) * 2
            cumulative_tokens += tokens_est
            total_input += tokens_est

            ts = msg.get("timestamp", "")
            if not first_timestamp:
                first_timestamp = ts
            last_timestamp = ts

            turns.append({
                "turn_id": turn_id,
                "parent_turn_id": turn_id - 1 if turn_id > 1 else None,
                "role": "user",
                "content_blocks": [{"type": "text", "text": text[:5000]}],
                "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                "input_tokens": tokens_est,
                "output_tokens": 0,
                "cumulative_tokens": cumulative_tokens,
            })

        elif msg_type == "assistant":
            turn_id += 1
            content_msg = msg.get("message", {})
            usage = content_msg.get("usage", {})
            in_tok = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)

            if in_tok == 0 and out_tok == 0:
                # Estimate from content
                content_blocks_raw = content_msg.get("content", [])
                text = ""
                for block in (content_blocks_raw if isinstance(content_blocks_raw, list) else []):
                    if isinstance(block, dict):
                        text += block.get("text", "") + block.get("thinking", "")
                in_tok = len(text.split())
                out_tok = in_tok

            cumulative_tokens += in_tok + out_tok
            total_input += in_tok
            total_output += out_tok

            # Extract content blocks
            content_blocks = []
            tool_calls = []
            content_raw = content_msg.get("content", [])
            for block in (content_raw if isinstance(content_raw, list) else []):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    content_blocks.append({"type": "text", "text": block.get("text", "")[:5000]})
                elif block_type == "tool_use":
                    contains_tools = True
                    tool_name = block.get("name", "unknown")
                    tool_input = json.dumps(block.get("input", {}))[:1000]
                    content_blocks.append({
                        "type": "code" if tool_name in ("Edit", "Write", "Read") else "log",
                        "text": f"[{tool_name}] {tool_input}",
                        "metadata": {"tool_name": tool_name},
                    })
                    tool_calls.append(tool_name)
                elif block_type == "thinking":
                    pass  # Skip thinking blocks

            if not content_blocks:
                content_blocks = [{"type": "text", "text": "(assistant response)"}]

            ts = msg.get("timestamp", "")
            if not first_timestamp:
                first_timestamp = ts
            last_timestamp = ts

            role = "assistant"
            if tool_calls:
                # If this turn is mostly tool calls, mark as tool
                if len(tool_calls) > len([b for b in content_blocks if b["type"] == "text"]):
                    role = "tool"

            turns.append({
                "turn_id": turn_id,
                "parent_turn_id": turn_id - 1,
                "role": role,
                "content_blocks": content_blocks[:10],  # Cap blocks
                "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cumulative_tokens": cumulative_tokens,
            })

        elif msg_type == "tool_result":
            turn_id += 1
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(str(c) for c in content)[:3000]
            else:
                text = str(content)[:3000]
            contains_tools = True

            tokens_est = len(text.split()) * 2
            cumulative_tokens += tokens_est
            total_input += tokens_est

            ts = msg.get("timestamp", "")
            last_timestamp = ts

            turns.append({
                "turn_id": turn_id,
                "parent_turn_id": turn_id - 1,
                "role": "execution",
                "content_blocks": [{"type": "log", "text": text}],
                "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                "input_tokens": tokens_est,
                "output_tokens": 0,
                "cumulative_tokens": cumulative_tokens,
            })

    if len(turns) < 4:
        return None

    # Classify family
    tool_turns = sum(1 for t in turns if t["role"] in ("tool", "execution"))
    tool_ratio = tool_turns / len(turns) if turns else 0
    if tool_ratio > 0.2:
        family = "coding"
    else:
        family = "long_chat"

    # Compute KV estimate (approximate: 2 bytes per token per layer, assume 80 layers GQA)
    kv_bytes_est = cumulative_tokens * 2 * 80

    # Compute idle windows from timestamps
    idle_windows = [0]
    for i in range(1, len(turns)):
        try:
            t1 = turns[i - 1].get("timestamp", "")
            t2 = turns[i].get("timestamp", "")
            if t1 and t2:
                from datetime import datetime
                dt1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
                dt2 = datetime.fromisoformat(t2.replace("Z", "+00:00"))
                delta_ms = int((dt2 - dt1).total_seconds() * 1000)
                idle_windows.append(max(0, delta_ms))
            else:
                idle_windows.append(0)
        except (ValueError, TypeError):
            idle_windows.append(0)

    max_idle = max(idle_windows) if idle_windows else 0
    cache_expectation = "reactivated" if max_idle > 60000 else "warm"

    # Compute prefix overlap (ratio of shared context between consecutive turns)
    if len(turns) >= 2:
        ctx_values = [t["cumulative_tokens"] for t in turns if t["cumulative_tokens"] > 0]
        if len(ctx_values) >= 2:
            new_per_turn = sum(ctx_values[i] - ctx_values[i - 1] for i in range(1, len(ctx_values))) / (len(ctx_values) - 1)
            avg_ctx = sum(ctx_values) / len(ctx_values)
            prefix_overlap = 1.0 - (new_per_turn / avg_ctx) if avg_ctx > 0 else 0.0
        else:
            prefix_overlap = 0.5
    else:
        prefix_overlap = 0.5

    return {
        "trace_id": f"real_claude_{session_id[:12]}",
        "dataset_version": "0.1.0",
        "source_type": "real",
        "workload_family": family,
        "language": "en",
        "tokenizer": {"name": "cl100k_base", "version": "1.0"},
        "session": {
            "session_id": session_id,
            "goal": goal,
            "contains_tools": contains_tools,
            "contains_rag": contains_rag,
            "contains_branching": contains_branching,
            "contains_multimodal": False,
        },
        "turns": turns,
        "replay": {
            "prefix_group_id": f"grp_claude_{session_id[:8]}",
            "prefix_overlap_ratio": round(min(max(prefix_overlap, 0.0), 1.0), 3),
            "reuse_distance_turns": 2,
            "reuse_distance_seconds": max_idle // 1000 if max_idle > 0 else 0,
            "estimated_kv_bytes_peak": kv_bytes_est,
            "branch_count": 0,
            "idle_windows_ms": idle_windows[:20],
            "arrival_pattern": "reactivation_heavy" if max_idle > 60000 else "steady_interactive",
        },
        "kv_pressure": {
            "cold_start": True,
            "warm_cache_hit_rate": prefix_overlap * 0.8,
            "estimated_offload_bytes": kv_bytes_est // 2 if cumulative_tokens > 50000 else 0,
            "estimated_reload_bytes": kv_bytes_est // 4 if max_idle > 60000 else 0,
            "tier": "CPU" if cumulative_tokens > 100000 else "HBM",
            "cache_expectation": cache_expectation,
            "eviction_risk_score": min(cumulative_tokens / 200000, 0.95),
        },
        "provenance": {
            "generator_model": "claude-code-real",
            "generator_prompt_version": "real-transcript-v1",
            "pipeline_version": "0.1.0",
        },
    }


# ---------------------------------------------------------------------------
# Codex converter
# ---------------------------------------------------------------------------

def convert_codex_session(jsonl_path: Path) -> dict | None:
    """Convert a Codex .jsonl session to SessionTraceV0 dict."""
    lines = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if len(lines) < 4:
        return None

    turns = []
    cumulative_tokens = 0
    turn_id = 0
    session_id = ""
    goal = ""
    contains_tools = False
    first_timestamp = ""
    last_timestamp = ""
    total_input = 0
    total_output = 0

    for msg in lines:
        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})
        ts = msg.get("timestamp", "")

        if not first_timestamp and ts:
            first_timestamp = ts
        if ts:
            last_timestamp = ts

        if msg_type == "session_meta":
            session_id = payload.get("id", uuid4().hex[:12])
            continue

        if msg_type == "response_item":
            item_type = payload.get("type", "")
            role = payload.get("role", "")

            if item_type == "message" and role == "user":
                turn_id += 1
                content = payload.get("content", [])
                text = ""
                for block in (content if isinstance(content, list) else []):
                    if isinstance(block, dict):
                        text += block.get("text", "")

                if turn_id == 1 or (not goal and text):
                    goal = text[:200]

                tokens_est = len(text.split()) * 2
                cumulative_tokens += tokens_est
                total_input += tokens_est

                turns.append({
                    "turn_id": turn_id,
                    "parent_turn_id": turn_id - 1 if turn_id > 1 else None,
                    "role": "user",
                    "content_blocks": [{"type": "text", "text": text[:5000]}],
                    "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                    "input_tokens": tokens_est,
                    "output_tokens": 0,
                    "cumulative_tokens": cumulative_tokens,
                })

            elif item_type == "message" and role == "assistant":
                turn_id += 1
                content = payload.get("content", [])
                text = ""
                for block in (content if isinstance(content, list) else []):
                    if isinstance(block, dict):
                        text += block.get("text", "")

                tokens_est = len(text.split()) * 2
                cumulative_tokens += tokens_est
                total_output += tokens_est

                turns.append({
                    "turn_id": turn_id,
                    "parent_turn_id": turn_id - 1,
                    "role": "assistant",
                    "content_blocks": [{"type": "text", "text": text[:5000]}],
                    "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                    "input_tokens": 0,
                    "output_tokens": tokens_est,
                    "cumulative_tokens": cumulative_tokens,
                })

            elif item_type == "function_call":
                turn_id += 1
                contains_tools = True
                func_name = payload.get("name", "unknown")
                args = payload.get("arguments", "")[:1000]

                tokens_est = len(args.split()) * 2
                cumulative_tokens += tokens_est
                total_output += tokens_est

                turns.append({
                    "turn_id": turn_id,
                    "parent_turn_id": turn_id - 1,
                    "role": "tool",
                    "content_blocks": [{
                        "type": "code" if func_name in ("write_file", "apply_patch") else "log",
                        "text": f"[{func_name}] {args}",
                        "metadata": {"tool_name": func_name},
                    }],
                    "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                    "input_tokens": 0,
                    "output_tokens": tokens_est,
                    "cumulative_tokens": cumulative_tokens,
                })

            elif item_type == "function_call_output":
                turn_id += 1
                output = payload.get("output", "")[:3000]

                tokens_est = len(output.split()) * 2
                cumulative_tokens += tokens_est
                total_input += tokens_est

                turns.append({
                    "turn_id": turn_id,
                    "parent_turn_id": turn_id - 1,
                    "role": "execution",
                    "content_blocks": [{"type": "log", "text": output}],
                    "timestamp": ts or f"2026-01-01T00:00:{turn_id:02d}Z",
                    "input_tokens": tokens_est,
                    "output_tokens": 0,
                    "cumulative_tokens": cumulative_tokens,
                })

        elif msg_type == "event_msg":
            evt_type = payload.get("type", "")
            if evt_type == "token_count":
                info = payload.get("info", {})
                if info:
                    usage = info.get("total_token_usage", {})
                    if usage:
                        real_input = usage.get("input_tokens", 0)
                        real_output = usage.get("output_tokens", 0)
                        if real_input > total_input:
                            total_input = real_input
                        if real_output > total_output:
                            total_output = real_output

    if len(turns) < 4:
        return None

    if not session_id:
        session_id = jsonl_path.stem

    # Classify family
    tool_turns = sum(1 for t in turns if t["role"] in ("tool", "execution"))
    tool_ratio = tool_turns / len(turns) if turns else 0
    family = "coding" if tool_ratio > 0.2 else "long_chat"

    kv_bytes_est = cumulative_tokens * 2 * 80

    # Idle windows
    idle_windows = [0]
    for i in range(1, min(len(turns), 20)):
        try:
            t1 = turns[i - 1].get("timestamp", "")
            t2 = turns[i].get("timestamp", "")
            if t1 and t2:
                from datetime import datetime
                dt1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
                dt2 = datetime.fromisoformat(t2.replace("Z", "+00:00"))
                delta_ms = int((dt2 - dt1).total_seconds() * 1000)
                idle_windows.append(max(0, delta_ms))
            else:
                idle_windows.append(0)
        except (ValueError, TypeError):
            idle_windows.append(0)

    max_idle = max(idle_windows) if idle_windows else 0
    cache_expectation = "reactivated" if max_idle > 60000 else "warm"

    prefix_overlap = 0.7  # Default for codex sessions

    return {
        "trace_id": f"real_codex_{session_id[:12]}",
        "dataset_version": "0.1.0",
        "source_type": "real",
        "workload_family": family,
        "language": "en",
        "tokenizer": {"name": "o200k_base", "version": "1.0"},
        "session": {
            "session_id": session_id,
            "goal": goal,
            "contains_tools": contains_tools,
            "contains_rag": False,
            "contains_branching": False,
            "contains_multimodal": False,
        },
        "turns": turns,
        "replay": {
            "prefix_group_id": f"grp_codex_{session_id[:8]}",
            "prefix_overlap_ratio": round(prefix_overlap, 3),
            "reuse_distance_turns": 2,
            "reuse_distance_seconds": max_idle // 1000 if max_idle > 0 else 0,
            "estimated_kv_bytes_peak": kv_bytes_est,
            "branch_count": 0,
            "idle_windows_ms": idle_windows[:20],
            "arrival_pattern": "reactivation_heavy" if max_idle > 60000 else "steady_interactive",
        },
        "kv_pressure": {
            "cold_start": True,
            "warm_cache_hit_rate": prefix_overlap * 0.8,
            "estimated_offload_bytes": kv_bytes_est // 2 if cumulative_tokens > 50000 else 0,
            "estimated_reload_bytes": kv_bytes_est // 4 if max_idle > 60000 else 0,
            "tier": "CPU" if cumulative_tokens > 100000 else "HBM",
            "cache_expectation": cache_expectation,
            "eviction_risk_score": min(cumulative_tokens / 200000, 0.95),
        },
        "provenance": {
            "generator_model": "codex-real",
            "generator_prompt_version": "real-transcript-v1",
            "pipeline_version": "0.1.0",
        },
    }


# ---------------------------------------------------------------------------
# Bin classifier
# ---------------------------------------------------------------------------

def classify_bin(trace: dict) -> str:
    """Classify trace into context bin based on cumulative tokens."""
    total = trace["turns"][-1]["cumulative_tokens"] if trace["turns"] else 0
    family = trace["workload_family"]

    if family == "long_chat":
        if total < 16000:
            return "lc1"
        elif total < 64000:
            return "lc2"
        else:
            return "lc3"
    else:  # coding, agent
        if total < 32000:
            return "ca1"
        elif total < 96000:
            return "ca2"
        else:
            return "ca3"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_conversion(source: str, limit: int, output_dir: Path) -> None:
    """Convert real sessions to SessionTraceV0."""

    sessions_to_convert = []

    if source in ("claude", "both"):
        claude_sessions = sorted(
            Path.home().glob(".claude/projects/*/*.jsonl"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        sessions_to_convert.extend([("claude", p) for p in claude_sessions[:limit]])
        print(f"Found {len(claude_sessions)} Claude Code sessions, converting top {min(limit, len(claude_sessions))} by size")

    if source in ("codex", "both"):
        codex_sessions = sorted(
            Path.home().glob(".codex/sessions/**/*.jsonl"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        sessions_to_convert.extend([("codex", p) for p in codex_sessions[:limit]])
        print(f"Found {len(codex_sessions)} Codex sessions, converting top {min(limit, len(codex_sessions))} by size")

    print(f"\nConverting {len(sessions_to_convert)} sessions...\n")

    converted = 0
    failed = 0
    skipped = 0

    for src, path in sessions_to_convert:
        try:
            if src == "claude":
                trace = convert_claude_session(path)
            else:
                trace = convert_codex_session(path)

            if trace is None:
                skipped += 1
                continue

            # Classify and save
            family = trace["workload_family"]
            bin_name = classify_bin(trace)
            family_dir = "long_chat" if family == "long_chat" else "coding"

            out_dir = output_dir / family_dir / bin_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"{trace['trace_id']}.json"
            with open(out_file, "w") as f:
                json.dump(trace, f, indent=2)

            total_tok = trace["turns"][-1]["cumulative_tokens"] if trace["turns"] else 0
            print(f"  {trace['trace_id']}: {len(trace['turns'])} turns, {total_tok:,} tok, {family}/{bin_name}")
            converted += 1

        except Exception as e:
            failed += 1

    print(f"\nDone: {converted} converted, {skipped} skipped (too short), {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Convert real Claude Code / Codex sessions to SessionTraceV0")
    parser.add_argument("--source", choices=["claude", "codex", "both"], default="both")
    parser.add_argument("--limit", type=int, default=201)
    parser.add_argument("--output", type=Path, default=Path("workloads/synthetic_v0/traces/real"))
    args = parser.parse_args()

    run_conversion(args.source, args.limit, args.output)


if __name__ == "__main__":
    main()
