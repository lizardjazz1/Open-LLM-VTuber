#!/usr/bin/env python3
"""Analyze centralized JSONL log: print tail and key frontend metrics.

Usage:
  uv run python scripts/analyze_log.py path/to/app_*.jsonl
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from collections import deque


def main() -> int:
    if len(sys.argv) < 2:
        print("USAGE: scripts/analyze_log.py <logfile.jsonl>")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print("FILE_NOT_FOUND:", p)
        return 1

    try:
        lines: list[str] = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        # Fallback to manual reading to avoid memory spikes on huge files
        lines = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line.rstrip("\n"))

    tail = deque(lines, maxlen=80)
    print("--- tail 80 ---")
    for s in tail:
        print(s)

    text = "\n".join(lines)
    # Flexible patterns: allow whitespace around colon and nested placement
    pat_component_frontend = re.compile(r"\"component\"\s*:\s*\"frontend\"")
    pat_event_uiclick = re.compile(r"\"event\"\s*:\s*\"ui\\.click\"")
    pat_event_chatsend = re.compile(r"\"event\"\s*:\s*\"chat\\.send\"")
    pat_event_wsstatus = re.compile(r"\"event\"\s*:\s*\"ws\\.status\"")
    pat_window_error = re.compile(r"\"message\"\s*:\s*\"window-error\"")

    counts = {
        "frontend": len(pat_component_frontend.findall(text)),
        "ui.click": len(pat_event_uiclick.findall(text)),
        "chat.send": len(pat_event_chatsend.findall(text)),
        "ws.status": len(pat_event_wsstatus.findall(text)),
        # also include generic containment as fallback
        "window-error": len(pat_window_error.findall(text))
        or text.count("window-error"),
    }

    print("\n--- counts ---")
    for k, v in counts.items():
        print(f"{k}={v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
