"""Нагрузка на POST /islive_video.

uv run --no-sync python bench_api.py --url http://localhost:8000 --requests 20 --concurrency 4 \
    data/f4frjppx5aof/video/261011_203155_875399.mp4
Стимул берётся из JSON рядом с видео.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np


async def _one(client: httpx.AsyncClient, url: str, video: bytes, stim: dict) -> tuple[float, dict]:
    data = {
        "user_id": "bench",
        # Неизвестный токен: сервер берёт stim_x и stim_y из формы.
        "token": "bench",
        "stim_x": [str(value) for value in stim["stim_x"]],
        "stim_y": [str(value) for value in stim["stim_y"]],
    }
    start = time.perf_counter()
    response = await client.post(
        f"{url}/islive_video", data=data, files={"file": ("video.mp4", video, "video/mp4")}
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return elapsed, response.json()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    video = args.video.read_bytes()
    stim = json.loads(args.video.with_suffix(".json").read_text())
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=600) as client:
        await _one(client, args.url, video, stim)

        async def limited():
            async with semaphore:
                return await _one(client, args.url, video, stim)

        start = time.perf_counter()
        results = await asyncio.gather(*(limited() for _ in range(args.requests)))
        wall = time.perf_counter() - start

    latencies = np.array([elapsed for elapsed, _ in results])
    scores = {round(result["score"], 4) for _, result in results}
    lives = {result["is_live"] for _, result in results}
    print(
        f"requests={args.requests} concurrency={args.concurrency} "
        f"wall={wall:.2f}s throughput={args.requests / wall:.2f} req/s "
        f"p50={np.percentile(latencies, 50):.2f}s p95={np.percentile(latencies, 95):.2f}s "
        f"max={latencies.max():.2f}s score={sorted(scores)} is_live={sorted(lives)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
