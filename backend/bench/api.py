"""Нагрузка на POST /islive_video. Запускать из backend/:

uv run --no-sync python -m bench.api --url http://localhost:8000 --requests 20 --concurrency 4 \
    data/k3f9a2p1/attempts
uv run --no-sync python -m bench.api --rps 0.1 0.25 0.5 --duration 60 data/k3f9a2p1/attempts

С --requests одновременно висит не больше --concurrency запросов. С --rps запросы приходят
с заданной частотой, сколько бы ни висело, — так видно, какой поток сервер выдерживает.
Видео передаются файлами или папками, на каждый запрос берётся случайное. Стимул берётся
из JSON рядом с видео.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import mimetypes
import random
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


def _stimulus(metadata: dict) -> dict | None:
    """Объект stim из JSON рядом с видео; None, если запись слишком старая для этого API."""
    if "stim" in metadata:
        return metadata["stim"]
    if "ctrl" in metadata:
        return {"type": "spline", "ctrl": metadata["ctrl"]}
    return None


def _recordings(paths: list[Path]) -> list[tuple[Path, bytes, dict]]:
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        else:
            files.append(path)
    recordings = []
    for file in files:
        sidecar = file.with_suffix(".json")
        stim = _stimulus(json.loads(sidecar.read_text())) if sidecar.is_file() else None
        if stim is not None:
            recordings.append((file, file.read_bytes(), stim))
    if not recordings:
        raise SystemExit("не нашлось видео с JSON, в котором есть stim или ctrl")
    return recordings


async def _one(client: httpx.AsyncClient, url: str, recording: tuple[Path, bytes, dict]) -> dict:
    path, video, stim = recording
    content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    row: dict = {"video": str(path)}
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/islive_video",
            data={"user_id": "bench", "stim": json.dumps(stim)},
            files={"file": (path.name, video, content_type)},
        )
        response.raise_for_status()
        body = response.json()
        row |= {"ok": True, "is_live": body["is_live"], "score": body["score"]}
    except httpx.HTTPStatusError as error:
        row |= {"ok": False, "error": f"HTTP {error.response.status_code}"}
    except Exception as error:
        row |= {"ok": False, "error": type(error).__name__}
    row["seconds"] = time.perf_counter() - start
    return row


async def _closed(client, url, recordings, requests: int, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def limited():
        async with semaphore:
            return await _one(client, url, random.choice(recordings))

    return await asyncio.gather(*(limited() for _ in range(requests)))


async def _open(client, url, recordings, rps: float, duration: float) -> list[dict]:
    start = time.perf_counter()
    tasks = []
    for i in range(math.ceil(rps * duration)):
        await asyncio.sleep(max(0.0, start + i / rps - time.perf_counter()))
        tasks.append(asyncio.create_task(_one(client, url, random.choice(recordings))))
    return await asyncio.gather(*tasks)


def _report(label: str, rows: list[dict], wall: float, show_scores: bool) -> None:
    ok = [row for row in rows if row["ok"]]
    line = f"{label}: ok={len(ok)}/{len(rows)} wall={wall:.1f}s throughput={len(ok) / wall:.2f} req/s"
    if ok:
        latencies = np.array([row["seconds"] for row in ok])
        line += (
            f" p50={np.percentile(latencies, 50):.2f}s p95={np.percentile(latencies, 95):.2f}s"
            f" max={latencies.max():.2f}s is_live={dict(Counter(row['is_live'] for row in ok))}"
        )
        if show_scores:
            line += f" score={sorted({round(row['score'], 4) for row in ok})}"
    errors = Counter(row["error"] for row in rows if not row["ok"])
    if errors:
        line += f" errors={dict(errors)}"
    print(line, flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path, help="видео или папки с видео")
    parser.add_argument("--url", default="http://localhost:8000")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--requests", type=int, default=10)
    mode.add_argument("--rps", type=float, nargs="+", help="одна или несколько частот подряд")
    parser.add_argument("--concurrency", type=int, default=1, help="только для --requests")
    parser.add_argument("--duration", type=float, default=60, help="секунд на каждую частоту --rps")
    parser.add_argument("--out", type=Path, help="записать каждый запрос строкой в JSONL")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    recordings = _recordings(args.videos)
    # Одно видео — скор должен совпадать во всех запросах, иначе что-то недетерминировано.
    show_scores = len(recordings) == 1
    print(f"videos={len(recordings)} url={url}/islive_video", flush=True)

    profiles = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as client:
        warmup = await _one(client, url, recordings[0])
        if not warmup["ok"]:
            raise SystemExit(f"прогревочный запрос не прошёл: {warmup['error']}")

        if args.rps:
            for rps in args.rps:
                start = time.perf_counter()
                rows = await _open(client, url, recordings, rps, args.duration)
                profiles.append((f"rps={rps:g}", rows, time.perf_counter() - start))
                _report(*profiles[-1], show_scores)
        else:
            start = time.perf_counter()
            rows = await _closed(client, url, recordings, args.requests, args.concurrency)
            profiles.append((f"concurrency={args.concurrency}", rows, time.perf_counter() - start))
            _report(*profiles[-1], show_scores)

    if args.out:
        with args.out.open("w") as file:
            for label, rows, _ in profiles:
                for row in rows:
                    file.write(json.dumps({"profile": label} | row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
