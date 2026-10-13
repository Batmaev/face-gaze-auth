"""Замер скорости траектории взгляда по видео.

uv run --no-sync python bench_gaze.py --device cuda --repeat 5 data/0_1.mp4
Делегат MediaPipe и fp16 задаются переменными MEDIAPIPE_DELEGATE и UNIGAZE_FP16.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# tqdm читает переменную при импорте.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import unigaze

from unigaze_helpers import trajectory


class _Timer:
    def __init__(self):
        self.seconds = 0.0

    def wrap(self, function, sync=None):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                if sync is not None:
                    sync()
                self.seconds += time.perf_counter() - start

        return timed


def _run(video: Path, model, args) -> tuple[dict, np.ndarray]:
    landmarks = _Timer()
    predict = _Timer()
    original_prepare = trajectory._prepare_frame
    original_predict = trajectory._predict
    sync = torch.cuda.synchronize if args.device == "cuda" else None
    trajectory._prepare_frame = landmarks.wrap(original_prepare)
    trajectory._predict = predict.wrap(original_predict, sync)
    try:
        start = time.perf_counter()
        gaze = trajectory.unigaze_gaze_from_video(
            video, model, batch_size=args.batch_size, target_fps=args.target_fps
        )
        total = time.perf_counter() - start
    finally:
        trajectory._prepare_frame = original_prepare
        trajectory._predict = original_predict
    frames = len(gaze.xy)
    return {
        "total_s": total,
        "prepare_s": landmarks.seconds,
        "predict_s": predict.seconds,
        "other_s": total - landmarks.seconds - predict.seconds,
        "frames": frames,
        "faces": int(np.isfinite(gaze.xy[:, 0]).sum()),
    }, gaze.xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-fps", type=float, default=15)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--save", type=Path, help="сохранить xy для сравнения")
    parser.add_argument("--compare", type=Path, help="сравнить xy с сохранёнными")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    start = time.perf_counter()
    model = unigaze.load("unigaze_b16_joint", device=args.device)
    load_s = time.perf_counter() - start
    print(
        f"device={args.device} fp16={args.device == 'cuda' and trajectory._use_fp16()} batch={args.batch_size} "
        f"mediapipe={trajectory._mediapipe_delegate()} torch={torch.__version__} "
        f"threads={torch.get_num_threads()} load={load_s:.2f}s"
    )

    saved = {}
    for video in args.videos:
        # Первый прогон прогревает CUDA и MediaPipe, в статистику не идёт.
        _run(video, model, args)
        runs = []
        xy = None
        for _ in range(args.repeat):
            stats, xy = _run(video, model, args)
            runs.append(stats)
        saved[str(video)] = xy.tolist()
        median = {key: float(np.median([run[key] for run in runs])) for key in runs[0]}
        print(
            f"{video.name}: frames={median['frames']:.0f} faces={median['faces']:.0f} "
            f"total={median['total_s']:.3f}s prepare={median['prepare_s']:.3f}s "
            f"predict={median['predict_s']:.3f}s other={median['other_s']:.3f}s "
            f"ms/frame={1000 * median['total_s'] / median['frames']:.1f}"
        )

    if args.save:
        args.save.write_text(json.dumps(saved))
    if args.compare:
        reference = json.loads(args.compare.read_text())
        for name, xy in saved.items():
            ref = np.asarray(reference[name], dtype=np.float64)
            cur = np.asarray(xy, dtype=np.float64)
            both = np.isfinite(ref[:, 0]) & np.isfinite(cur[:, 0])
            scale = np.nanstd(ref, axis=0).mean()
            diff = np.abs(ref[both] - cur[both]).max() if both.any() else float("nan")
            print(
                f"{Path(name).name}: max|Δxy|={diff:.4g} (std траектории {scale:.4g}), "
                f"NaN совпадают: {bool((np.isfinite(ref[:, 0]) == np.isfinite(cur[:, 0])).all())}"
            )


if __name__ == "__main__":
    main()
