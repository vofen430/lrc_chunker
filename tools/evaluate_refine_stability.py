#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_chunk_words(payload: dict) -> List[dict]:
    words: List[dict] = []
    for chunk in payload.get("chunks", []) or []:
        words.extend(chunk.get("words", []) or [])
    return words


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(ordered[low])
    weight = pos - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _ratio_le(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value <= threshold) / len(values)


def _duration_stats(words: Sequence[dict]) -> Dict[str, object]:
    durations = [float(word.get("end", 0.0)) - float(word.get("start", 0.0)) for word in words]
    overlaps = sum(
        1
        for idx in range(len(words) - 1)
        if float(words[idx + 1].get("start", 0.0)) < float(words[idx].get("end", 0.0))
    )
    start_backwards = sum(
        1
        for idx in range(len(words) - 1)
        if float(words[idx + 1].get("start", 0.0)) < float(words[idx].get("start", 0.0))
    )
    zero_rows = [
        {
            "index": idx,
            "text": str(word.get("text") or ""),
            "start": float(word.get("start", 0.0)),
            "end": float(word.get("end", 0.0)),
        }
        for idx, (word, duration) in enumerate(zip(words, durations))
        if duration <= 0.0
    ]
    return {
        "word_count": len(words),
        "non_positive_duration_count": sum(1 for duration in durations if duration <= 0.0),
        "under_100ms_count": sum(1 for duration in durations if duration < 0.1),
        "min_duration": min(durations) if durations else 0.0,
        "median_duration": statistics.median(durations) if durations else 0.0,
        "p90_duration": _quantile(durations, 0.9),
        "max_duration": max(durations) if durations else 0.0,
        "overlap_pair_count": overlaps,
        "start_backward_count": start_backwards,
        "zero_duration_examples": zero_rows[:20],
    }


def _shift_rows(base_words: Sequence[dict], opt_words: Sequence[dict], anchor_indices: set[int]) -> List[dict]:
    rows: List[dict] = []
    for idx, (base_word, opt_word) in enumerate(zip(base_words, opt_words)):
        is_anchor = idx in anchor_indices
        base_start = float(base_word.get("start", 0.0))
        base_end = float(base_word.get("end", 0.0))
        opt_start = float(opt_word.get("start", 0.0))
        opt_end = float(opt_word.get("end", 0.0))
        rows.append(
            {
                "index": idx,
                "text": str(opt_word.get("text") or base_word.get("text") or ""),
                "is_anchor": is_anchor,
                "base_start": base_start,
                "opt_start": opt_start,
                "base_end": base_end,
                "opt_end": opt_end,
                "start_shift": opt_start - base_start,
                "end_shift": opt_end - base_end,
                "duration_shift": (opt_end - opt_start) - (base_end - base_start),
                "time_ref": base_start,
            }
        )
    return rows


def _shift_stats(rows: Sequence[dict]) -> Dict[str, object]:
    start_abs = [abs(float(row["start_shift"])) for row in rows]
    end_abs = [abs(float(row["end_shift"])) for row in rows]
    dur_abs = [abs(float(row["duration_shift"])) for row in rows]
    return {
        "count": len(rows),
        "abs_start_shift": {
            "median": statistics.median(start_abs) if start_abs else 0.0,
            "p90": _quantile(start_abs, 0.9),
            "p95": _quantile(start_abs, 0.95),
            "max": max(start_abs) if start_abs else 0.0,
            "ratio_le_50ms": _ratio_le(start_abs, 0.05),
            "ratio_le_100ms": _ratio_le(start_abs, 0.1),
            "ratio_le_200ms": _ratio_le(start_abs, 0.2),
        },
        "abs_end_shift": {
            "median": statistics.median(end_abs) if end_abs else 0.0,
            "p90": _quantile(end_abs, 0.9),
            "p95": _quantile(end_abs, 0.95),
            "max": max(end_abs) if end_abs else 0.0,
            "ratio_le_50ms": _ratio_le(end_abs, 0.05),
            "ratio_le_100ms": _ratio_le(end_abs, 0.1),
            "ratio_le_200ms": _ratio_le(end_abs, 0.2),
        },
        "abs_duration_shift": {
            "median": statistics.median(dur_abs) if dur_abs else 0.0,
            "p90": _quantile(dur_abs, 0.9),
            "p95": _quantile(dur_abs, 0.95),
            "max": max(dur_abs) if dur_abs else 0.0,
            "ratio_le_50ms": _ratio_le(dur_abs, 0.05),
            "ratio_le_100ms": _ratio_le(dur_abs, 0.1),
            "ratio_le_200ms": _ratio_le(dur_abs, 0.2),
        },
    }


def _write_top_movers(rows: Sequence[dict], out_csv: Path) -> None:
    top_rows = sorted(
        rows,
        key=lambda row: max(abs(float(row["start_shift"])), abs(float(row["end_shift"])), abs(float(row["duration_shift"]))),
        reverse=True,
    )[:50]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "text",
                "is_anchor",
                "base_start",
                "opt_start",
                "base_end",
                "opt_end",
                "start_shift",
                "end_shift",
                "duration_shift",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(top_rows)


def _render_plots(
    *,
    shift_rows: Sequence[dict],
    anchor_rows: Sequence[dict],
    base_stats: Dict[str, object],
    opt_stats: Dict[str, object],
    out_dir: Path,
) -> List[Path]:
    import matplotlib.pyplot as plt

    out_paths: List[Path] = []
    non_anchor = [row for row in shift_rows if not bool(row["is_anchor"])]
    start_abs = [abs(float(row["start_shift"])) for row in non_anchor]
    end_abs = [abs(float(row["end_shift"])) for row in non_anchor]
    dur_abs = [abs(float(row["duration_shift"])) for row in non_anchor]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 30
    ax.hist(start_abs, bins=bins, alpha=0.65, label="|start shift|")
    ax.hist(end_abs, bins=bins, alpha=0.65, label="|end shift|")
    ax.hist(dur_abs, bins=bins, alpha=0.65, label="|duration shift|")
    ax.set_title("Non-anchor Shift Distribution")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Word Count")
    ax.legend()
    hist_path = out_dir / "non_anchor_shift_hist.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    out_paths.append(hist_path)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    times = [float(row["time_ref"]) for row in non_anchor]
    shifts = [float(row["start_shift"]) for row in non_anchor]
    ax.scatter(times, shifts, s=10, alpha=0.65)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Non-anchor Start Shifts Over Time")
    ax.set_xlabel("Song Time (s)")
    ax.set_ylabel("Start Shift (s)")
    timeline_path = out_dir / "non_anchor_start_shift_timeline.png"
    fig.tight_layout()
    fig.savefig(timeline_path, dpi=160)
    plt.close(fig)
    out_paths.append(timeline_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(
        ["base zero", "opt zero", "base <100ms", "opt <100ms"],
        [
            int(base_stats["non_positive_duration_count"]),
            int(opt_stats["non_positive_duration_count"]),
            int(base_stats["under_100ms_count"]),
            int(opt_stats["under_100ms_count"]),
        ],
        color=["#6f7d8c", "#cc6b5a", "#6f7d8c", "#cc6b5a"],
    )
    axes[0].set_title("Duration Validity Counts")
    axes[0].tick_params(axis="x", rotation=20)

    anchor_labels = [str(row.get("anchor_text") or "") for row in anchor_rows]
    before_errors = [abs(float(row.get("base_error", 0.0))) for row in anchor_rows]
    after_errors = [abs(float(row.get("opt_error", 0.0))) for row in anchor_rows]
    x = range(len(anchor_rows))
    axes[1].bar([i - 0.2 for i in x], before_errors, width=0.4, label="before")
    axes[1].bar([i + 0.2 for i in x], after_errors, width=0.4, label="after")
    axes[1].set_xticks(list(x), anchor_labels, rotation=25)
    axes[1].set_title("Anchor Absolute Error")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()

    combo_path = out_dir / "duration_validity_and_anchor_errors.png"
    fig.tight_layout()
    fig.savefig(combo_path, dpi=160)
    plt.close(fig)
    out_paths.append(combo_path)
    return out_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate refined word timing stability against a baseline alignment.")
    parser.add_argument("--base", required=True, type=str)
    parser.add_argument("--opt", required=True, type=str)
    parser.add_argument("--anchor-eval", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args()

    base_payload = _read_json(Path(args.base))
    opt_payload = _read_json(Path(args.opt))
    anchor_eval = _read_json(Path(args.anchor_eval))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_words = _flatten_chunk_words(base_payload)
    opt_words = _flatten_chunk_words(opt_payload)
    if len(base_words) != len(opt_words):
        raise ValueError(f"word count mismatch: {len(base_words)} != {len(opt_words)}")

    anchor_indices = {
        int(row["matched_index"])
        for row in anchor_eval.get("rows", [])
        if row.get("matched") and row.get("matched_index") is not None
    }
    shift_rows = _shift_rows(base_words, opt_words, anchor_indices)
    non_anchor_rows = [row for row in shift_rows if not bool(row["is_anchor"])]

    anchor_rows: List[dict] = []
    for row in anchor_eval.get("rows", []):
        if not row.get("matched"):
            continue
        idx = int(row["matched_index"])
        anchor_rows.append(
            {
                **row,
                "base_start": float(base_words[idx]["start"]),
                "opt_start": float(opt_words[idx]["start"]),
                "base_error": float(base_words[idx]["start"]) - float(row["anchor_start"]),
                "opt_error": float(opt_words[idx]["start"]) - float(row["anchor_start"]),
            }
        )

    base_stats = _duration_stats(base_words)
    opt_stats = _duration_stats(opt_words)
    shift_stats = _shift_stats(non_anchor_rows)

    consistency = {
        "top_level_words_match_chunks_base": base_payload.get("words", []) == base_words,
        "top_level_words_match_chunks_opt": opt_payload.get("words", []) == opt_words,
    }

    summary = {
        "base_path": str(Path(args.base).resolve()),
        "opt_path": str(Path(args.opt).resolve()),
        "anchor_eval_path": str(Path(args.anchor_eval).resolve()),
        "word_count": len(base_words),
        "anchor_count": len(anchor_indices),
        "non_anchor_count": len(non_anchor_rows),
        "payload_consistency": consistency,
        "duration_validity": {
            "base": base_stats,
            "optimized": opt_stats,
        },
        "non_anchor_shift_stats": shift_stats,
        "anchor_rows": anchor_rows,
    }

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    top_movers_path = out_dir / "non_anchor_top_movers.csv"
    _write_top_movers(non_anchor_rows, top_movers_path)
    plot_paths = _render_plots(
        shift_rows=shift_rows,
        anchor_rows=anchor_rows,
        base_stats=base_stats,
        opt_stats=opt_stats,
        out_dir=out_dir,
    )

    print(summary_path)
    print(top_movers_path)
    for path in plot_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
