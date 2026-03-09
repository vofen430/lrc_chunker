from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrc_chunker.alignment import AlignmentConfig, ForcedAlignmentWord, _assign_aligned_words_to_lines, _build_alignment_windows, _flatten_qwen_alignment, build_alignment_payload, fallback_align_from_lrc
from lrc_chunker.chunking import ChunkingConfig, build_chunks
from lrc_chunker.display_timing import apply_chunk_display_timing
from lrc_chunker.external_processor import _collect_folder_pairs, build_parser as build_external_parser, format_lrc_timestamp, load_job_request, main as external_main, render_chunk_lrc, run_job_dir
from lrc_chunker.lrc import LrcParseConfig, parse_lrc
from lrc_chunker.models import LyricLine, WordTiming
from lrc_chunker.motion_m0_extract import extract_m0_features
from lrc_chunker.utils import artifact_name_prefix, find_payload_vocals_path
from lrc_chunker.word_refine import PROFILES, _apply_breath_guard, build_parser as build_refine_parser, refine_payload


def build_word(
    text: str,
    start: float,
    end: float,
    *,
    line_id: int | None = None,
    confidence: float = 1.0,
    index: int = 0,
) -> WordTiming:
    return WordTiming(
        text=text,
        start=start,
        end=end,
        line_id=line_id,
        confidence=confidence,
        index=index,
    )


def test_parse_lrc_removes_head_metadata_and_prefers_primary_language(tmp_path: Path) -> None:
    lrc_path = tmp_path / "sample.lrc"
    lrc_path.write_text(
        "\n".join(
            [
                "[00:00.000][by:test_user]",
                "[00:00.000]作曲 : Someone",
                "[00:00.500]Hello from the other side",
                "[00:00.500]你好 来自另一边",
                "[00:03.000]I must have called a thousand times",
                "[00:03.000]我一定打过上千次电话",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_lrc(str(lrc_path), LrcParseConfig(use_metadata_embedding=False))

    assert [row.text for row in rows] == [
        "Hello from the other side",
        "I must have called a thousand times",
    ]


def test_parse_lrc_keeps_zero_second_lyric_when_metadata_shares_timestamp(tmp_path: Path) -> None:
    lrc_path = tmp_path / "sample_zero.lrc"
    lrc_path.write_text(
        "\n".join(
            [
                "[00:00.000]Cry cry cry baby",
                "[00:00.000][by:虹烧虾的红烧鱼]",
                "[00:00.000]作曲 : Bert Berns/Chris Martin",
                "[00:06.636]Cry cry cry",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_lrc(str(lrc_path), LrcParseConfig(use_metadata_embedding=False))

    assert rows[0].text == "Cry cry cry baby"
    assert any("作曲" in alt for alt in rows[0].alternatives)
    assert any("by:" in alt.lower() for alt in rows[0].alternatives)


def test_parse_lrc_removes_tail_metadata_only_in_edge_window(tmp_path: Path) -> None:
    lrc_path = tmp_path / "sample_tail.lrc"
    lrc_path.write_text(
        "\n".join(
            [
                "[00:00.000]Hello world",
                "[00:10.000]Stay with me",
                "[03:30.000]作词 : Someone",
            ]
        ),
        encoding="utf-8",
    )

    rows = parse_lrc(str(lrc_path), LrcParseConfig(use_metadata_embedding=False))

    assert [row.text for row in rows] == ["Hello world", "Stay with me"]


def test_assign_aligned_words_to_lines_uses_lrc_tokens_for_word_text() -> None:
    lines = [
        LyricLine(line_id=0, timestamp=0.0, text="don't go"),
    ]
    aligned_words = [
        ForcedAlignmentWord(text="dont", start=0.10, end=0.40, confidence=0.9, source="stable_ts"),
        ForcedAlignmentWord(text="gooo", start=0.45, end=0.80, confidence=0.8, source="stable_ts"),
    ]

    words, records = _assign_aligned_words_to_lines(lines, aligned_words)

    assert [word.text for word in words] == ["don't", "go"]
    assert records[0]["word_count"] == 2


def test_flatten_qwen_alignment_accepts_nested_list_results() -> None:
    result = [
        [
            {"text": "Baby", "start_time": 0.10, "end_time": 0.48},
            {"text": "Chop", "start_time": 0.50, "end_time": 0.92},
        ]
    ]

    flat = _flatten_qwen_alignment(result)

    assert [item.text for item in flat] == ["Baby", "Chop"]
    assert flat[0].source == "qwen_forced_aligner"


def test_flatten_qwen_alignment_accepts_result_objects_with_items() -> None:
    class DummyItem:
        def __init__(self, text: str, start_time: float, end_time: float) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class DummyResult:
        def __init__(self, items) -> None:
            self.items = items

    result = [
        DummyResult(
            [
                DummyItem("Cry", 0.12, 0.41),
                DummyItem("Cry", 0.45, 0.72),
            ]
        )
    ]

    flat = _flatten_qwen_alignment(result)

    assert [item.text for item in flat] == ["Cry", "Cry"]
    assert flat[0].start == 0.12
    assert flat[0].source == "qwen_forced_aligner"


def test_build_alignment_windows_splits_long_sequences() -> None:
    lines = [
        LyricLine(line_id=0, timestamp=0.0, text="one"),
        LyricLine(line_id=1, timestamp=8.0, text="two"),
        LyricLine(line_id=2, timestamp=27.0, text="three"),
        LyricLine(line_id=3, timestamp=36.0, text="four"),
        LyricLine(line_id=4, timestamp=61.0, text="five"),
    ]

    windows = _build_alignment_windows(
        lines,
        audio_duration=120.0,
        config=AlignmentConfig(
            alignment_backend="qwen_forced_aligner",
            align_window_target_seconds=30.0,
            align_window_max_seconds=170.0,
            align_window_max_lines=99,
            align_window_pre_roll=1.0,
            align_window_post_roll=1.0,
        ),
    )

    assert len(windows) == 3
    assert [line.line_id for line in windows[0].lines] == [0, 1]
    assert [line.line_id for line in windows[1].lines] == [2, 3]
    assert [line.line_id for line in windows[2].lines] == [4]


def test_fallback_pipeline_produces_non_overlapping_chunks(tmp_path: Path) -> None:
    lrc_path = tmp_path / "sample.lrc"
    lrc_path.write_text(
        "\n".join(
            [
                "[00:00.500]Hello from the other side",
                "[00:03.000]I must have called a thousand times",
                "[00:06.000]To tell you I'm sorry",
            ]
        ),
        encoding="utf-8",
    )

    lines = parse_lrc(str(lrc_path), LrcParseConfig(use_metadata_embedding=False))
    words, _, backend_used = fallback_align_from_lrc(lines, audio_path="demo.wav")
    chunks = build_chunks(words, ChunkingConfig(merge_gap=0.005))
    payload = build_alignment_payload(
        audio_path="demo.wav",
        lrc_path=str(lrc_path),
        lines=lines,
        words=words,
        chunks=chunks,
        config=AlignmentConfig(alignment_backend="lrc"),
        backend_used=backend_used,
    )

    refined, _ = refine_payload(payload, profile="balanced")
    features, report = extract_m0_features(refined)

    assert features["chunks"]
    assert report["no_negative_chunk_gaps"] is True
    assert report["max_visible_chunks_at_once"] == 1


def test_display_timing_uses_lrc_interval_for_single_chunk_line() -> None:
    payload = {
        "meta": {},
        "lines": [
            {"line_id": 0, "timestamp": 6.636, "text": "Cry cry cry"},
            {"line_id": 1, "timestamp": 15.017, "text": "In a book about the world"},
        ],
        "chunks": [
            {
                "chunk_id": 0,
                "start": 15.68,
                "end": 15.68,
                "text": "Cry cry cry",
                "line_ids": [0],
                "words": [
                    {"text": "Cry", "start": 15.68, "end": 15.68, "line_id": 0},
                    {"text": "cry", "start": 15.68, "end": 15.68, "line_id": 0},
                    {"text": "cry", "start": 15.68, "end": 15.68, "line_id": 0},
                ],
            }
        ],
    }

    apply_chunk_display_timing(payload)

    assert payload["chunks"][0]["display_start"] == 6.636
    assert payload["chunks"][0]["display_end"] == 15.017


def test_display_timing_splits_line_budget_by_chunk_duration_weights() -> None:
    payload = {
        "meta": {},
        "lines": [
            {"line_id": 0, "timestamp": 0.0, "text": "one two three four"},
            {"line_id": 1, "timestamp": 10.0, "text": "later"},
        ],
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.2,
                "end": 1.2,
                "text": "one two",
                "line_ids": [0],
                "words": [{"text": "one", "start": 0.2, "end": 0.7, "line_id": 0}],
            },
            {
                "chunk_id": 1,
                "start": 1.2,
                "end": 4.2,
                "text": "three four",
                "line_ids": [0],
                "words": [{"text": "three", "start": 1.2, "end": 2.2, "line_id": 0}],
            },
        ],
    }

    apply_chunk_display_timing(payload)

    assert payload["chunks"][0]["display_start"] == 0.0
    assert payload["chunks"][0]["display_end"] == 2.5
    assert payload["chunks"][1]["display_start"] == 2.5
    assert payload["chunks"][1]["display_end"] == 10.0


def test_m0_prefers_display_times_over_acoustic_chunk_times() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 15.754,
                "end": 15.794,
                "display_start": 6.636,
                "display_end": 15.017,
                "text": "Cry cry cry",
                "words": [
                    {"text": "Cry", "start": 15.754, "end": 15.889},
                    {"text": "cry", "start": 15.957, "end": 16.091},
                ],
            }
        ],
    }

    features, _ = extract_m0_features(payload)
    chunk = features["chunks"][0]

    assert chunk["start"] == 6.636
    assert chunk["end"] == 15.017
    assert chunk["audio_start"] == 15.754
    assert chunk["audio_end"] == 15.794


def test_m0_prefers_simulated_word_times_when_present() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 1.0,
                "end": 2.0,
                "display_start": 0.5,
                "display_end": 2.5,
                "text": "hold on",
                "words": [
                    {
                        "text": "hold",
                        "start": 1.0,
                        "end": 1.0,
                        "simulated_start": 0.9,
                        "simulated_end": 1.25,
                        "simulated_midpoint": 1.075,
                        "simulated_duration": 0.35,
                        "timing_source": "simulated_from_chunk_context",
                        "timing_confidence": 0.35,
                    },
                    {"text": "on", "start": 1.4, "end": 1.7},
                ],
            }
        ],
    }

    features, report = extract_m0_features(payload)
    word = features["chunks"][0]["words"][0]

    assert word["start"] == 0.9
    assert word["end"] == 1.25
    assert word["audio_start"] == 1.0
    assert word["audio_end"] == 1.0
    assert word["midpoint"] == 1.075
    assert word["simulated_duration"] == 0.35
    assert word["is_simulated"] is True
    assert word["timing_source"] == "simulated_from_chunk_context"
    assert report["positive_effective_word_durations"] is True


def test_payload_vocals_path_is_discovered_from_meta(tmp_path: Path) -> None:
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"fake")
    payload = {
        "meta": {
            "denoiser_output_path": str(vocals),
        }
    }

    assert find_payload_vocals_path(payload) == str(vocals.resolve())


def test_breath_guard_pushes_start_to_voiced_frame() -> None:
    params = dict(PROFILES["balanced"])
    frame_times = np.asarray([0.50, 0.56, 0.62, 0.68, 0.74], dtype=np.float64)
    rms = np.asarray([0.01, 0.015, 0.08, 0.11, 0.10], dtype=np.float64)
    flatness = np.asarray([0.92, 0.86, 0.34, 0.28, 0.30], dtype=np.float64)

    shifted = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        params,
    )

    assert shifted >= 0.62


def test_breath_guard_keeps_voiced_start_when_not_breathy() -> None:
    params = dict(PROFILES["balanced"])
    frame_times = np.asarray([0.50, 0.56, 0.62], dtype=np.float64)
    rms = np.asarray([0.08, 0.10, 0.11], dtype=np.float64)
    flatness = np.asarray([0.30, 0.28, 0.26], dtype=np.float64)

    kept = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        params,
    )

    assert kept == 0.50


def test_slow_attack_profile_waits_for_later_attack() -> None:
    balanced = dict(PROFILES["balanced"])
    slow_attack = dict(PROFILES["slow_attack"])
    frame_times = np.asarray([0.50, 0.56, 0.62, 0.68, 0.74, 0.80], dtype=np.float64)
    rms = np.asarray([0.03, 0.04, 0.06, 0.09, 0.13, 0.14], dtype=np.float64)
    flatness = np.asarray([0.40, 0.38, 0.32, 0.28, 0.24, 0.22], dtype=np.float64)

    balanced_shift = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        balanced,
    )
    slow_attack_shift = _apply_breath_guard(
        0.50,
        0.95,
        0.25,
        0,
        frame_times,
        rms,
        flatness,
        0.04,
        slow_attack,
    )

    assert balanced_shift == 0.62
    assert slow_attack_shift >= 0.74


def test_refine_payload_syncs_top_level_words() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "words": [
                    {"text": "hello", "start": 0.10, "end": 0.40},
                    {"text": "world", "start": 0.45, "end": 0.90},
                ],
            }
        ],
        "words": [
            {"text": "hello", "start": 0.00, "end": 0.30},
            {"text": "world", "start": 0.35, "end": 0.70},
        ],
    }

    refined, _ = refine_payload(payload, profile="balanced")

    assert refined["words"] == refined["chunks"][0]["words"]


def test_refine_payload_can_anchor_line_first_word_to_lrc_timestamp() -> None:
    payload = {
        "meta": {},
        "lines": [
            {"line_id": 0, "timestamp": 1.0, "text": "hello world"},
        ],
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.6,
                "end": 1.8,
                "text": "hello world",
                "words": [
                    {"text": "hello", "start": 0.62, "end": 0.74, "line_id": 0},
                    {"text": "world", "start": 1.15, "end": 1.70, "line_id": 0},
                ],
            }
        ],
        "words": [
            {"text": "hello", "start": 0.62, "end": 0.74, "line_id": 0},
            {"text": "world", "start": 1.15, "end": 1.70, "line_id": 0},
        ],
    }

    refined, report = refine_payload(
        payload,
        profile="balanced",
        use_lrc_anchors=True,
        lrc_anchor_window=0.30,
        lrc_anchor_weight=4.0,
        lrc_anchor_keep_weight=0.20,
        lrc_anchor_span_words=4,
        lrc_anchor_max_ratio=0.80,
    )

    assert refined["chunks"][0]["words"][0]["start"] == 1.0
    assert refined["chunks"][0]["words"][1]["start"] > 1.15
    assert refined["chunks"][0]["words"][1]["start"] - 1.15 < 1.0 - 0.62
    assert refined["chunks"][0]["words"][1]["end"] > refined["chunks"][0]["words"][1]["start"]
    assert report["lrc_anchor_moves"] == 1


def test_refine_cli_defaults_to_small_flow_profile_and_lrc_anchors() -> None:
    parser = build_refine_parser()
    args = parser.parse_args(["demo.json"])

    assert args.profile == "slow_attack"
    assert args.use_lrc_anchors is True
    assert args.lrc_anchor_weight == 4.5
    assert args.lrc_anchor_keep_weight == 0.20
    assert args.lrc_anchor_span_words == 4
    assert args.lrc_anchor_max_ratio == 0.35


def test_format_lrc_timestamp_uses_millisecond_precision() -> None:
    assert format_lrc_timestamp(62.345) == "[01:02.345]"


def test_artifact_name_prefix_uses_job_or_run_dir_context() -> None:
    prefix = artifact_name_prefix(
        run_dir="/tmp/ae_acoldplay_check2/visualization/m1",
        audio_path="/home/dev/workspace/lrc_chunker/test/A COLD PLAY - The Kid LAROI.wav",
    )

    assert prefix == "ae_acoldplay_check2_A_COLD_PLAY_-_The_Kid_LAROI"


def test_artifact_name_prefix_avoids_duplicate_audio_suffix_when_run_dir_already_identifies_source() -> None:
    prefix = artifact_name_prefix(
        run_dir="/home/dev/workspace/lrc_chunker/artifacts/m1/Memories_small_en",
        audio_path="/home/dev/workspace/lrc_chunker/test/Memories.wav",
    )

    assert prefix == "Memories_small_en"


def test_render_chunk_lrc_keeps_ground_truth_on_first_split_chunk() -> None:
    payload = {
        "lines": [
            {"line_id": 0, "timestamp": 10.0, "text": "hello world again"},
            {"line_id": 1, "timestamp": 15.0, "text": "later line"},
        ],
        "chunks": [
            {
                "chunk_id": 0,
                "start": 10.120,
                "end": 11.0,
                "text": "hello world",
                "line_ids": [0],
                "words": [
                    {"text": "hello", "start": 10.120, "end": 10.400, "line_id": 0},
                    {"text": "world", "start": 10.410, "end": 10.900, "line_id": 0},
                ],
            },
            {
                "chunk_id": 1,
                "start": 11.250,
                "end": 11.700,
                "text": "again",
                "line_ids": [0],
                "words": [
                    {"text": "again", "start": 11.250, "end": 11.700, "line_id": 0},
                ],
            },
            {
                "chunk_id": 2,
                "start": 15.100,
                "end": 15.700,
                "text": "later line",
                "line_ids": [1],
                "words": [
                    {"text": "later", "start": 15.100, "end": 15.300, "line_id": 1},
                    {"text": "line", "start": 15.320, "end": 15.700, "line_id": 1},
                ],
            },
        ],
    }

    text, warnings = render_chunk_lrc(payload)

    assert text.splitlines() == [
        "[00:10.000]hello world",
        "[00:11.250]again",
        "[00:15.000]later line",
    ]
    assert warnings == []


def test_load_job_request_reads_batch_manifest_and_preserves_order(tmp_path: Path) -> None:
    audio1 = tmp_path / "a.wav"
    audio2 = tmp_path / "b.mp3"
    lrc1 = tmp_path / "a.lrc"
    lrc2 = tmp_path / "b.lrc"
    for path in (audio1, audio2, lrc1, lrc2):
        path.write_text("x", encoding="utf-8")

    job_dir = tmp_path / "job"
    result_dir = job_dir / "results"
    manifest_path = job_dir / "batch_pairs.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        (
            "{\n"
            '  "protocol_version": 1,\n'
            '  "manifest_type": "ae_lrc_batch_pairs",\n'
            '  "created_utc": "2026-03-07T00:00:00Z",\n'
            '  "source": {"host": "after-effects", "panel": "LRC_Panel_Modular"},\n'
            '  "batch": {"order_mode": "ui_order", "row_count": 3, "ready_count": 2, "skipped_count": 1},\n'
            '  "items": [\n'
            f'    {{"row_index": 2, "row_id": 22, "pair_state": "ready", "title": "", "artist": "", "lrc_path": "{lrc2.resolve().as_posix()}", "audio_path": "{audio2.resolve().as_posix()}"}},\n'
            f'    {{"row_index": 1, "row_id": 11, "pair_state": "ready", "title": "", "artist": "", "lrc_path": "{lrc1.resolve().as_posix()}", "audio_path": "{audio1.resolve().as_posix()}"}}\n'
            "  ],\n"
            '  "skipped_rows": [{"row_index": 3, "row_id": 33, "pair_state": "missing_audio", "title": "", "artist": "", "lrc_path": "", "audio_path": "", "reason": "missing"}]\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (job_dir / "request.json").write_text(
        (
            "{\n"
            '  "protocol_version": 1,\n'
            '  "job_id": "job_1",\n'
            '  "created_utc": "2026-03-07T00:00:00Z",\n'
            f'  "input": {{"mode": "batch_manifest", "batch_manifest_path": "{manifest_path.resolve().as_posix()}"}},\n'
            f'  "output": {{"result_dir": "{result_dir.resolve().as_posix()}"}},\n'
            '  "options": {"mode": "default"},\n'
            f'  "callback": {{"status_file": "{(job_dir / "status.json").resolve().as_posix()}", "result_file": "{(job_dir / "result.json").resolve().as_posix()}", "complete_flag": "{(job_dir / "complete.flag").resolve().as_posix()}", "failed_flag": "{(job_dir / "failed.flag").resolve().as_posix()}", "cancel_flag": "{(job_dir / "cancel.flag").resolve().as_posix()}"}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    request = load_job_request(job_dir)

    assert request.mode == "batch_manifest"
    assert [pair.row_index for pair in request.pairs] == [2, 1]
    assert Path(request.pairs[0].result_lrc_path).name.startswith("0002_")


def test_render_chunk_lrc_prefers_word_text_over_chunk_text() -> None:
    payload = {
        "lines": [
            {"line_id": 0, "timestamp": 1.0, "text": "hello world"},
        ],
        "chunks": [
            {
                "chunk_id": 0,
                "start": 1.0,
                "end": 1.5,
                "text": "model drifted text",
                "line_ids": [0],
                "words": [
                    {"text": "hello", "start": 1.0, "end": 1.2, "line_id": 0},
                    {"text": "world", "start": 1.2, "end": 1.5, "line_id": 0},
                ],
            }
        ],
    }

    text, warnings = render_chunk_lrc(payload)

    assert warnings == []
    assert text.splitlines() == ["[00:01.000]hello world"]


def test_refine_payload_enforces_positive_chunk_duration() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 1.0,
                "end": 1.0,
                "text": "hello",
                "words": [
                    {"text": "hello", "start": 1.0, "end": 1.0},
                ],
            }
        ],
        "words": [
            {"text": "hello", "start": 1.0, "end": 1.0},
        ],
    }

    refined, report = refine_payload(payload, profile="slow_attack")

    assert refined["chunks"] == []
    assert report["dropped_zero_chunks"] == 1


def test_refine_payload_drops_leading_and_trailing_zero_duration_chunks() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.0,
                "end": 0.0,
                "text": "meta",
                "words": [{"text": "meta", "start": 0.0, "end": 0.0}],
            },
            {
                "chunk_id": 1,
                "start": 1.0,
                "end": 1.4,
                "text": "hello",
                "words": [{"text": "hello", "start": 1.0, "end": 1.4}],
            },
            {
                "chunk_id": 2,
                "start": 2.0,
                "end": 2.0,
                "text": "tail",
                "words": [{"text": "tail", "start": 2.0, "end": 2.0}],
            },
        ],
        "words": [
            {"text": "meta", "start": 0.0, "end": 0.0},
            {"text": "hello", "start": 1.0, "end": 1.4},
            {"text": "tail", "start": 2.0, "end": 2.0},
        ],
    }

    refined, report = refine_payload(payload, profile="slow_attack")

    assert len(refined["chunks"]) == 1
    assert refined["chunks"][0]["text"] == "hello"
    assert report["dropped_zero_chunks"] == 2


def test_refine_payload_keeps_interior_zero_duration_chunk_but_makes_it_positive() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 1.0,
                "end": 1.4,
                "text": "hello",
                "words": [{"text": "hello", "start": 1.0, "end": 1.4}],
            },
            {
                "chunk_id": 1,
                "start": 1.6,
                "end": 1.6,
                "text": "bridge",
                "words": [{"text": "bridge", "start": 1.6, "end": 1.6}],
            },
            {
                "chunk_id": 2,
                "start": 2.0,
                "end": 2.5,
                "text": "world",
                "words": [{"text": "world", "start": 2.0, "end": 2.5}],
            },
        ],
        "words": [
            {"text": "hello", "start": 1.0, "end": 1.4},
            {"text": "bridge", "start": 1.6, "end": 1.6},
            {"text": "world", "start": 2.0, "end": 2.5},
        ],
    }

    refined, report = refine_payload(payload, profile="slow_attack")

    assert len(refined["chunks"]) == 3
    assert refined["chunks"][1]["end"] > refined["chunks"][1]["start"]
    assert report["interior_zero_chunks_fixed"] == 1


def test_refine_payload_simulates_zero_duration_word_from_chunk_context() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 1.0,
                "end": 2.2,
                "display_start": 1.0,
                "display_end": 2.2,
                "text": "hello there world",
                "words": [
                    {"text": "hello", "start": 1.0, "end": 1.3, "confidence": 0.9},
                    {"text": "there", "start": 1.5, "end": 1.5, "confidence": 0.2},
                    {"text": "world", "start": 1.8, "end": 2.2, "confidence": 0.9},
                ],
            }
        ],
        "words": [
            {"text": "hello", "start": 1.0, "end": 1.3, "confidence": 0.9},
            {"text": "there", "start": 1.5, "end": 1.5, "confidence": 0.2},
            {"text": "world", "start": 1.8, "end": 2.2, "confidence": 0.9},
        ],
    }

    refined, report = refine_payload(payload, profile="slow_attack")

    repaired = refined["chunks"][0]["words"][1]
    assert repaired["timing_source"] == "simulated_from_chunk_context"
    assert repaired["simulated_duration"] > 0.0
    assert 1.3 <= repaired["simulated_midpoint"] <= 1.8
    assert repaired["simulated_start"] < repaired["simulated_end"]
    assert report["simulated_zero_words"] == 1
    assert report["simulated_zero_word_groups"] == 1


def test_refine_payload_simulates_consecutive_zero_duration_words_as_a_group() -> None:
    payload = {
        "meta": {},
        "chunks": [
            {
                "chunk_id": 0,
                "start": 0.0,
                "end": 2.4,
                "display_start": 0.0,
                "display_end": 2.4,
                "text": "around my house tonight",
                "words": [
                    {"text": "around", "start": 0.0, "end": 0.4},
                    {"text": "my", "start": 0.9, "end": 0.9},
                    {"text": "house", "start": 1.1, "end": 1.1},
                    {"text": "tonight", "start": 1.8, "end": 2.3},
                ],
            }
        ],
        "words": [
            {"text": "around", "start": 0.0, "end": 0.4},
            {"text": "my", "start": 0.9, "end": 0.9},
            {"text": "house", "start": 1.1, "end": 1.1},
            {"text": "tonight", "start": 1.8, "end": 2.3},
        ],
    }

    refined, report = refine_payload(payload, profile="slow_attack")

    my_word = refined["chunks"][0]["words"][1]
    house_word = refined["chunks"][0]["words"][2]
    assert my_word["timing_source"] == "simulated_from_chunk_context"
    assert house_word["timing_source"] == "simulated_from_chunk_context"
    assert my_word["simulated_end"] <= house_word["simulated_start"]
    assert my_word["simulated_midpoint"] < house_word["simulated_midpoint"]
    assert report["simulated_zero_words"] == 2
    assert report["simulated_zero_word_groups"] == 1


def test_chunking_splits_high_confidence_short_span_line_break() -> None:
    words = [
        build_word("hello", 0.00, 0.07, line_id=0, confidence=0.995, index=0),
        build_word("world", 0.08, 0.15, line_id=1, confidence=0.992, index=1),
    ]

    chunks = build_chunks(words, ChunkingConfig(merge_gap=0.005, chunker_model="legacy"))

    assert len(chunks) == 2
    assert [chunk.text for chunk in chunks] == ["hello", "world"]


def test_chunking_keeps_low_confidence_line_break_together() -> None:
    words = [
        build_word("hello", 0.00, 0.07, line_id=0, confidence=0.97, index=0),
        build_word("world", 0.08, 0.15, line_id=1, confidence=0.96, index=1),
    ]

    chunks = build_chunks(words, ChunkingConfig(chunker_model="legacy"))

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunking_no_longer_splits_on_character_limit_alone() -> None:
    words = [
        build_word("supercalifragilisticexpialidocious", 0.00, 0.40, line_id=0, confidence=0.999, index=0),
        build_word("pneumonoultramicroscopicsilicovolcanoconiosis", 0.44, 0.86, line_id=0, confidence=0.999, index=1),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(
            chunker_model="legacy",
            max_chars=8,
            max_words=6,
            max_gap=1.0,
            max_dur=5.0,
            merge_gap=0.5,
        ),
    )

    assert len(chunks) == 1
    assert "supercalifragilisticexpialidocious" in chunks[0].text
    assert "pneumonoultramicroscopicsilicovolcanoconiosis" in chunks[0].text


def test_semantic_chunking_never_crosses_line_boundaries() -> None:
    words = [
        build_word("hello", 0.00, 0.30, line_id=0, confidence=0.90, index=0),
        build_word("world", 0.32, 0.62, line_id=0, confidence=0.90, index=1),
        build_word("again", 0.70, 1.00, line_id=1, confidence=0.90, index=2),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(chunker_model="semantic_dp"),
        line_timestamps={0: 0.0, 1: 0.7},
    )

    assert [chunk.text for chunk in chunks] == ["hello world", "again"]


def test_semantic_chunking_hard_anchors_first_chunk_to_line_timestamp() -> None:
    words = [
        build_word("hello", 1.05, 1.30, line_id=0, confidence=0.99, index=0),
        build_word("world", 1.34, 1.72, line_id=0, confidence=0.99, index=1),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(chunker_model="semantic_dp", line_start_anchor=True),
        line_timestamps={0: 1.0},
    )

    assert chunks[0].start == 1.0


def test_semantic_chunking_suppresses_line_anchor_when_alignment_is_far_from_lrc() -> None:
    words = [
        build_word("cry", 15.04, 15.20, line_id=0, confidence=0.99, index=0),
        build_word("baby", 15.22, 15.84, line_id=0, confidence=0.99, index=1),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(
            chunker_model="semantic_dp",
            line_start_anchor=True,
            line_start_anchor_tolerance=1.0,
        ),
        line_timestamps={0: 0.0},
    )

    assert chunks[0].start == 15.04
    assert chunks[0].flags["line_anchor_suppressed"] is True


def test_semantic_chunking_blocks_short_gap_cut() -> None:
    words = [
        build_word("hold", 0.00, 0.20, line_id=0, confidence=0.99, index=0),
        build_word("on", 0.23, 0.43, line_id=0, confidence=0.99, index=1),
        build_word("tight", 0.70, 1.10, line_id=0, confidence=0.99, index=2),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(chunker_model="semantic_dp", short_gap_block=0.12),
        line_timestamps={0: 0.0},
    )

    assert [chunk.text for chunk in chunks] == ["hold on", "tight"]


def test_semantic_chunking_prefers_two_plus_two_when_gap_is_long() -> None:
    words = [
        build_word("when", 0.00, 0.25, line_id=0, confidence=0.99, index=0),
        build_word("you", 0.28, 0.52, line_id=0, confidence=0.99, index=1),
        build_word("cry", 1.10, 1.42, line_id=0, confidence=0.99, index=2),
        build_word("baby", 1.45, 1.85, line_id=0, confidence=0.99, index=3),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(chunker_model="semantic_dp"),
        line_timestamps={0: 0.0},
    )

    assert [chunk.text for chunk in chunks] == ["when you", "cry baby"]


def test_semantic_chunking_prefers_three_plus_three_with_flat_internal_gaps() -> None:
    words = [
        build_word("one", 0.00, 0.20, line_id=0, confidence=0.99, index=0),
        build_word("two", 0.20, 0.40, line_id=0, confidence=0.99, index=1),
        build_word("three", 0.40, 0.60, line_id=0, confidence=0.99, index=2),
        build_word("four", 0.60, 0.80, line_id=0, confidence=0.99, index=3),
        build_word("five", 0.80, 1.00, line_id=0, confidence=0.99, index=4),
        build_word("six", 1.00, 1.20, line_id=0, confidence=0.99, index=5),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(chunker_model="semantic_dp", embedding_model_path="./DOES_NOT_EXIST"),
        line_timestamps={0: 0.0},
    )

    assert [chunk.text for chunk in chunks] == ["one two three", "four five six"]


def test_semantic_chunking_can_split_long_line_without_large_gaps() -> None:
    words = [
        build_word("around", 0.00, 0.18, line_id=0, confidence=0.99, index=0),
        build_word("my", 0.18, 0.36, line_id=0, confidence=0.99, index=1),
        build_word("house", 0.36, 0.54, line_id=0, confidence=0.99, index=2),
        build_word("i", 0.54, 0.72, line_id=0, confidence=0.99, index=3),
        build_word("still", 0.72, 0.90, line_id=0, confidence=0.99, index=4),
        build_word("got", 0.90, 1.08, line_id=0, confidence=0.99, index=5),
        build_word("up", 1.08, 1.26, line_id=0, confidence=0.99, index=6),
        build_word("all", 1.26, 1.44, line_id=0, confidence=0.99, index=7),
    ]

    chunks = build_chunks(
        words,
        ChunkingConfig(chunker_model="semantic_dp", embedding_model_path="./DOES_NOT_EXIST"),
        line_timestamps={0: 0.0},
    )

    assert [chunk.text for chunk in chunks] == ["around my house i", "still got up all"]


def test_collect_folder_pairs_matches_same_basename_and_prefers_wav(tmp_path: Path) -> None:
    (tmp_path / "song.lrc").write_text("[00:01.000]song", encoding="utf-8")
    (tmp_path / "song.wav").write_bytes(b"wav")
    (tmp_path / "song.mp3").write_bytes(b"mp3")
    (tmp_path / "other.lrc").write_text("[00:01.000]other", encoding="utf-8")

    pairs = _collect_folder_pairs(tmp_path, tmp_path / "out")

    assert len(pairs) == 1
    assert pairs[0].audio_path.endswith("song.wav")


def test_external_cli_requires_ae_flag_for_launch() -> None:
    parser = build_external_parser()
    args = parser.parse_args(["launch", "--job-dir", "/tmp/demo"])

    assert args.ae is False
    assert external_main(["launch", "--job-dir", "/tmp/demo"]) == 10


def test_run_job_dir_writes_result_and_complete_flag(tmp_path: Path, monkeypatch) -> None:
    from lrc_chunker import external_processor as ext

    audio = tmp_path / "song.wav"
    lrc = tmp_path / "song.lrc"
    audio.write_bytes(b"fake")
    lrc.write_text("[00:01.000]hello world", encoding="utf-8")
    job_dir = tmp_path / "job"
    result_dir = job_dir / "results"
    manifest_path = job_dir / "batch_pairs.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        (
            "{\n"
            '  "protocol_version": 1,\n'
            '  "manifest_type": "ae_lrc_batch_pairs",\n'
            '  "created_utc": "2026-03-07T00:00:00Z",\n'
            '  "source": {"host": "after-effects", "panel": "LRC_Panel_Modular"},\n'
            '  "batch": {"order_mode": "ui_order", "row_count": 1, "ready_count": 1, "skipped_count": 0},\n'
            '  "items": [\n'
            f'    {{"row_index": 1, "row_id": 101, "pair_state": "ready", "title": "Song", "artist": "Artist", "lrc_path": "{lrc.resolve().as_posix()}", "audio_path": "{audio.resolve().as_posix()}"}}\n'
            "  ],\n"
            '  "skipped_rows": []\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    (job_dir / "request.json").write_text(
        (
            "{\n"
            '  "protocol_version": 1,\n'
            '  "job_id": "job_batch",\n'
            '  "created_utc": "2026-03-07T00:00:00Z",\n'
            f'  "input": {{"mode": "batch_manifest", "batch_manifest_path": "{manifest_path.resolve().as_posix()}"}},\n'
            f'  "output": {{"result_dir": "{result_dir.resolve().as_posix()}"}},\n'
            '  "options": {"mode": "default"},\n'
            f'  "callback": {{"status_file": "{(job_dir / "status.json").resolve().as_posix()}", "result_file": "{(job_dir / "result.json").resolve().as_posix()}", "complete_flag": "{(job_dir / "complete.flag").resolve().as_posix()}", "failed_flag": "{(job_dir / "failed.flag").resolve().as_posix()}", "cancel_flag": "{(job_dir / "cancel.flag").resolve().as_posix()}"}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    def fake_process_pair(pair, *, options, work_dir):
        Path(pair.result_lrc_path).parent.mkdir(parents=True, exist_ok=True)
        Path(pair.result_lrc_path).write_text("[00:01.000]hello world\n", encoding="utf-8")
        return {
            "row_index": pair.row_index,
            "row_id": pair.row_id,
            "result_lrc_path": pair.result_lrc_path,
            "input_line_count": 1,
            "output_line_count": 1,
            "chunk_count": 1,
            "warnings": [],
        }

    monkeypatch.setattr(ext, "process_pair_to_lrc", fake_process_pair)

    code = run_job_dir(job_dir)

    assert code == 0
    assert (job_dir / "complete.flag").is_file()
    result_payload = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    status_payload = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    assert result_payload["state"] == "completed"
    assert status_payload["state"] == "completed"
    assert status_payload["detail"]["items_total"] == 1
    assert status_payload["detail"]["items_completed"] == 1
    assert status_payload["detail"]["signals"]["is_terminal"] is True
    assert status_payload["detail"]["signals"]["is_success"] is True
    assert status_payload["detail"]["current_summary"] == "completed"
    assert status_payload["detail"]["result_overview"]["items"][0]["result_lrc_path"].endswith(".lrc")
