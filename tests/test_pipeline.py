from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrc_chunker.alignment import AlignmentConfig, _assign_aligned_words_to_lines, build_alignment_payload, fallback_align_from_lrc
from lrc_chunker.chunking import ChunkingConfig, build_chunks
from lrc_chunker.external_processor import _collect_folder_pairs, build_parser as build_external_parser, format_lrc_timestamp, load_job_request, main as external_main, render_chunk_lrc, run_job_dir
from lrc_chunker.lrc import parse_lrc
from lrc_chunker.models import LyricLine
from lrc_chunker.motion_m0_extract import extract_m0_features
from lrc_chunker.utils import find_payload_vocals_path
from lrc_chunker.word_refine import PROFILES, _apply_breath_guard, build_parser as build_refine_parser, refine_payload


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

    rows = parse_lrc(str(lrc_path))

    assert [row.text for row in rows] == [
        "Hello from the other side",
        "I must have called a thousand times",
    ]


def test_assign_aligned_words_to_lines_uses_lrc_tokens_for_word_text() -> None:
    lines = [
        LyricLine(line_id=0, timestamp=0.0, text="don't go"),
    ]
    aligned_words = [
        {"text": "dont", "start": 0.10, "end": 0.40, "probability": 0.9},
        {"text": "gooo", "start": 0.45, "end": 0.80, "probability": 0.8},
    ]

    words, records = _assign_aligned_words_to_lines(lines, aligned_words)

    assert [word.text for word in words] == ["don't", "go"]
    assert records[0]["word_count"] == 2


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

    lines = parse_lrc(str(lrc_path))
    words, _, backend_used = fallback_align_from_lrc(lines, audio_path="demo.wav")
    chunks = build_chunks(words, ChunkingConfig())
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


def test_format_lrc_timestamp_uses_millisecond_precision() -> None:
    assert format_lrc_timestamp(62.345) == "[01:02.345]"


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
