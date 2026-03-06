from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrc_chunker.alignment import AlignmentConfig, build_alignment_payload, fallback_align_from_lrc
from lrc_chunker.chunking import ChunkingConfig, build_chunks
from lrc_chunker.lrc import parse_lrc
from lrc_chunker.motion_m0_extract import extract_m0_features
from lrc_chunker.word_refine import refine_payload


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
