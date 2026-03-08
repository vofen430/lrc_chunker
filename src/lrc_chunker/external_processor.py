from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .alignment import AlignmentConfig, align_lyrics, build_alignment_payload
from .baseline import _default_denoiser_output
from .chunking import ChunkingConfig, build_chunks
from .lrc import parse_lrc
from .utils import (
    atomic_write_json,
    atomic_write_text,
    find_payload_vocals_path,
    normalize_ws,
    read_json,
    safe_stem,
    write_json,
)
from .word_refine import refine_payload


PROTOCOL_VERSION = 1
HEARTBEAT_INTERVAL_SEC = 10.0


class ExternalProcessorError(RuntimeError):
    def __init__(self, error_code: str, message: str, *, exit_code: int = 10) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.exit_code = int(exit_code)


class JobCancelledError(ExternalProcessorError):
    def __init__(self, message: str = "Cancellation requested by caller.") -> None:
        super().__init__("CANCELLED_BY_USER", message, exit_code=0)


@dataclass
class PairJob:
    row_index: int
    row_id: int
    lrc_path: str
    audio_path: str
    result_lrc_path: str
    title: str = ""
    artist: str = ""


@dataclass
class CallbackPaths:
    status_file: Path
    result_file: Optional[Path]
    complete_flag: Optional[Path]
    failed_flag: Optional[Path]
    cancel_flag: Optional[Path]


@dataclass
class JobRequest:
    job_id: str
    job_dir: Path
    mode: str
    pairs: List[PairJob]
    callbacks: CallbackPaths
    options: Dict[str, object]
    request_payload: dict
    result_dir: Optional[Path] = None


def _pair_detail(pair: PairJob) -> Dict[str, object]:
    return {
        "row_index": pair.row_index,
        "row_id": pair.row_id,
        "title": pair.title,
        "artist": pair.artist,
        "lrc_path": pair.lrc_path,
        "audio_path": pair.audio_path,
        "result_lrc_path": pair.result_lrc_path,
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coerce_abs_path(value: object, *, field_name: str, required: bool = True) -> Path:
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise ExternalProcessorError("REQUEST_INVALID", f"Missing required path: {field_name}", exit_code=10)
        return Path()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ExternalProcessorError("REQUEST_INVALID", f"{field_name} must be an absolute path", exit_code=10)
    return path


def _coerce_optional_abs_path(value: object, *, field_name: str) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    return _coerce_abs_path(raw, field_name=field_name, required=True)


def _require_protocol_version(payload: dict) -> None:
    if int(payload.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ExternalProcessorError("UNSUPPORTED_PROTOCOL_VERSION", "Unsupported protocol version.", exit_code=15)


def _validate_existing_file(path: Path, error_code: str) -> None:
    if not path.is_file():
        raise ExternalProcessorError(error_code, f"Missing input file: {path}", exit_code=11)


def _validate_writable_parent(path: Path) -> None:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ExternalProcessorError("OUTPUT_PATH_INVALID", f"Unable to create output directory: {parent}", exit_code=12) from exc
    if not os.access(parent, os.W_OK):
        raise ExternalProcessorError("OUTPUT_PATH_INVALID", f"Output directory is not writable: {parent}", exit_code=12)


def _load_batch_manifest(path: Path, *, result_dir: Path) -> List[PairJob]:
    payload = read_json(path)
    _require_protocol_version(payload)
    if str(payload.get("manifest_type") or "") != "ae_lrc_batch_pairs":
        raise ExternalProcessorError("REQUEST_INVALID", "Unsupported batch manifest type.", exit_code=10)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ExternalProcessorError("REQUEST_INVALID", "batch_pairs.json items must be an array.", exit_code=10)
    jobs: List[PairJob] = []
    for item in items:
        if not isinstance(item, dict):
            raise ExternalProcessorError("REQUEST_INVALID", "Each batch item must be an object.", exit_code=10)
        if str(item.get("pair_state") or "") != "ready":
            continue
        row_index = int(item.get("row_index", len(jobs) + 1))
        row_id = int(item.get("row_id", row_index))
        lrc_path = _coerce_abs_path(item.get("lrc_path"), field_name=f"items[{row_index}].lrc_path")
        audio_path = _coerce_abs_path(item.get("audio_path"), field_name=f"items[{row_index}].audio_path")
        _validate_existing_file(lrc_path, "INPUT_LRC_MISSING")
        _validate_existing_file(audio_path, "INPUT_AUDIO_MISSING")
        result_name = f"{row_index:04d}_{safe_stem(str(lrc_path))}.lrc"
        result_path = result_dir / result_name
        _validate_writable_parent(result_path)
        jobs.append(
            PairJob(
                row_index=row_index,
                row_id=row_id,
                lrc_path=str(lrc_path),
                audio_path=str(audio_path),
                result_lrc_path=str(result_path),
                title=str(item.get("title") or ""),
                artist=str(item.get("artist") or ""),
            )
        )
    return jobs


def load_job_request(job_dir: str | Path) -> JobRequest:
    root = Path(job_dir).expanduser().resolve()
    request_path = root / "request.json"
    if not request_path.is_file():
        raise ExternalProcessorError("REQUEST_INVALID", f"request.json not found in {root}", exit_code=10)
    payload = read_json(request_path)
    _require_protocol_version(payload)
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise ExternalProcessorError("REQUEST_INVALID", "request.json missing job_id.", exit_code=10)

    callback = payload.get("callback", {})
    if not isinstance(callback, dict):
        raise ExternalProcessorError("REQUEST_INVALID", "callback must be an object.", exit_code=10)
    callbacks = CallbackPaths(
        status_file=_coerce_abs_path(callback.get("status_file"), field_name="callback.status_file"),
        result_file=_coerce_optional_abs_path(callback.get("result_file"), field_name="callback.result_file"),
        complete_flag=_coerce_optional_abs_path(callback.get("complete_flag"), field_name="callback.complete_flag"),
        failed_flag=_coerce_optional_abs_path(callback.get("failed_flag"), field_name="callback.failed_flag"),
        cancel_flag=_coerce_optional_abs_path(callback.get("cancel_flag"), field_name="callback.cancel_flag"),
    )

    input_cfg = payload.get("input", {})
    output_cfg = payload.get("output", {})
    options = payload.get("options", {})
    if not isinstance(input_cfg, dict) or not isinstance(output_cfg, dict) or not isinstance(options, dict):
        raise ExternalProcessorError("REQUEST_INVALID", "request.json input/output/options must be objects.", exit_code=10)

    mode = str(input_cfg.get("mode") or "single").strip().lower()
    pairs: List[PairJob] = []
    result_dir: Optional[Path] = None
    if mode == "batch_manifest":
        manifest_path = _coerce_abs_path(input_cfg.get("batch_manifest_path"), field_name="input.batch_manifest_path")
        result_dir = _coerce_abs_path(output_cfg.get("result_dir"), field_name="output.result_dir")
        _validate_existing_file(manifest_path, "REQUEST_INVALID")
        _validate_writable_parent(result_dir / "placeholder.txt")
        pairs = _load_batch_manifest(manifest_path, result_dir=result_dir)
    else:
        lrc_path = _coerce_abs_path(input_cfg.get("lrc_path"), field_name="input.lrc_path")
        audio_path = _coerce_abs_path(input_cfg.get("audio_path"), field_name="input.audio_path")
        result_lrc_path = _coerce_abs_path(output_cfg.get("result_lrc_path"), field_name="output.result_lrc_path")
        _validate_existing_file(lrc_path, "INPUT_LRC_MISSING")
        _validate_existing_file(audio_path, "INPUT_AUDIO_MISSING")
        _validate_writable_parent(result_lrc_path)
        pairs = [
            PairJob(
                row_index=1,
                row_id=1,
                lrc_path=str(lrc_path),
                audio_path=str(audio_path),
                result_lrc_path=str(result_lrc_path),
            )
        ]

    for path in (
        callbacks.status_file,
        callbacks.result_file,
        callbacks.complete_flag,
        callbacks.failed_flag,
    ):
        if path is not None:
            _validate_writable_parent(path)

    return JobRequest(
        job_id=job_id,
        job_dir=root,
        mode=mode,
        pairs=pairs,
        callbacks=callbacks,
        options=options,
        request_payload=payload,
        result_dir=result_dir,
    )


def format_lrc_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    minutes, ms_rem = divmod(total_ms, 60000)
    whole_sec, millis = divmod(ms_rem, 1000)
    return f"[{minutes:02d}:{whole_sec:02d}.{millis:03d}]"


def render_chunk_lrc(payload: dict) -> Tuple[str, List[str]]:
    line_timestamps: Dict[int, float] = {}
    for raw in payload.get("lines", []) or []:
        try:
            line_timestamps[int(raw.get("line_id"))] = float(raw.get("timestamp"))
        except (TypeError, ValueError):
            continue

    seen_line_ids: set[int] = set()
    warnings: List[str] = []
    last_ts = 0.0
    out_lines: List[str] = []
    for chunk in payload.get("chunks", []) or []:
        text = normalize_ws(str(chunk.get("text") or ""))
        if not text:
            continue
        words = chunk.get("words", []) or []
        first_line_id: Optional[int] = None
        for word in words:
            try:
                first_line_id = int(word.get("line_id"))
                break
            except (TypeError, ValueError):
                continue
        if first_line_id is None:
            line_ids = chunk.get("line_ids", []) or []
            if line_ids:
                try:
                    first_line_id = int(line_ids[0])
                except (TypeError, ValueError):
                    first_line_id = None

        timestamp = float(chunk.get("start", 0.0))
        if first_line_id is not None and first_line_id in line_timestamps and first_line_id not in seen_line_ids:
            timestamp = float(line_timestamps[first_line_id])
            seen_line_ids.add(first_line_id)
        if timestamp < last_ts:
            warnings.append(f"non_monotonic_chunk_timestamp:{chunk.get('chunk_id', len(out_lines))}")
            timestamp = last_ts
        out_lines.append(f"{format_lrc_timestamp(timestamp)}{text}")
        last_ts = timestamp
    return "\n".join(out_lines) + ("\n" if out_lines else ""), warnings


def _pair_work_dir(base_dir: Path, pair: PairJob) -> Path:
    return base_dir / f"{pair.row_index:04d}_{safe_stem(pair.lrc_path)}"


def process_pair_to_lrc(pair: PairJob, *, options: Dict[str, object], work_dir: Path) -> Dict[str, object]:
    lines = parse_lrc(pair.lrc_path)
    if not lines:
        raise ExternalProcessorError("LRC_PARSE_FAILED", f"No usable lyric rows found in {pair.lrc_path}", exit_code=10)

    work_dir.mkdir(parents=True, exist_ok=True)
    language = str(options.get("language") or "en").strip()
    if language.lower() == "auto":
        language = "en"
    profile = str(options.get("profile") or "slow_attack").strip() or "slow_attack"
    artifacts_dir = work_dir / "artifacts"
    align_config = AlignmentConfig(
        model=str(options.get("model") or "small.en"),
        language=language,
        vad_threshold=float(options.get("vad_threshold") or 0.35),
        denoiser=str(options.get("denoiser") or "auto"),
        denoiser_output_path=str(
            options.get("denoiser_output_path")
            or _default_denoiser_output(str(artifacts_dir), pair.audio_path, str(options.get("denoiser") or "auto"))
        ),
        alignment_backend=str(options.get("alignment_backend") or "stable_ts"),
        allow_lrc_fallback=bool(options.get("allow_lrc_fallback", False)),
        only_voice_freq=bool(options.get("only_voice_freq", False)),
    )
    chunk_config = ChunkingConfig(
        max_gap=float(options.get("max_gap") or 0.35),
        merge_gap=float(options.get("merge_gap") or 0.12),
        max_chars=int(options.get("max_chars") or 42),
        max_words=int(options.get("max_words") or 6),
        max_dur=float(options.get("max_dur") or 3.2),
        hard_max_chunk_dur=float(options.get("hard_max_chunk_dur") or 6.0),
        rhythm_weight=float(options.get("rhythm_weight") or 2.8),
        hard_line_breaks=bool(options.get("hard_line_breaks", True)),
        emphasize_long_words=bool(options.get("emphasize_long_words", True)),
        long_word_single_threshold=float(options.get("long_word_single_threshold") or 0.78),
        long_word_bonus=float(options.get("long_word_bonus") or 2.6),
        apply_clamp_max=bool(options.get("apply_clamp_max", True)),
    )

    words, _, backend_used = align_lyrics(pair.audio_path, lines, align_config)
    chunks = build_chunks(words, chunk_config)
    payload = build_alignment_payload(
        audio_path=pair.audio_path,
        lrc_path=pair.lrc_path,
        lines=lines,
        words=words,
        chunks=chunks,
        config=align_config,
        backend_used=backend_used,
    )
    payload.setdefault("meta", {})
    payload["meta"]["external_pair"] = {
        "row_index": pair.row_index,
        "row_id": pair.row_id,
        "title": pair.title,
        "artist": pair.artist,
    }
    write_json(work_dir / "alignment.json", payload)

    refined, refine_report = refine_payload(
        payload,
        profile=profile,
        audio_mix=pair.audio_path,
        audio_vocals=find_payload_vocals_path(payload),
        use_lrc_anchors=bool(options.get("use_lrc_anchors", True)),
        lrc_anchor_window=float(options.get("lrc_anchor_window") or 0.18),
        lrc_anchor_weight=float(options.get("lrc_anchor_weight") or 3.5),
        lrc_anchor_keep_weight=float(options.get("lrc_anchor_keep_weight") or 0.30),
        lrc_anchor_min_delta=float(options.get("lrc_anchor_min_delta") or 0.04),
        lrc_anchor_span_words=int(options.get("lrc_anchor_span_words") or 1),
        lrc_anchor_max_ratio=float(options.get("lrc_anchor_max_ratio") or 0.15),
        sr=int(options.get("sr") or 22050),
        hop_length=int(options.get("hop_length") or 256),
        early_thr=float(options.get("early_thr") or 0.05),
        func_long_thr=float(options.get("func_long_thr") or 0.55),
    )
    write_json(work_dir / "refined.json", refined)
    write_json(work_dir / "refine_report.json", refine_report)

    lrc_text, warnings = render_chunk_lrc(refined)
    output_path = Path(pair.result_lrc_path)
    atomic_write_text(output_path, lrc_text)
    return {
        "row_index": pair.row_index,
        "row_id": pair.row_id,
        "title": pair.title,
        "artist": pair.artist,
        "lrc_path": pair.lrc_path,
        "audio_path": pair.audio_path,
        "result_lrc_path": str(output_path),
        "input_line_count": len(refined.get("lines", []) or []),
        "output_line_count": len([line for line in lrc_text.splitlines() if line.strip()]),
        "chunk_count": len(refined.get("chunks", []) or []),
        "warnings": warnings,
    }


class StatusTracker:
    def __init__(self, request: JobRequest) -> None:
        self.request = request
        self._lock = threading.Lock()
        now = _utc_now()
        self._status = {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": request.job_id,
            "state": "queued",
            "progress": 0,
            "stage": "validating",
            "message": "queued",
            "heartbeat_utc": now,
            "started_utc": now,
            "updated_utc": now,
            "result_lrc_path": "",
            "error_code": "",
            "error_message": "",
            "detail": {
                "mode": request.mode,
                "items_total": len(request.pairs),
                "items_completed": 0,
                "current_item": None,
                "last_completed_item": None,
                "current_summary": "queued",
                "eta_seconds": None,
                "eta_utc": None,
                "signals": {
                    "is_terminal": False,
                    "is_success": False,
                    "is_error": False,
                    "is_cancelled": False,
                },
                "result_overview": {
                    "result_dir": str(request.result_dir) if request.result_dir else "",
                    "result_lrc_path": "",
                    "warnings": [],
                    "metrics": {},
                    "items": [],
                },
            },
        }
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_monotonic = time.monotonic()

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._heartbeat_loop, name=f"lrc-job-{self.request.job_id}", daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SEC):
            with self._lock:
                self._status["heartbeat_utc"] = _utc_now()
                self._status["updated_utc"] = self._status["heartbeat_utc"]
                payload = dict(self._status)
            atomic_write_json(self.request.callbacks.status_file, payload)

    def write(self) -> None:
        with self._lock:
            self._status["heartbeat_utc"] = _utc_now()
            self._status["updated_utc"] = self._status["heartbeat_utc"]
            payload = dict(self._status)
        atomic_write_json(self.request.callbacks.status_file, payload)

    def update(self, *, state: Optional[str] = None, progress: Optional[int] = None, stage: Optional[str] = None, message: Optional[str] = None, result_lrc_path: Optional[str] = None, error_code: Optional[str] = None, error_message: Optional[str] = None, current_item: Optional[Dict[str, object]] = None, items_completed: Optional[int] = None, last_completed_item: Optional[Dict[str, object]] = None, result_overview: Optional[Dict[str, object]] = None) -> None:
        with self._lock:
            if state is not None:
                self._status["state"] = state
            if progress is not None:
                self._status["progress"] = max(0, min(100, int(progress)))
            if stage is not None:
                self._status["stage"] = stage
            if message is not None:
                self._status["message"] = message
            if result_lrc_path is not None:
                self._status["result_lrc_path"] = result_lrc_path
            if error_code is not None:
                self._status["error_code"] = error_code
            if error_message is not None:
                self._status["error_message"] = error_message
            detail = self._status["detail"]
            if current_item is not None:
                detail["current_item"] = current_item
            if items_completed is not None:
                detail["items_completed"] = max(0, int(items_completed))
            if last_completed_item is not None:
                detail["last_completed_item"] = last_completed_item
            if message is not None:
                detail["current_summary"] = message
            if result_overview is not None:
                detail["result_overview"] = result_overview
            total = max(0, int(detail.get("items_total", 0)))
            completed = max(0, int(detail.get("items_completed", 0)))
            elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            if completed > 0 and total > completed:
                avg = elapsed / completed
                eta_seconds = max(0, int(round(avg * (total - completed))))
                detail["eta_seconds"] = eta_seconds
                detail["eta_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + eta_seconds))
            elif total > 0 and completed >= total:
                detail["eta_seconds"] = 0
                detail["eta_utc"] = _utc_now()
            else:
                detail["eta_seconds"] = None
                detail["eta_utc"] = None
            signals = detail["signals"]
            current_state = self._status["state"]
            signals["is_terminal"] = current_state in {"completed", "failed", "cancelled"}
            signals["is_success"] = current_state == "completed"
            signals["is_error"] = current_state == "failed"
            signals["is_cancelled"] = current_state == "cancelled"
        self.write()

    def check_cancelled(self) -> None:
        if self.request.callbacks.cancel_flag is not None and self.request.callbacks.cancel_flag.exists():
            raise JobCancelledError()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _build_result_payload(*, request: JobRequest, state: str, result_lrc_path: str = "", result_dir: str = "", metrics: Dict[str, object], items: Optional[List[dict]] = None, warnings: Optional[List[str]] = None, error_code: str = "", error_message: str = "") -> dict:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "job_id": request.job_id,
        "state": state,
        "completed_utc": _utc_now(),
        "result_lrc_path": result_lrc_path,
        "warnings": warnings or [],
        "metrics": metrics,
        "error_code": error_code,
        "error_message": error_message,
    }
    if result_dir:
        payload["result_dir"] = result_dir
    if items is not None:
        payload["items"] = items
    return payload


def _write_flag(path: Path, text: str) -> None:
    atomic_write_text(path, text + "\n")


def _maybe_write_json(path: Optional[Path], payload: dict) -> None:
    if path is not None:
        atomic_write_json(path, payload)


def _maybe_write_flag(path: Optional[Path], text: str) -> None:
    if path is not None:
        _write_flag(path, text)


def run_job_dir(job_dir: str | Path) -> int:
    request = load_job_request(job_dir)
    tracker = StatusTracker(request)
    tracker.start()
    start_monotonic = time.monotonic()
    try:
        tracker.update(state="running", progress=1, stage="loading", message="loading request", items_completed=0)
        tracker.check_cancelled()
        items_out: List[dict] = []
        warnings: List[str] = []
        pair_total = max(1, len(request.pairs))
        work_root = request.job_dir / "work"
        for idx, pair in enumerate(request.pairs):
            tracker.check_cancelled()
            progress_base = int(idx * 100 / pair_total)
            tracker.update(
                state="running",
                progress=progress_base,
                stage="aligning",
                message=f"processing {idx + 1}/{pair_total}: {Path(pair.lrc_path).name}",
                current_item=_pair_detail(pair),
                items_completed=idx,
            )
            item_result = process_pair_to_lrc(pair, options=request.options, work_dir=_pair_work_dir(work_root, pair))
            items_out.append(item_result)
            warnings.extend(item_result.get("warnings", []))
            tracker.update(
                state="running",
                progress=int((idx + 1) * 100 / pair_total),
                stage="writing_output",
                message=f"wrote {Path(pair.result_lrc_path).name}",
                result_lrc_path=pair.result_lrc_path if request.mode != "batch_manifest" else "",
                current_item=_pair_detail(pair),
                items_completed=idx + 1,
                last_completed_item=item_result,
            )

        duration_sec = round(time.monotonic() - start_monotonic, 3)
        metrics = {
            "duration_sec": duration_sec,
            "item_count": len(request.pairs),
            "success_count": len(items_out),
            "failed_count": 0,
            "input_line_count": sum(int(item.get("input_line_count", 0)) for item in items_out),
            "output_line_count": sum(int(item.get("output_line_count", 0)) for item in items_out),
        }
        result_payload = _build_result_payload(
            request=request,
            state="completed",
            result_lrc_path=items_out[0]["result_lrc_path"] if request.mode != "batch_manifest" and items_out else "",
            result_dir=str(request.result_dir) if request.result_dir else "",
            metrics=metrics,
            items=items_out if request.mode == "batch_manifest" else items_out,
            warnings=warnings,
        )
        _maybe_write_json(request.callbacks.result_file, result_payload)
        tracker.update(
            state="completed",
            progress=100,
            stage="finalizing",
            message="completed",
            result_lrc_path=result_payload.get("result_lrc_path", ""),
            current_item=None,
            items_completed=len(items_out),
            result_overview={
                "result_dir": result_payload.get("result_dir", ""),
                "result_lrc_path": result_payload.get("result_lrc_path", ""),
                "warnings": result_payload.get("warnings", []),
                "metrics": result_payload.get("metrics", {}),
                "items": result_payload.get("items", []),
            },
        )
        _maybe_write_flag(request.callbacks.complete_flag, "completed")
        return 0
    except JobCancelledError as exc:
        result_payload = _build_result_payload(
            request=request,
            state="cancelled",
            metrics={"item_count": len(request.pairs)},
            error_code=exc.error_code,
            error_message=str(exc),
        )
        _maybe_write_json(request.callbacks.result_file, result_payload)
        tracker.update(
            state="cancelled",
            progress=100,
            stage="finalizing",
            message="cancelled",
            error_code=exc.error_code,
            error_message=str(exc),
            current_item=None,
            result_overview={
                "result_dir": result_payload.get("result_dir", ""),
                "result_lrc_path": result_payload.get("result_lrc_path", ""),
                "warnings": result_payload.get("warnings", []),
                "metrics": result_payload.get("metrics", {}),
                "items": result_payload.get("items", []),
            },
        )
        _maybe_write_flag(request.callbacks.failed_flag, "cancelled")
        return 0
    except ExternalProcessorError as exc:
        result_payload = _build_result_payload(
            request=request,
            state="failed",
            metrics={"item_count": len(request.pairs)},
            error_code=exc.error_code,
            error_message=str(exc),
        )
        _maybe_write_json(request.callbacks.result_file, result_payload)
        tracker.update(
            state="failed",
            progress=100,
            stage="finalizing",
            message="failed",
            error_code=exc.error_code,
            error_message=str(exc),
            current_item=None,
            result_overview={
                "result_dir": result_payload.get("result_dir", ""),
                "result_lrc_path": result_payload.get("result_lrc_path", ""),
                "warnings": result_payload.get("warnings", []),
                "metrics": result_payload.get("metrics", {}),
                "items": result_payload.get("items", []),
            },
        )
        _maybe_write_flag(request.callbacks.failed_flag, "failed")
        return int(exc.exit_code or 1)
    except Exception as exc:
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result_payload = _build_result_payload(
            request=request,
            state="failed",
            metrics={"item_count": len(request.pairs)},
            error_code="UNKNOWN_INTERNAL_ERROR",
            error_message=message,
        )
        _maybe_write_json(request.callbacks.result_file, result_payload)
        tracker.update(
            state="failed",
            progress=100,
            stage="finalizing",
            message="failed",
            error_code="UNKNOWN_INTERNAL_ERROR",
            error_message=message,
            current_item=None,
            result_overview={
                "result_dir": result_payload.get("result_dir", ""),
                "result_lrc_path": result_payload.get("result_lrc_path", ""),
                "warnings": result_payload.get("warnings", []),
                "metrics": result_payload.get("metrics", {}),
                "items": result_payload.get("items", []),
            },
        )
        _maybe_write_flag(request.callbacks.failed_flag, "failed")
        return 1
    finally:
        tracker.stop()


def _spawn_worker(job_dir: Path, stdout_log: Path, stderr_log: Path) -> None:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_dir)
    cmd = [sys.executable, "-m", "lrc_chunker.external_processor", "run-worker", "--job-dir", str(job_dir)]
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
        kwargs = {
            "stdout": stdout_handle,
            "stderr": stderr_handle,
            "stdin": subprocess.DEVNULL,
            "cwd": str(job_dir),
            "env": env,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kwargs)
        if proc.poll() is not None:
            raise ExternalProcessorError("WORKER_SPAWN_FAILED", "Worker exited during launch.", exit_code=14)


def _write_launch_status(request: JobRequest) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "job_id": request.job_id,
        "state": "queued",
        "progress": 0,
        "stage": "validating",
        "message": "launch accepted",
        "heartbeat_utc": _utc_now(),
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "result_lrc_path": "",
        "error_code": "",
        "error_message": "",
        "detail": {
            "mode": request.mode,
            "items_total": len(request.pairs),
            "items_completed": 0,
            "current_item": None,
            "last_completed_item": None,
            "current_summary": "launch accepted",
            "eta_seconds": None,
            "eta_utc": None,
            "signals": {
                "is_terminal": False,
                "is_success": False,
                "is_error": False,
                "is_cancelled": False,
            },
            "result_overview": {
                "result_dir": str(request.result_dir) if request.result_dir else "",
                "result_lrc_path": "",
                "warnings": [],
                "metrics": {},
                "items": [],
            },
        },
    }
    atomic_write_json(request.callbacks.status_file, payload)


def launch_job(job_dir: str | Path) -> int:
    request = load_job_request(job_dir)
    _write_launch_status(request)
    try:
        _spawn_worker(request.job_dir, request.job_dir / "stdout.log", request.job_dir / "stderr.log")
    except ExternalProcessorError:
        raise
    except ModuleNotFoundError as exc:
        raise ExternalProcessorError("DEPENDENCY_MISSING", str(exc), exit_code=13) from exc
    except Exception as exc:
        raise ExternalProcessorError("WORKER_SPAWN_FAILED", str(exc), exit_code=14) from exc
    return 0


def run_self_test() -> int:
    diagnostics: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version.split()[0],
        "modules": {},
    }
    required_modules = [
        ("stable_whisper", "stable-ts"),
        ("whisper", "openai-whisper"),
        ("torchaudio", "torchaudio"),
        ("librosa", "librosa"),
        ("imageio_ffmpeg", "imageio-ffmpeg"),
        ("numpy", "numpy"),
        ("demucs", "demucs"),
    ]
    exit_code = 0
    for module_name, label in required_modules:
        try:
            __import__(module_name)
            diagnostics["modules"][label] = "ok"
        except Exception as exc:
            diagnostics["modules"][label] = f"missing: {exc}"
            exit_code = 13
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    return exit_code


def _collect_folder_pairs(input_dir: Path, output_dir: Path) -> List[PairJob]:
    lrc_files = sorted(input_dir.glob("*.lrc"))
    pairs: List[PairJob] = []
    for idx, lrc_path in enumerate(lrc_files, start=1):
        stem = lrc_path.stem
        audio_path = input_dir / f"{stem}.wav"
        if not audio_path.is_file():
            audio_path = input_dir / f"{stem}.mp3"
        if not audio_path.is_file():
            continue
        pairs.append(
            PairJob(
                row_index=idx,
                row_id=idx,
                lrc_path=str(lrc_path.resolve()),
                audio_path=str(audio_path.resolve()),
                result_lrc_path=str((output_dir / f"{stem}.lrc").resolve()),
            )
        )
    return pairs


def run_batch_folder(input_dir: str, output_dir: str, options: Dict[str, object]) -> int:
    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _collect_folder_pairs(in_dir, out_dir)
    if not pairs:
        raise ExternalProcessorError("INPUT_AUDIO_MISSING", f"No matching audio found for any LRC in {in_dir}", exit_code=11)
    work_root = out_dir / "_work"
    results = [process_pair_to_lrc(pair, options=options, work_dir=_pair_work_dir(work_root, pair)) for pair in pairs]
    summary = {
        "processed": len(results),
        "results": results,
    }
    atomic_write_json(out_dir / "batch_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External LRC processor CLI for AE file-protocol jobs.")
    parser.add_argument("-A", "--ae", action="store_true", help="Enable AE protocol commands such as launch/run-worker.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_launch = subparsers.add_parser("launch", help="Validate request.json and spawn a detached worker.")
    p_launch.add_argument("--job-dir", required=True, type=str)

    p_worker = subparsers.add_parser("run-worker", help=argparse.SUPPRESS)
    p_worker.add_argument("--job-dir", required=True, type=str)

    subparsers.add_parser("version", help="Print machine-readable version.")
    subparsers.add_parser("self-test", help="Check runtime dependencies.")

    p_batch = subparsers.add_parser("batch-folder", help="Debug entry for batch processing a local folder.")
    p_batch.add_argument("--input-dir", required=True, type=str)
    p_batch.add_argument("--output-dir", required=True, type=str)
    p_batch.add_argument("--model", default="small.en")
    p_batch.add_argument("--language", default="en")
    p_batch.add_argument("--profile", default="slow_attack")
    p_batch.add_argument("--denoiser", default="auto")
    p_batch.add_argument("--use-lrc-anchors", dest="use_lrc_anchors", action="store_true", default=True)
    p_batch.add_argument("--no-lrc-anchors", dest="use_lrc_anchors", action="store_false")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.cmd == "version":
            print("0.1.0")
            return 0
        if args.cmd == "self-test":
            return run_self_test()
        if args.cmd in {"launch", "run-worker"} and not bool(args.ae):
            raise ExternalProcessorError("REQUEST_INVALID", "AE protocol commands require -A / --ae.", exit_code=10)
        if args.cmd == "launch":
            return launch_job(args.job_dir)
        if args.cmd == "run-worker":
            return run_job_dir(args.job_dir)
        if args.cmd == "batch-folder":
            options = {
                "model": args.model,
                "language": args.language,
                "profile": args.profile,
                "denoiser": args.denoiser,
                "use_lrc_anchors": args.use_lrc_anchors,
            }
            return run_batch_folder(args.input_dir, args.output_dir, options)
        raise ExternalProcessorError("REQUEST_INVALID", f"Unsupported command: {args.cmd}", exit_code=10)
    except ExternalProcessorError as exc:
        print(f"{exc.error_code}: {exc}", file=sys.stderr)
        return int(exc.exit_code or 1)
