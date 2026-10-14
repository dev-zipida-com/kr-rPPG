"""Command line: `python -m krrppg_release build ...` and `python -m krrppg_release verify ...`."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from . import constants as C
from .bfdt import check_release_setting, release_keep_ratio
from .builder import BuildConfig, build_release, refresh_metadata
from .verify import verify_release

# Custodian-only source tree (<site>/<group>/pNNN/T0x/ + metadata/); never part of any release.
DEFAULT_SOURCE = Path(os.environ["KRRPPG_SOURCE"]) if os.environ.get("KRRPPG_SOURCE") else None
MAX_LISTED = 50


def _frame_size(text: str) -> Tuple[int, int]:
    try:
        width, height = (int(value) for value in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from exc
    return width, height


def _participants(text: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


BLOCK_SIZE_HELP = (f"BFDT block size in px: {C.BLOCK_SIZE} (default, paper setting) or 16 (optional); only the block "
                   "mean (DC) is kept, k_r = 0.02 for 8 and 1/256 for 16")


def _keep_ratio(parser: argparse.ArgumentParser, args: argparse.Namespace) -> float:
    keep_ratio = release_keep_ratio(args.block_size)
    try:
        check_release_setting(args.block_size, keep_ratio, args.frame_size)
    except ValueError as exc:
        parser.error(str(exc))
    return keep_ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m krrppg_release", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build or resume the release (the source is only read)")
    build.add_argument("--source", type=Path, default=DEFAULT_SOURCE, required=DEFAULT_SOURCE is None,
                       help="read-only source dataset (default: $KRRPPG_SOURCE)")
    build.add_argument("--metadata-dir", type=Path, help="source metadata folder (default: <source>/metadata)")
    build.add_argument("--output-dir", type=Path, required=True, help="KR-rPPG/ is created inside this folder")
    build.add_argument("--private-dir", type=Path, required=True,
                       help="ID mapping, session records and build report; never publish this folder")
    build.add_argument("--participants", type=_participants, default=(),
                       help="comma-separated site_pNNN keys or pNNN IDs (default: everyone)")
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--timestamps", choices=C.TIMESTAMP_MODES, default="relative",
                       help="relative: seconds since the first frame (default); absolute: source epoch times")
    build.add_argument("--metadata-only", action="store_true",
                       help="rewrite only KR-rPPG/metadata/ of an existing release (needs its ID mapping); no session "
                            "file is written and the report proves it (<private-dir>/metadata_report.json)")
    build.add_argument("--crf", type=Path, help="CRF workbook (default: <source>/$KRRPPG_CRF_FILENAME)")
    build.add_argument("--crf-ild", type=Path, help="workbook with the CRF ILD item (default: $KRRPPG_ILD_XLSX)")
    build.add_argument("--seed", type=int,
                       help="test builds only: seed the random p001.. numbering (ignored once a mapping exists)")
    build.add_argument("--frame-size", type=_frame_size, default=C.FRAME_SIZE)
    build.add_argument("--block-size", type=int, choices=C.RELEASE_BLOCK_SIZES, default=C.BLOCK_SIZE,
                       help=BLOCK_SIZE_HELP)
    build.add_argument("--protect", type=Path, action="append", default=[],
                       help="folder the build must never write to, e.g. another release (repeatable)")
    build.add_argument("--dry-run", action="store_true", help="list the sessions without writing anything")

    verify = commands.add_parser("verify", help="check a built release (read-only)")
    verify.add_argument("--release-root", type=Path, required=True, help="the KR-rPPG/ folder")
    verify.add_argument("--timestamps", choices=C.TIMESTAMP_MODES, default="relative")
    verify.add_argument("--crf", type=Path, help="CRF workbook for --source (default: <source>/$KRRPPG_CRF_FILENAME)")
    verify.add_argument("--crf-ild", type=Path, help="workbook with the CRF ILD item for --source")
    verify.add_argument("--frame-size", type=_frame_size, default=C.FRAME_SIZE)
    verify.add_argument("--block-size", type=int, choices=C.RELEASE_BLOCK_SIZES, default=C.BLOCK_SIZE,
                        help=BLOCK_SIZE_HELP)
    verify.add_argument("--partial", action="store_true", help="allow a subset of the released sessions")
    verify.add_argument("--no-video", action="store_true", help="skip decoding the videos")
    verify.add_argument("--source", type=Path, help="also compare every file with a fresh render of the source")
    verify.add_argument("--mapping", type=Path, help="private participant_id_map.csv (required with --source)")
    verify.add_argument("--workers", type=int, default=1, help="check sessions in parallel processes")
    verify.add_argument("--json", action="store_true", help="print the full report as JSON")
    return parser


def _print_list(prefix: str, items: List[str]) -> None:
    for item in items[:MAX_LISTED]:
        print(f"{prefix} {item}")
    if len(items) > MAX_LISTED:
        print(f"{prefix} ... {len(items) - MAX_LISTED} more")


def _run_build(args: argparse.Namespace, keep_ratio: float) -> int:
    cfg = BuildConfig(
        source_root=args.source,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        private_dir=args.private_dir,
        timestamps=args.timestamps,
        crf_xlsx=args.crf,
        ild_xlsx=args.crf_ild,
        id_order="random",  # sorted numbering would preserve hospital screening order
        seed=args.seed,
        participants=args.participants,
        workers=args.workers,
        block_size=args.block_size,
        keep_ratio=keep_ratio,
        expected_frame_size=args.frame_size,
        protected_roots=C.PROTECTED_ROOTS + tuple(str(path) for path in args.protect),
        dry_run=args.dry_run,
    )
    if args.metadata_only:
        return _run_metadata_refresh(cfg)
    report = build_release(cfg)
    if cfg.dry_run:
        print(f"dry run: {report['sessions_total']} sessions would be built")
        return 0
    keys = ("builder_version", "settings", "mapping_created", "sessions_total", "built", "skipped", "failed",
            "frames_total", "source_unchanged", "data_quality_rule_counts")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    _print_list("FAILED", [f"{f['session']}: {f['error']}" for f in report["failures"]])
    _print_list("STALE (not a released session)", report["stale_entries"])
    _print_list("WARNING", report["warnings"])
    _print_list("SOURCE CHANGED DURING BUILD", report["source_changes"])
    print(f"release: {cfg.release_root}\nreport:  {cfg.report_path}")
    clean = report["failed"] == 0 and report["source_unchanged"] and not report["stale_entries"]
    return 0 if clean else 1


def _run_metadata_refresh(cfg: BuildConfig) -> int:
    report = refresh_metadata(cfg)
    if cfg.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    keys = ("builder_version", "mode", "metadata_written", "metadata_unchanged", "release_entries_watched",
            "release_unchanged_outside_metadata", "source_unchanged", "data_quality_rule_counts")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    _print_list("RELEASE CHANGED OUTSIDE metadata/", report["release_changes_outside_metadata"])
    _print_list("SOURCE CHANGED DURING REFRESH", report["source_changes"])
    print(f"release: {cfg.release_root}\nreport:  {cfg.metadata_report_path}")
    return 0 if report["release_unchanged_outside_metadata"] and report["source_unchanged"] else 1


def _run_verify(args: argparse.Namespace, keep_ratio: float) -> int:
    report = verify_release(
        args.release_root,
        timestamps=args.timestamps,
        expected_frame_size=args.frame_size,
        block_size=args.block_size,
        keep_ratio=keep_ratio,
        partial=args.partial,
        check_video=not args.no_video,
        source_root=args.source,
        mapping_path=args.mapping,
        workers=args.workers,
        crf_xlsx=args.crf,
        ild_xlsx=args.crf_ild,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"participants={report.participants} sessions={report.sessions} frames={report.frames} "
              f"errors={len(report.errors)} warnings={len(report.warnings)} block_size={args.block_size}")
        _print_list("ERROR", report.errors)
        _print_list("WARNING", report.warnings)
    return 0 if report.ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "metadata_only", False) and args.participants:
        parser.error("--metadata-only always rewrites the metadata of every participant; drop --participants")
    keep_ratio = _keep_ratio(parser, args)  # bad block sizes stop here, before anything is read or written
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "build":
        return _run_build(args, keep_ratio)
    return _run_verify(args, keep_ratio)
