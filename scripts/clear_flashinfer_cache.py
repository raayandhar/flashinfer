#!/usr/bin/env python3
"""
Utility to delete FlashInfer JIT/build caches that frequently get stale.

By default it removes:
  * ~/.cache/flashinfer
  * ./flashinfer-jit-cache
  * ./flashinfer-cubin

Additional directories can be passed via repeated --extra flags. Use --dry-run
to inspect what would be removed without touching disk.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_CACHE_DIRS: List[Path] = [
    Path("~/.cache/flashinfer"),
    Path("flashinfer-jit-cache"),
    Path("flashinfer-cubin"),
]


def resolve_targets(extra_dirs: Iterable[str]) -> List[Path]:
    seen = set()
    targets: List[Path] = []

    for raw_path in list(map(str, DEFAULT_CACHE_DIRS)) + list(extra_dirs):
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path

        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append(path.expanduser())
    return targets


def remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        print(f"[skip] {path} (missing)")
        return

    if dry_run:
        print(f"[dry-run] would remove {path}")
        return

    if path.is_file() or path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path)
    print(f"[removed] {path}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="PATH",
        help="additional cache directory to remove (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the directories that would be removed without deleting them",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    targets = resolve_targets(args.extra)
    if not targets:
        print("No cache directories configured.")
        return 0

    for target in targets:
        remove_path(target, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
