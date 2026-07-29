"""Sorting a shoot's clips into folders by take.

Copying is the default and moving is opt-in, because the input here may be the
only copy of a shoot. A name collision gets a numeric suffix.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .takes import Take

_UNSAFE = '<>:"/\\|?*'


def sanitize(text: str, fallback: str = "take", limit: int = 60) -> str:
    """Make a string safe for a filename on Windows, macOS and Linux alike."""
    cleaned = "".join("-" if ch in _UNSAFE else ch for ch in text)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    cleaned = " ".join(cleaned.split()).strip(" .")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip(" .-_")
    return cleaned or fallback


def _unique(path: Path, used: set[Path]) -> Path:
    if path not in used and not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(2, 10000):
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if candidate not in used and not candidate.exists():
            return candidate
    raise FileExistsError(f"cannot find a free name near {path}")


@dataclass
class PlannedMove:
    source: Path
    target: Path

    @property
    def changed(self) -> bool:
        try:
            return self.source.resolve() != self.target.resolve()
        except OSError:
            return True


def plan(takes: list[Take], destination: str | Path, *, rename: bool = True,
         folder_template: str = "{index:02d}_{label}",
         file_template: str = "{label}_{clip}") -> list[PlannedMove]:
    """Return the destination for every clip before filesystem changes."""
    destination = Path(destination)
    moves: list[PlannedMove] = []
    used: set[Path] = set()
    for index, take in enumerate(takes, start=1):
        label = sanitize(take.take, fallback=take.take)
        folder = destination / sanitize(
            folder_template.format(index=index, label=label, take=take.take),
            fallback=take.take)
        for clip in take.clips:
            if rename:
                name = sanitize(
                    file_template.format(label=label, take=take.take,
                                         clip=clip.path.stem, index=index),
                    fallback=clip.path.stem)
                target = folder / f"{name}{clip.path.suffix}"
            else:
                target = folder / clip.path.name
            target = _unique(target, used)
            used.add(target)
            moves.append(PlannedMove(source=clip.path, target=target))
    return moves


def plan_prepend_ids(
    takes: list[Take],
) -> tuple[list[PlannedMove], dict[Path, list[str]]]:
    """Plan safe in-place names of the form ``TAKEID_original.ext``.

    A long-running recorder can contain chirps from several takes. Such files
    are returned in ``ambiguous`` for manual naming. Re-running the operation is
    idempotent for files already carrying their decoded ID.
    """
    ids_by_source: dict[Path, set[str]] = {}
    for take in takes:
        for clip in take.clips:
            ids_by_source.setdefault(clip.path, set()).add(take.take)

    ambiguous = {
        source: sorted(ids)
        for source, ids in ids_by_source.items()
        if len(ids) != 1
    }
    moves: list[PlannedMove] = []
    used: set[Path] = set()
    for source, ids in sorted(ids_by_source.items(), key=lambda item: str(item[0])):
        if source in ambiguous:
            continue
        take = next(iter(ids))
        prefix = f"{sanitize(take, fallback=take)}_"
        if source.name.upper().startswith(prefix.upper()):
            continue
        target = _unique(source.with_name(f"{prefix}{source.name}"), used)
        used.add(target)
        moves.append(PlannedMove(source=source, target=target))
    return moves, ambiguous


def apply(moves: list[PlannedMove], *, move: bool = False,
          dry_run: bool = False) -> list[PlannedMove]:
    """Carry out a plan.  Copies unless ``move`` is set."""
    done = []
    for item in moves:
        if not item.changed:
            continue
        if not dry_run:
            item.target.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(item.source), str(item.target))
            else:
                shutil.copy2(str(item.source), str(item.target))
        done.append(item)
    return done
