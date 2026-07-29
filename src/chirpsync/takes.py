"""Turn raw take-ID detections into synchronized takes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClipDetection:
    """One take-ID chirp heard in one file."""

    path: Path
    take: str
    take_id: int
    time: float                  # seconds into the clip
    score: float
    clarity_db: float
    direct_ratio: float
    duration: float = 0.0
    has_video: bool = False
    frame_rate: float = 0.0
    width: int = 0
    height: int = 0

    def as_dict(self) -> dict:
        return {
            "file": str(self.path),
            "name": self.path.name,
            "take": self.take,
            "sync_time": round(self.time, 6),
            "score": round(self.score, 4),
            "clarity_db": round(self.clarity_db, 1),
            "direct_ratio": round(self.direct_ratio, 3),
            "duration": round(self.duration, 3),
            "has_video": self.has_video,
            "frame_rate": round(self.frame_rate, 6) if self.frame_rate else None,
        }


@dataclass
class Take:
    """Every clip that heard the same take ID, and how to line them up."""

    take: str
    take_id: int
    clips: list[ClipDetection] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.take

    def offsets(self) -> dict[Path, float]:
        """Where each clip sits on a shared timeline, in seconds.

        The chirp is one instant in the world, so a clip that heard it later
        into its own recording must have started earlier. Shifting everything
        by the largest sync time keeps every offset at or above zero, which is
        what timelines want.
        """
        if not self.clips:
            return {}
        latest = max(c.time for c in self.clips)
        return {c.path: latest - c.time for c in self.clips}

    def as_dict(self) -> dict:
        offs = self.offsets()
        return {
            "take": self.take,
            "clip_count": len(self.clips),
            "clips": [dict(c.as_dict(), timeline_offset=round(offs[c.path], 6))
                      for c in self.clips],
        }


def build_takes(clips: list[ClipDetection]) -> list[Take]:
    """Group clip detections by their decoded take ID."""
    groups: dict[int, Take] = {}
    for clip in sorted(clips, key=lambda c: (c.take_id, str(c.path))):
        take = groups.get(clip.take_id)
        if take is None:
            take = Take(take=clip.take, take_id=clip.take_id)
            groups[clip.take_id] = take
        take.clips.append(clip)

    out = list(groups.values())
    for take in out:
        take.clips.sort(key=lambda c: str(c.path))
    out.sort(key=lambda t: t.take)
    return out
