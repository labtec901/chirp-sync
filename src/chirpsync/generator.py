"""Chirp generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import codec, css, frame


@dataclass
class GeneratedChirp:
    audio: np.ndarray
    sample_rate: int
    payload: codec.Payload
    layout: dict
    lead_in: float
    repeats: int
    gap: float

    @property
    def duration(self) -> float:
        return self.audio.size / self.sample_rate

    @property
    def sync_offsets(self) -> list[float]:
        """Seconds from the start of the rendered file to each sync reference."""
        span = self.layout["duration"] + self.gap
        return [self.lead_in + i * span for i in range(self.repeats)]

def generate(
    *,
    profile: css.Profile | str = css.DEFAULT_PROFILE,
    take_id: int | None = None,
    sample_rate: int = 48000,
    level_dbfs: float = -3.0,
    lead_in: float = 0.25,
    lead_out: float = 0.25,
    repeats: int = 1,
    gap: float = 0.25,
) -> GeneratedChirp:
    """Render a chirp burst.

    ``repeats`` emits the whole burst more than once with ``gap`` seconds of
    silence between; the parser decodes each independently and keeps the best,
    which is cheap insurance on a noisy set.
    """
    if isinstance(profile, str):
        profile = css.get_profile(profile)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    payload = codec.Payload(
        take_id=codec.new_take_id() if take_id is None else take_id,
    )

    items, layout = frame.frame_items(payload, profile)
    layout["profile"] = profile.name
    burst = css.synthesize(profile, items, sample_rate)

    gap_samples = np.zeros(int(round(gap * sample_rate)))
    pieces: list[np.ndarray] = [np.zeros(int(round(lead_in * sample_rate)))]
    for i in range(repeats):
        if i:
            pieces.append(gap_samples)
        pieces.append(burst)
    pieces.append(np.zeros(int(round(lead_out * sample_rate))))
    audio = np.concatenate(pieces)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (10 ** (level_dbfs / 20.0) / peak)

    return GeneratedChirp(
        audio=audio,
        sample_rate=sample_rate,
        payload=payload,
        layout=layout,
        lead_in=lead_in,
        repeats=repeats,
        gap=gap,
    )
