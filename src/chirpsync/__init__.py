"""chirp-sync: acoustic take IDs for multi-camera shoots.

A short musical chirp encodes only a take ID. Play it once with all cameras
rolling; afterwards the parser recovers the ID from every clip that heard it
and reports, to well under a millisecond, exactly when it arrived -- which is
both the take grouping and the sync point.
"""

from ._version import __version__
from .codec import Payload, take_id_from_str, take_id_to_str
from .css import DEFAULT_PROFILE, PROFILES, Profile, get_profile
from .detector import Detection, detect
from .generator import GeneratedChirp, generate

__all__ = [
    "__version__",
    "Payload",
    "Profile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "get_profile",
    "take_id_to_str",
    "take_id_from_str",
    "Detection",
    "detect",
    "GeneratedChirp",
    "generate",
]
