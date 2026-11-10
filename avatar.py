"""
avatar.py
Server-side support for the 3D avatar (Speech-to-Sign) feature.

IMPORTANT / HONEST SCOPE NOTE:
Full photorealistic sign-language playback (the kind produced by
professional motion-capture libraries such as those behind commercial ASL
avatars) requires a licensed motion corpus and is outside what can be
generated here. What this module DOES provide is a genuine, working
procedural animation system: for each gesture in config.GESTURES it
defines a compact keyframe descriptor (joint angles for shoulder, elbow,
wrist and a simple hand-open/closed pose over a normalized time axis).
static/js/avatar.js consumes this data with Three.js to drive a rigged
humanoid figure in the browser in real time. It is simplified but fully
functional, extensible, and clearly labeled as such in the UI.
"""

import config

# Each keyframe: t (0..1 normalized time), then joint angles in degrees.
# Joints animated: rShoulderX, rShoulderZ, rElbow, rWristRot, handOpen(0-1)
# A mirrored left-arm variant is derived automatically for two-handed signs.
DEFAULT_KEYFRAMES = [
    {"t": 0.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
    {"t": 0.5, "rShoulderX": -60, "rShoulderZ": 10, "rElbow": 70, "rWristRot": 20, "handOpen": 0.5},
    {"t": 1.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
]

# Per-gesture overrides give each sign a visually distinct motion so the
# avatar doesn't just play the same generic wave for every word.
GESTURE_KEYFRAMES = {
    "hello": [
        {"t": 0.0, "rShoulderX": -80, "rShoulderZ": 20, "rElbow": 90, "rWristRot": 0, "handOpen": 1},
        {"t": 0.25, "rShoulderX": -80, "rShoulderZ": 20, "rElbow": 90, "rWristRot": -30, "handOpen": 1},
        {"t": 0.5, "rShoulderX": -80, "rShoulderZ": 20, "rElbow": 90, "rWristRot": 30, "handOpen": 1},
        {"t": 0.75, "rShoulderX": -80, "rShoulderZ": 20, "rElbow": 90, "rWristRot": -30, "handOpen": 1},
        {"t": 1.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
    ],
    "thanks": [
        {"t": 0.0, "rShoulderX": -40, "rShoulderZ": 0, "rElbow": 130, "rWristRot": 0, "handOpen": 0.3},
        {"t": 0.5, "rShoulderX": -90, "rShoulderZ": 10, "rElbow": 40, "rWristRot": 0, "handOpen": 0.3},
        {"t": 1.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
    ],
    "yes": [
        {"t": 0.0, "rShoulderX": -50, "rShoulderZ": 0, "rElbow": 60, "rWristRot": 0, "handOpen": 0},
        {"t": 0.33, "rShoulderX": -50, "rShoulderZ": 0, "rElbow": 90, "rWristRot": 0, "handOpen": 0},
        {"t": 0.66, "rShoulderX": -50, "rShoulderZ": 0, "rElbow": 60, "rWristRot": 0, "handOpen": 0},
        {"t": 1.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
    ],
    "no": [
        {"t": 0.0, "rShoulderX": -45, "rShoulderZ": 10, "rElbow": 80, "rWristRot": -30, "handOpen": 0.4},
        {"t": 0.5, "rShoulderX": -45, "rShoulderZ": -10, "rElbow": 80, "rWristRot": 30, "handOpen": 0.4},
        {"t": 1.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
    ],
    "love": [
        {"t": 0.0, "rShoulderX": -70, "rShoulderZ": 20, "rElbow": 100, "rWristRot": 0, "handOpen": 0},
        {"t": 0.5, "rShoulderX": -60, "rShoulderZ": 10, "rElbow": 110, "rWristRot": 0, "handOpen": 0},
        {"t": 1.0, "rShoulderX": 0, "rShoulderZ": 0, "rElbow": 10, "rWristRot": 0, "handOpen": 1},
    ],
}


def get_keyframes(gesture_label):
    """Return the keyframe list for a gesture, falling back to a generic
    wave-like motion for any gesture without a hand-authored animation."""
    gesture_label = (gesture_label or "").lower()
    return GESTURE_KEYFRAMES.get(gesture_label, DEFAULT_KEYFRAMES)


def build_animation_sequence(gesture_labels, seconds_per_sign=1.2):
    """
    Build the full timeline payload sent to the browser for a list of
    gesture words, e.g. from speech.map_text_to_gestures().
    """
    sequence = []
    for i, label in enumerate(gesture_labels):
        sequence.append({
            "word": label,
            "start": round(i * seconds_per_sign, 3),
            "duration": seconds_per_sign,
            "keyframes": get_keyframes(label),
        })
    return {
        "total_duration": round(len(gesture_labels) * seconds_per_sign, 3),
        "signs": sequence,
    }


def available_gestures():
    return list(config.GESTURES)


if __name__ == "__main__":
    import json
    print(json.dumps(build_animation_sequence(["hello", "thanks", "i", "love", "you"]), indent=2))
