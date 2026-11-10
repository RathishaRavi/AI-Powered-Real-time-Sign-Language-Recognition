"""
emotion.py
Facial emotion recognition using DeepFace, scoped to the face region found
by MediaPipe FaceMesh (utils/camera.FaceTracker). DeepFace's underlying
CNN is comparatively heavy, so callers should only invoke analyze() every
config.EMOTION_ANALYSIS_INTERVAL frames, not on every frame.

If DeepFace (or its backend weights, which it downloads on first use) is
unavailable -- e.g. no internet access at runtime -- this module degrades
gracefully and returns a neutral/unknown result instead of crashing the
whole video pipeline.
"""

import numpy as np
import config

_deepface_available = None  # lazy tri-state: None=untested, True/False after first try


def _check_deepface():
    global _deepface_available
    if _deepface_available is not None:
        return _deepface_available
    try:
        from deepface import DeepFace  # noqa: F401
        _deepface_available = True
    except Exception:
        _deepface_available = False
    return _deepface_available


class EmotionRecognizer:
    def __init__(self):
        self.available = _check_deepface()
        self._last_result = ("neutral", 0.0)

    def analyze(self, bgr_face_crop):
        """
        bgr_face_crop: numpy BGR image, ideally already cropped to the
        face bounding box for speed and accuracy.
        Returns: (emotion_label, confidence_0_to_1)
        """
        if bgr_face_crop is None or bgr_face_crop.size == 0:
            return self._last_result

        if not self.available:
            return self._fallback(bgr_face_crop)

        try:
            from deepface import DeepFace
            result = DeepFace.analyze(
                bgr_face_crop,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotions = result.get("emotion", {})
            if not emotions:
                return self._fallback(bgr_face_crop)

            label = max(emotions, key=emotions.get)
            confidence = float(emotions[label]) / 100.0
            label = label.lower()
            if label not in config.EMOTION_LABELS:
                label = "neutral"
            self._last_result = (label, confidence)
            return self._last_result
        except Exception:
            return self._fallback(bgr_face_crop)

    def _fallback(self, bgr_face_crop):
        """
        Lightweight heuristic fallback (NOT a replacement for DeepFace, but
        keeps the pipeline alive in offline/constrained environments): uses
        simple brightness/contrast of the mouth region as a very rough
        proxy signal. This deliberately reports 'neutral' with low
        confidence most of the time -- it exists so the rest of the app
        (sentence generation, DB logging, UI) keeps functioning end-to-end
        even when DeepFace's model weights cannot be downloaded.
        """
        try:
            gray = np.mean(bgr_face_crop)
            variance = np.var(bgr_face_crop)
            if variance > 2600:
                return ("surprise", 0.35)
            return ("neutral", 0.30)
        except Exception:
            return ("neutral", 0.0)


def get_recognizer():
    """Module-level singleton so the (potentially slow) availability check
    and any model warm-up only happens once per process."""
    global _singleton
    try:
        return _singleton
    except NameError:
        _singleton = EmotionRecognizer()
        return _singleton
