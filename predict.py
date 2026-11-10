"""
predict.py
Real-time gesture prediction engine. This module exposes a SignPredictor
class that:
  - loads the trained LSTM model + label encoder once
  - maintains a rolling buffer of the last SEQUENCE_LENGTH landmark frames
  - runs inference once the buffer is full
  - applies confidence thresholding + majority-vote smoothing

It is used both by app.py (Flask MJPEG stream) and can be run standalone
for a desktop OpenCV demo window (`python predict.py --standalone`).
"""

import os
import argparse
import pickle
import collections
import numpy as np

import config
from utils.helpers import extract_hand_landmarks, PredictionSmoother, FPSCounter


class SignPredictor:
    def __init__(self, model_path=None, encoder_path=None):
        self.model_path = model_path or config.MODEL_FILE
        self.encoder_path = encoder_path or config.LABEL_ENCODER_FILE
        self.model = None
        self.encoder = None
        self.buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
        self.smoother = PredictionSmoother()
        self.loaded = False
        self._try_load()

    def _try_load(self):
        """Attempt to load the model. If it doesn't exist yet (user hasn't
        trained), we fail gracefully so the Flask app can still boot and
        show a friendly 'model not trained yet' message instead of crashing."""
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            import tensorflow as tf  # local import: keeps app startup fast when unused
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
            self.loaded = True
        else:
            self.loaded = False

    def is_ready(self):
        return self.loaded

    def reset_buffer(self):
        self.buffer.clear()
        self.smoother.reset()

    def push_frame(self, multi_hand_landmarks, multi_handedness):
        """Add one frame of landmarks to the rolling window."""
        vec = extract_hand_landmarks(multi_hand_landmarks, multi_handedness)
        self.buffer.append(vec)
        return vec

    def has_hand_presence(self):
        """True if at least one recent frame contained a detected hand
        (i.e. buffer isn't all zeros), used to avoid predicting on empty
        frames when no hand is in view."""
        if not self.buffer:
            return False
        recent = np.array(self.buffer)
        return bool(np.any(np.abs(recent[-1]) > 1e-6))

    def predict(self):
        """
        Run inference if the buffer is full. Returns:
            (label, raw_confidence, smoothed_label, smoothed_confidence)
        or (None, 0.0, None, 0.0) if not enough frames yet / model missing.
        """
        if not self.loaded:
            return None, 0.0, None, 0.0

        if len(self.buffer) < config.SEQUENCE_LENGTH:
            return None, 0.0, None, 0.0

        sequence = np.array(self.buffer, dtype=np.float32)
        sequence = np.expand_dims(sequence, axis=0)  # (1, seq_len, features)

        probs = self.model.predict(sequence, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        label = self.encoder.inverse_transform([top_idx])[0]

        self.smoother.push(label, confidence)
        stable_label, stable_conf = self.smoother.get_stable_prediction()

        if confidence < config.PREDICTION_CONFIDENCE_THRESHOLD:
            return label, confidence, None, 0.0

        return label, confidence, stable_label, stable_conf

    def top_k(self, k=3):
        """Return the top-k (label, confidence) pairs for the current
        buffer state, useful for a 'suggestions' UI element."""
        if not self.loaded or len(self.buffer) < config.SEQUENCE_LENGTH:
            return []
        sequence = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
        probs = self.model.predict(sequence, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]
        return [(self.encoder.inverse_transform([i])[0], float(probs[i])) for i in top_indices]


def run_standalone_demo(camera_index=0):
    """Desktop OpenCV window demo (for local testing on a machine with a
    webcam). Not used by the Flask web app, which streams frames instead."""
    import cv2
    from utils.camera import HandTracker, VideoStream

    predictor = SignPredictor()
    if not predictor.is_ready():
        print("[ERROR] No trained model found. Run train_model.py first.")
        return

    tracker = HandTracker()
    stream = VideoStream(source=camera_index)
    fps_counter = FPSCounter()

    if not stream.is_opened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return

    print("Press 'q' to quit.")
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        fps_counter.tick()

        results = tracker.process(frame)
        predictor.push_frame(results.multi_hand_landmarks, results.multi_handedness)
        display = tracker.draw(frame.copy(), results)

        label, conf, stable_label, stable_conf = predictor.predict()
        text = f"{stable_label} ({stable_conf*100:.1f}%)" if stable_label else "..."
        cv2.putText(display, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {fps_counter.get_fps():.1f}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Sign Language Prediction", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.standalone:
        run_standalone_demo(args.camera)
    else:
        print("Use --standalone to run the desktop demo window, "
              "or start app.py for the web interface.")
