"""
app.py
Main Flask application for the AI-Powered Real-Time Sign Language
Recognition System. Wires together every module:
  - camera.py / predict.py  -> MJPEG live prediction stream
  - emotion.py               -> facial emotion overlay
  - nlp.py                   -> word sequence -> sentence
  - tts.py                   -> sentence -> speech
  - speech.py / avatar.py    -> speech-to-sign avatar animation
  - database.py              -> history persistence

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import io
import time
import json
import tempfile
import threading

import cv2
import numpy as np
from flask import (
    Flask, Response, render_template, request, jsonify,
    send_from_directory, redirect, url_for, flash
)

import config
import database
import nlp
import tts
import speech
import avatar
from emotion import get_recognizer
from predict import SignPredictor
from utils.camera import HandTracker, FaceTracker, VideoStream
from utils.helpers import FPSCounter

app = Flask(__name__)
app.config.from_object(config.FlaskConfig)

database.init_db()

# ----------------------------------------------------------------------------
# Shared runtime state
# Flask's dev server can be multi-threaded, so camera/model objects that are
# expensive to create are built once and protected by a lock where mutated.
# ----------------------------------------------------------------------------
_state_lock = threading.Lock()
_predictor = SignPredictor()
_sentence_buffer = []           # words collected during the current "utterance"
_last_word_time = 0.0
_current_emotion = {"label": "neutral", "confidence": 0.0}
_last_stable_word = None


def _reset_sentence_buffer_if_idle():
    global _sentence_buffer, _last_word_time
    if _sentence_buffer and (time.time() - _last_word_time) > config.SENTENCE_WORD_GAP_SECONDS:
        words = list(_sentence_buffer)
        _sentence_buffer = []
        sentence = nlp.words_to_sentence(words)
        if sentence:
            database.insert_sentence(" ".join(words), sentence, source="sign")
        return sentence
    return None


def gen_prediction_frames():
    """MJPEG generator for the /video_feed route. Runs hand tracking, face
    tracking (throttled), emotion recognition (throttled), gesture
    prediction and overlays it all onto the frame."""
    global _last_word_time, _last_stable_word, _current_emotion

    hand_tracker = HandTracker()
    face_tracker = FaceTracker()
    emotion_recognizer = get_recognizer()
    stream = VideoStream(source=0)
    fps_counter = FPSCounter()
    frame_idx = 0

    if not stream.is_opened():
        # Yield a single friendly placeholder frame explaining the problem,
        # rather than silently failing / hanging the <img> tag forever.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No webcam detected on the server.", (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(blank, "Connect a camera and restart the app.", (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", blank)
        frame_bytes = buf.tobytes()
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1)

    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            fps_counter.tick()
            frame_idx += 1

            hand_results = hand_tracker.process(frame)
            _predictor.push_frame(hand_results.multi_hand_landmarks, hand_results.multi_handedness)
            display = hand_tracker.draw(frame.copy(), hand_results)

            if frame_idx % config.EMOTION_ANALYSIS_INTERVAL == 0:
                face_results = face_tracker.process(frame)
                bbox = face_tracker.get_face_bbox(frame, face_results)
                if bbox:
                    x, y, w, h = bbox
                    crop = frame[y:y + h, x:x + w]
                    label, conf = emotion_recognizer.analyze(crop)
                    with _state_lock:
                        _current_emotion = {"label": label, "confidence": round(conf, 3)}
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 200, 0), 2)

            label, conf, stable_label, stable_conf = _predictor.predict()

            if stable_label and stable_label != _last_stable_word:
                with _state_lock:
                    _sentence_buffer.append(stable_label)
                    _last_word_time = time.time()
                    _last_stable_word = stable_label
                database.insert_prediction(
                    stable_label, stable_conf,
                    emotion=_current_emotion["label"],
                    emotion_confidence=_current_emotion["confidence"],
                )

            finalized_sentence = _reset_sentence_buffer_if_idle()
            if finalized_sentence:
                tts.speak_async(finalized_sentence)

            # ---- overlay text ----
            y0 = 30
            if not _predictor.is_ready():
                cv2.putText(display, "Model not trained yet - run train_model.py",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif stable_label:
                cv2.putText(display, f"{stable_label} ({stable_conf*100:.0f}%)",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display, f"FPS: {fps_counter.get_fps():.1f}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Emotion: {_current_emotion['label']}", (450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            with _state_lock:
                sentence_preview = " ".join(_sentence_buffer)
            if sentence_preview:
                cv2.putText(display, sentence_preview, (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

            ok, buf = cv2.imencode(".jpg", display)
            if not ok:
                continue
            frame_bytes = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        hand_tracker.close()
        face_tracker.close()
        stream.release()


# ----------------------------------------------------------------------------
# PAGE ROUTES
# ----------------------------------------------------------------------------
@app.route("/")
def dashboard():
    predictions = database.get_all_predictions(limit=10)
    sentences = database.get_all_sentences(limit=5)
    dataset_summary = database.get_dataset_summary()
    return render_template(
        "dashboard.html",
        predictions=predictions,
        sentences=sentences,
        dataset_summary=dataset_summary,
        gestures=config.GESTURES,
        model_ready=_predictor.is_ready(),
    )


@app.route("/camera")
def camera_page():
    return render_template("camera.html", model_ready=_predictor.is_ready())


@app.route("/prediction")
def prediction_page():
    predictions = database.get_all_predictions(limit=50)
    return render_template("prediction.html", predictions=predictions,
                            model_ready=_predictor.is_ready())


@app.route("/history")
def history_page():
    date_filter = request.args.get("date")
    if date_filter:
        predictions = database.get_predictions_by_date(date_filter)
    else:
        predictions = database.get_all_predictions(limit=500)
    sentences = database.get_all_sentences(limit=100)
    return render_template("history.html", predictions=predictions, sentences=sentences,
                            date_filter=date_filter or "")


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        for key in ("tts_rate", "confidence_threshold", "camera_index"):
            value = request.form.get(key)
            if value is not None:
                database.set_setting(key, value)
        flash("Settings saved.", "success")
        return redirect(url_for("settings_page"))
    current = database.get_all_settings()
    return render_template("settings.html", settings=current, config=config)


@app.route("/speech")
def speech_page():
    return render_template("speech.html", gestures=config.GESTURES)


@app.route("/avatar")
def avatar_page():
    return render_template("avatar.html", gestures=config.GESTURES)


@app.route("/emotion")
def emotion_page():
    return render_template("emotion.html")


# ----------------------------------------------------------------------------
# STREAMING / API ROUTES
# ----------------------------------------------------------------------------
@app.route("/video_feed")
def video_feed():
    return Response(gen_prediction_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/current_prediction")
def api_current_prediction():
    with _state_lock:
        return jsonify({
            "sentence_buffer": list(_sentence_buffer),
            "emotion": _current_emotion,
            "model_ready": _predictor.is_ready(),
        })


@app.route("/api/finalize_sentence", methods=["POST"])
def api_finalize_sentence():
    """Manually flush the current word buffer into a sentence (e.g. user
    clicks 'Finish Sentence' in the UI instead of waiting for the idle gap)."""
    global _sentence_buffer
    with _state_lock:
        words = list(_sentence_buffer)
        _sentence_buffer = []
    if not words:
        return jsonify({"sentence": "", "words": []})
    sentence = nlp.words_to_sentence(words)
    database.insert_sentence(" ".join(words), sentence, source="sign")
    tts.speak_async(sentence)
    return jsonify({"sentence": sentence, "words": words})


@app.route("/api/speak", methods=["POST"])
def api_speak():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"ok": False, "error": "No text provided"}), 400
    tts.speak_async(text)
    return jsonify({"ok": True})


@app.route("/api/speech-to-text", methods=["POST"])
def api_speech_to_text():
    """Accepts an uploaded audio blob (recorded in-browser), transcribes it,
    maps the words onto known gestures, and returns an avatar animation
    payload for the Speech-to-Sign page."""
    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        text = speech.transcribe_audio_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text:
        return jsonify({"ok": False, "error": "Could not transcribe audio. "
                                                "Try a clearer WAV recording."}), 200

    gestures_found = speech.map_text_to_gestures(text)
    animation = avatar.build_animation_sequence(gestures_found)
    return jsonify({"ok": True, "text": text, "gestures": gestures_found, "animation": animation})


@app.route("/api/text-to-sign", methods=["POST"])
def api_text_to_sign():
    """Same as speech-to-sign but starting from typed text (useful when the
    browser lacks microphone permissions, or for quick testing)."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    gestures_found = speech.map_text_to_gestures(text)
    animation = avatar.build_animation_sequence(gestures_found)
    return jsonify({"ok": True, "text": text, "gestures": gestures_found, "animation": animation})


@app.route("/api/dataset-summary")
def api_dataset_summary():
    return jsonify(database.get_dataset_summary())


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    database.clear_predictions()
    return jsonify({"ok": True})


@app.route("/api/model-status")
def api_model_status():
    return jsonify({
        "ready": _predictor.is_ready(),
        "gestures": config.GESTURES,
        "model_path": config.MODEL_FILE,
    })


@app.route("/api/reload-model", methods=["POST"])
def api_reload_model():
    """Hot-reload the model after retraining, without restarting the Flask
    process (handy during iterative development)."""
    global _predictor
    _predictor = SignPredictor()
    return jsonify({"ok": True, "ready": _predictor.is_ready()})


# ----------------------------------------------------------------------------
# STATIC / MISC
# ----------------------------------------------------------------------------
@app.route("/saved_models/<path:filename>")
def saved_models_static(filename):
    return send_from_directory(config.SAVED_MODELS_DIR, filename)


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
