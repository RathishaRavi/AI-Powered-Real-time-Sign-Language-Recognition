"""
tts.py
Text-to-speech engine wrapper using pyttsx3 (fully offline, works on
Windows/macOS/Linux via SAPI5 / NSSpeechSynthesizer / espeak respectively).

pyttsx3's engine object is NOT safe to reuse across threads reliably in
all backends, so we create a fresh engine per call by default, and also
expose a background-thread `speak_async` for use inside the Flask app so
speech synthesis never blocks the request/response cycle or the video
stream generator.
"""

import threading
import queue
import config

_speech_queue = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _speak_now(text, rate=None, volume=None):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate or config.TTS_RATE)
        engine.setProperty("volume", volume or config.TTS_VOLUME)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as e:
        print(f"[TTS] Failed to synthesize speech: {e}")
        return False


def _worker_loop():
    while True:
        item = _speech_queue.get()
        if item is None:
            break
        text, rate, volume = item
        _speak_now(text, rate, volume)
        _speech_queue.task_done()


def _ensure_worker():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True)
            t.start()
            _worker_started = True


def speak(text, rate=None, volume=None, blocking=True):
    """Synchronously speak `text`. Blocks the calling thread until done."""
    if not text or not text.strip():
        return False
    if blocking:
        return _speak_now(text, rate, volume)
    return speak_async(text, rate, volume)


def speak_async(text, rate=None, volume=None):
    """Queue `text` to be spoken by a background worker thread, so the
    Flask request handler (or camera stream generator) returns immediately."""
    if not text or not text.strip():
        return False
    _ensure_worker()
    _speech_queue.put((text, rate, volume))
    return True


def save_to_file(text, output_path, rate=None, volume=None):
    """Render speech to a WAV/MP3 file instead of playing it live -- useful
    for the Flask app to generate a downloadable/playable audio clip that
    the browser's <audio> tag can fetch, since the server process's audio
    output is not accessible to a remote browser client."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate or config.TTS_RATE)
        engine.setProperty("volume", volume or config.TTS_VOLUME)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS] Failed to save speech to file: {e}")
        return False


if __name__ == "__main__":
    speak("Hello, this is the sign language system speaking.")
