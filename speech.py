"""
speech.py
Speech-to-text using the SpeechRecognition library. Supports both
microphone capture (desktop use) and uploaded audio file transcription
(what the Flask web app actually uses, since browsers cannot hand the
server direct microphone access without WebRTC -- the /speech page
records audio in the browser via the MediaRecorder API and POSTs the
resulting blob to the server for transcription).

Also provides map_text_to_gestures(), which tokenizes recognized speech
and matches each word/synonym against config.GESTURES so the front end
can drive the 3D avatar (avatar.py + static/js/avatar.js) for the
Speech-to-Sign feature.
"""

import os
import config

# Small synonym table so everyday phrasing maps onto our limited
# gesture vocabulary (extend this as the gesture set grows).
SYNONYMS = {
    "hi": "hello", "hey": "hello",
    "thank": "thanks", "thankyou": "thanks", "thank_you": "thanks",
    "sorry": "sorry", "apologize": "sorry",
    "ok": "yes", "okay": "yes", "yeah": "yes", "yep": "yes",
    "nope": "no", "nah": "no",
    "me": "i", "myself": "i",
    "u": "you", "ya": "you",
    "assist": "help", "aid": "help",
    "great": "good", "nice": "good", "fine": "good",
    "terrible": "bad", "awful": "bad",
    "leave": "go", "going": "go",
    "house": "home",
    "food": "eat", "hungry": "eat",
    "water": "drink", "thirsty": "drink",
    "buddy": "friend", "pal": "friend",
    "halt": "stop", "wait": "stop",
    "again": "more", "extra": "more",
}


def transcribe_microphone(timeout=5, phrase_time_limit=8):
    """
    Desktop-only helper: listens on the default microphone and returns the
    recognized text (or None on failure). Requires PyAudio to be installed
    for microphone access, which is a common source of install friction on
    Windows -- see README.md for the exact install command.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        print("[Speech] SpeechRecognition is not installed.")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening... speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        print("[Speech] Listening timed out with no speech detected.")
        return None
    except sr.UnknownValueError:
        print("[Speech] Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"[Speech] Recognition service error: {e}")
        return None
    except Exception as e:
        print(f"[Speech] Unexpected error: {e}")
        return None


def transcribe_audio_file(file_path):
    """
    Transcribe an uploaded audio file (wav/aiff/flac natively; other
    formats such as webm/ogg from the browser's MediaRecorder should be
    converted to WAV first -- see utils note in app.py's /api/speech-to-text
    route, which uses pydub/ffmpeg for that conversion).
    Returns recognized text, or None on failure.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        print("[Speech] SpeechRecognition is not installed.")
        return None

    if not os.path.exists(file_path):
        print(f"[Speech] File not found: {file_path}")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("[Speech] Could not understand audio in file.")
        return None
    except sr.RequestError as e:
        print(f"[Speech] Recognition service error: {e}")
        return None
    except Exception as e:
        print(f"[Speech] Unexpected error transcribing file: {e}")
        return None


def map_text_to_gestures(text):
    """
    Tokenize free-form recognized speech and map each token onto the
    closest known gesture label (direct match, synonym match, or dropped
    if unrecognized). Returns an ordered list of gesture labels ready to
    hand to the avatar animation engine.
    """
    if not text:
        return []

    known = set(config.GESTURES)
    words = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()

    mapped = []
    for w in words:
        if w in known:
            mapped.append(w)
        elif w in SYNONYMS and SYNONYMS[w] in known:
            mapped.append(SYNONYMS[w])
        # unmapped words are silently dropped -- the avatar simply has no
        # sign for them yet; the raw recognized text is still shown in the UI
    return mapped


if __name__ == "__main__":
    sample = "Hi, thank you, I am going home"
    print("Recognized:", sample)
    print("Mapped gestures:", map_text_to_gestures(sample))
