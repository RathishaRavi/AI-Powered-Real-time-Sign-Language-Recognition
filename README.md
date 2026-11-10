# AI-Powered Real-Time Sign Language Recognition System

A bidirectional communication platform between hearing-impaired and hearing
users: a webcam-based LSTM model recognizes hand signs in real time and
assembles them into grammatical sentences (with facial emotion context and
text-to-speech output), while a Speech-to-Sign pipeline transcribes spoken
input and drives a 3D avatar to sign it back.

Built as an MCA final year project. Tech stack: Python 3.11, Flask,
OpenCV, MediaPipe, TensorFlow/Keras (LSTM), DeepFace, spaCy, LanguageTool,
SpeechRecognition, pyttsx3, SQLite, Three.js.

> **Scope note:** the 3D avatar uses a procedural joint-angle rig (built in
> `avatar.py` / `static/js/avatar.js`), not a licensed motion-capture sign
> corpus. It demonstrates the full pipeline — recognized word → keyframes →
> real-time 3D playback — end to end, and is a natural place to swap in a
> production motion library or a rigged ReadyPlayerMe GLB avatar later.

---

## 1. Features

| Module | Description |
|---|---|
| Hand landmark detection | MediaPipe Hands, 21 landmarks × (x,y,z) × up to 2 hands = 126 features/frame |
| Face landmark detection | MediaPipe FaceMesh, used to crop the face region for emotion analysis |
| Sign recognition | 3-layer LSTM (64→128→64) trained on 30-frame landmark sequences |
| Emotion recognition | DeepFace (happy/sad/angry/fear/neutral/surprise/disgust), offline-safe fallback |
| Sentence generation | Rule-based gloss→English conversion + spaCy + LanguageTool grammar correction |
| Text-to-speech | pyttsx3, fully offline |
| Speech-to-sign | Browser mic recording → SpeechRecognition (Google Web Speech API) → gesture matching → 3D avatar animation |
| Web interface | Flask app: Dashboard, Camera, Prediction, Emotion, Speech, Avatar, History, Settings |
| Dataset tool | `collect_dataset.py` — webcam-based recorder with multi-user support and automatic augmentation |
| Training pipeline | `train_model.py` — full train/val/test split, accuracy & loss plots, confusion matrix, classification report |
| Database | SQLite — predictions, generated sentences, dataset sample index, settings |

---

## 2. Project Structure

```
AI_Sign_Language_System/
├── app.py                  # Flask application (all routes + MJPEG stream)
├── config.py                # Central configuration (paths, hyperparameters, gesture list)
├── database.py               # SQLite schema + CRUD helpers
├── collect_dataset.py        # Webcam dataset recorder
├── train_model.py            # LSTM training pipeline
├── predict.py                # Real-time inference engine
├── emotion.py                 # DeepFace emotion recognition wrapper
├── nlp.py                     # Gloss → sentence generation (spaCy + LanguageTool)
├── tts.py                     # pyttsx3 text-to-speech wrapper
├── speech.py                  # SpeechRecognition wrapper + gesture matching
├── avatar.py                  # Server-side keyframe generator for the 3D avatar
├── requirements.txt
├── README.md
├── utils/
│   ├── helpers.py             # Landmark extraction, sequence normalization, augmentation, smoothing
│   └── camera.py               # OpenCV + MediaPipe camera wrappers
├── templates/                  # Jinja2 HTML (Dashboard, Camera, Prediction, History, Settings, Speech, Avatar, Emotion)
├── static/
│   ├── css/style.css
│   └── js/{main.js, avatar.js}
├── dataset/                    # Recorded gesture sequences (created at runtime)
├── saved_models/               # Trained model, label encoder, plots (created at runtime)
└── logs/
```

---

## 3. Installation (Windows / macOS / Linux)

### 3.1 Prerequisites
- Python 3.11 (https://www.python.org/downloads/)
- A webcam and (optionally) a microphone
- Java Runtime Environment 8+ (only required for LanguageTool grammar correction)

### 3.2 Set up a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Windows note on PyAudio:** if `pip install pyaudio` fails to build, run:
```bash
pip install pipwin
pipwin install pyaudio
```

**TensorFlow / MediaPipe / OpenCV** install automatically via
`requirements.txt`; no extra steps are needed on a typical Windows 10/11 or
macOS/Linux machine with 64-bit Python 3.11.

### 3.4 Initialize the database

```bash
python database.py
```

---

## 4. Usage

### 4.1 Collect a dataset

```bash
# List configured gestures (edit config.py -> GESTURES to customize)
python collect_dataset.py --list

# Record one gesture, 30 takes, for a given user
python collect_dataset.py --gesture hello --user user1 --takes 30

# Record every configured gesture in one session
python collect_dataset.py --all --user user1 --takes 30
```
Controls: press **s** to start a take (records `SEQUENCE_LENGTH` frames
automatically), **q** to quit. Each real take also generates 3 augmented
copies (jitter/scale/shift) by default (`--augment N` to change).

### 4.2 Train the model

```bash
python train_model.py
```
Outputs, all written to `saved_models/`:
- `sign_lstm_model.h5` — trained Keras model
- `label_encoder.pkl` — gesture label encoder
- `accuracy_plot.png`, `loss_plot.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `training_history.json`

### 4.3 Run the web app

```bash
python app.py
```
Open **http://127.0.0.1:5000** in a browser.

### 4.4 Standalone desktop prediction demo (optional)

```bash
python predict.py --standalone
```

---

## 5. Pages

- **Dashboard** — system status, dataset summary, recent activity
- **Camera** — live MJPEG feed with real-time gesture + emotion overlay, auto-assembled sentences, TTS playback
- **Prediction** — full prediction log with confidence bars
- **Emotion** — live facial emotion analysis view
- **Speech** — record or type text → transcription → matched signs → avatar
- **Avatar** — 3D procedural avatar player (Three.js)
- **History** — filterable prediction/sentence history, clear-all option
- **Settings** — TTS rate, confidence threshold, camera index

---

## 6. Model Architecture

```
Input (30 frames × 126 features)
  → Masking
  → LSTM(64, return_sequences=True) → Dropout(0.3)
  → LSTM(128, return_sequences=True) → Dropout(0.3)
  → LSTM(64)
  → Dense(64, relu)
  → Dense(32, relu)
  → Dense(num_classes, softmax)

Optimizer: Adam (lr=0.001)
Loss: Categorical Crossentropy
Metrics: Accuracy
Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
```

Each frame's 126 features = 2 hands × 21 MediaPipe landmarks × (x, y, z),
zero-padded when only one hand is visible.

---

## 7. Database Schema

- `predictions(id, date, time, gesture, confidence, emotion, emotion_confidence, sentence, created_at)`
- `sentences(id, raw_words, corrected_sentence, source, created_at)`
- `dataset_samples(id, gesture_label, user_id, sequence_path, num_frames, created_at)`
- `settings(key, value)`

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Camera page shows "No webcam detected" | Confirm no other app is using the camera; try `--camera 1` in `collect_dataset.py` or change `camera_index` in Settings |
| `ModuleNotFoundError: deepface` model download fails | Requires internet on first run to fetch weights; the app falls back to a neutral-emotion heuristic if unavailable |
| LanguageTool errors | Ensure a JRE is installed and on PATH; the app silently skips grammar correction if unavailable |
| Low prediction accuracy | Record more takes per gesture (50+), ensure consistent lighting/background, retrain |
| PyAudio build fails on Windows | Use `pipwin install pyaudio` (see Installation) |

---

## 9. Future Scope

- Swap the procedural avatar for a rigged glTF (ReadyPlayerMe) character driven by the same keyframe API
- Two-way continuous sentence-level sign recognition (beyond isolated-gesture classification)
- Mobile app client using the same Flask API
- Transformer-based (not LSTM) temporal model for larger vocabularies
- Multi-language sign support (ISL, BSL) alongside ASL-style gestures
- User accounts and personalized model fine-tuning

---

## 10. License / Academic Use

Built for academic demonstration as part of an MCA final year project.
Third-party libraries retain their own licenses (Apache 2.0 for MediaPipe/TensorFlow, MIT for Flask/Three.js, etc.).
