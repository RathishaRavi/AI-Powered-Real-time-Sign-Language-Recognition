"""
config.py
Central configuration for the AI-Powered Real-Time Sign Language Recognition System.
All paths, model hyperparameters and runtime constants live here so every other
module can import a single source of truth.
"""

import os

# ----------------------------------------------------------------------------
# BASE PATHS
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

for d in (DATASET_DIR, MODELS_DIR, SAVED_MODELS_DIR, LOGS_DIR,
          os.path.join(STATIC_DIR, "img")):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------
# DATABASE
# ----------------------------------------------------------------------------
DATABASE_PATH = os.path.join(BASE_DIR, "sign_language.db")

# ----------------------------------------------------------------------------
# MEDIAPIPE / LANDMARK SETTINGS
# ----------------------------------------------------------------------------
NUM_HAND_LANDMARKS = 21          # MediaPipe Hands landmark count
COORDS_PER_LANDMARK = 3          # x, y, z
HAND_FEATURE_SIZE = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63

# We support one or two hands. When only one hand is visible the second
# hand's 63 values are zero-padded. This keeps the model robust to
# single / double handed gestures without changing the architecture.
MAX_HANDS = 2
FULL_FEATURE_SIZE = HAND_FEATURE_SIZE * MAX_HANDS  # 126

SEQUENCE_LENGTH = 30             # frames collected per gesture sample

MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# ----------------------------------------------------------------------------
# GESTURE LABELS
# Edit this list to add / remove signs. Each label becomes a sub-folder
# inside dataset/ during collection.
# ----------------------------------------------------------------------------
GESTURES = [
    "hello", "thanks", "please", "sorry", "yes", "no",
    "i", "you", "name", "help", "good", "bad",
    "go", "home", "eat", "drink", "love", "friend",
    "stop", "more"
]

# ----------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ----------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 20  # early stopping patience

MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "sign_lstm_model.h5")
LABEL_ENCODER_FILE = os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl")
TRAINING_HISTORY_FILE = os.path.join(SAVED_MODELS_DIR, "training_history.json")
CONFUSION_MATRIX_FILE = os.path.join(SAVED_MODELS_DIR, "confusion_matrix.png")
ACCURACY_PLOT_FILE = os.path.join(SAVED_MODELS_DIR, "accuracy_plot.png")
LOSS_PLOT_FILE = os.path.join(SAVED_MODELS_DIR, "loss_plot.png")
CLASSIFICATION_REPORT_FILE = os.path.join(SAVED_MODELS_DIR, "classification_report.txt")

# ----------------------------------------------------------------------------
# PREDICTION SETTINGS
# ----------------------------------------------------------------------------
PREDICTION_CONFIDENCE_THRESHOLD = 0.70
PREDICTION_SMOOTHING_WINDOW = 10   # frames used for majority-vote smoothing
SENTENCE_WORD_GAP_SECONDS = 1.5    # gap of silence before a sentence is finalized

# ----------------------------------------------------------------------------
# EMOTION SETTINGS
# ----------------------------------------------------------------------------
EMOTION_LABELS = ["happy", "sad", "angry", "fear", "neutral", "surprise", "disgust"]
EMOTION_ANALYSIS_INTERVAL = 15   # analyze emotion every N frames (DeepFace is heavy)

# ----------------------------------------------------------------------------
# TEXT TO SPEECH
# ----------------------------------------------------------------------------
TTS_RATE = 160
TTS_VOLUME = 1.0

# ----------------------------------------------------------------------------
# FLASK
# ----------------------------------------------------------------------------
class FlaskConfig:
    SECRET_KEY = os.environ.get("SIGN_APP_SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = True
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB uploads max
